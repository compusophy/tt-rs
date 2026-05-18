# tt-rs

A tensor-train (TT) decomposition library, written in Rust, with custom CUDA
kernels for the operations that matter for LLM inference.

The goal is a fused, on-device **TT-attention** kernel that lets a Llama-class
model hold a 256k-token context on a single 24 GB consumer GPU — by storing
the attention KV cache in TT format and decompressing the relevant slice
inside the attention kernel rather than materializing it.

```
┌─────────── tt-core ───────────┐    ┌─────────── tt-cuda ──────────┐
│ TtTensor (CPU)                │    │ CudaTtTensor (device fp16)   │
│ ├─ from_dense  (TT-SVD)       ├──►│ ├─ TT-matvec kernel           │
│ ├─ to_dense    (reconstruct)  │    │ ├─ TT-round kernel            │
│ ├─ eval        (one element)  │    │ └─ tt_attention_fwd kernel    │
│ ├─ norm_sq     (no dense)     │    └──────────────┬─────────────────┘
│ └─ round       (re-truncate)  │                   │
└───────────────────────────────┘    ┌──────────────▼─────────────────┐
                                     │ policy/                         │
                                     │ ├─ offline BO over rank profile │
                                     │ └─ online bandit per head       │
                                     └────────────────────────────────┘
```

## Why

The KV cache is the hard memory bottleneck for long-context LLM inference.
A 70B model at 128k context burns ~40 GB **just for KV**. TT decomposition
fits this case unusually well: KV tensors have low effective rank, the
compression ratio is tunable per layer, and TT compression is *orthogonal*
to integer quantization — they stack.

The library underneath is also a clean Rust replacement for `tntorch` /
`t3f` / `ttpy` — pure-Python NumPy projects with small academic audiences.
A 10–100× speedup on decomposition and rounding is the v0.1 baseline win;
the CUDA TT-attention kernel is the v0.2 ambition.

## Status

**v0.2 (this commit)**
- `tt-core`: `TtTensor` (TT-SVD, to_dense, eval, in-place norm_sq, round)
  AND `TtMatrix` (interleaved (m_1, n_1, …, m_d, n_d) layout, TT-aware
  matvec that contracts mode-by-mode — no dense reconstruction).
- 9 unit tests cover roundtrip, exact rank-1 factorization, element eval,
  ‖T‖² without dense, rank truncation, identity matvec, rank-1 matvec,
  random small matvec, and compression-ratio sanity.
- `tt-cuda`: cudarc 0.13 + CUDA 11.7 toolchain verified end-to-end on the
  RTX 3090 — NVRTC compiles a kernel, launch dispatches, device→host copy
  matches CPU bit-for-bit on a 64k-float vector_add. `build.rs` handles
  the Windows NVRTC DLL ABI-version naming so `cargo test` just works.
- `tt-bench`: criterion harness for decomposition + element-eval timing.

**v0.3 (planned)**
- CUDA TT-matvec kernel (fp16, tensor cores via `wmma`)
- Benchmark vs `tntorch` (target: 50× on `from_dense`, 100× on `round`)
- PyO3 bindings (`tt-py`) so it drops into HF Transformers pipelines

**v0.4 (the ambition)**
- Fused `tt_attention_fwd` CUDA kernel
- KV cache replacement in a `candle`-based Llama-3 inference loop
- Headline benchmark vs FlashAttention-3 on the (memory, throughput,
  RULER accuracy) Pareto frontier
- Offline + online policy for per-layer rank selection

## Run

```sh
cargo test --workspace --release                       # CPU unit tests
cargo test -p tt-cuda --release -- --ignored           # GPU smoke + tt_eval + tt_matvec
cargo run -p tt-bench --release --bin matvec-numbers   # the table below
```

Hardware target for v0.3/v0.4: NVIDIA Ampere or newer with ≥16 GB VRAM.
Verified development hardware: RTX 3090 (Ampere, SM 8.6, 24 GB),
CUDA Toolkit 11.7 on Windows.

## Numbers (RTX 3090, v0.2)

A square `rows × cols` low-rank matrix is decomposed into a TT-matrix of
depth `d` (uniform mode size 4), then multiplied by a random vector. All
four paths produce bit-close outputs (`< 1e-3`).

| rows  | cols  | d | rank | compression | dense_ms | cpu_tt_ms | gpu_naive_ms | **gpu_smart_ms** |
|------:|------:|--:|-----:|------------:|---------:|----------:|-------------:|-----------------:|
|    16 |    16 | 2 |    4 |       3.2 × |    0.000 |     0.011 |        0.078 |        0.075 |
|    64 |    64 | 3 |    8 |      19.7 × |    0.002 |     0.024 |        0.538 |        0.282 |
|   256 |   256 | 4 |   16 |     123.2 × |    0.051 |     0.221 |        2.946 |    **0.195** |
|  1024 |  1024 | 5 |   16 |     136.3 × |    1.102 |    13.338 |       58.356 |    **0.568** |
|  4096 |  4096 | 6 |   16 |     700.9 × |   15.048 |   163.284 |      611.770 |    **0.315** |
| 16384 | 16384 | 7 |   16 |  **8244 ×** |  329.379 |   957.102 |     3499.878 |    **0.983** |

**Two headlines.**

**1. Memory.** At 16384×16384, the dense matrix is 1 GB; the TT
factorisation is ~130 KB — **8244× smaller**. This is the property
that lets a TT-compressed KV cache fit a 256k-token context into a
24 GB consumer GPU (the v0.4 target).

**2. Speed.** The smart kernel walks the TT cores mode-by-mode (two
small kernels — `row_contract` + `col_contract` — dispatched in
sequence) instead of the naive "one thread per output element" approach.
The result on the 3090:

- **47× faster than dense matvec at 4096×4096** (0.32 ms vs 15.0 ms)
- **335× faster than dense matvec at 16384×16384** (0.98 ms vs 329 ms)

At those sizes we're memory-bandwidth bound on a few-KB working
buffer rather than compute bound on a million-FLOP matrix. The
work scales as O(d · r² · max_mode · ∏others), not O(M · N).

The naive kernel is left in the binary for the table — both
correctness tests still pass against it.

**Coming.** The host→device transfer of x and the device→host transfer
of y are *included* in the smart timings, but the v0.2 API is host-in /
host-out. A device-resident API in v0.3 will drop the transfer overhead
entirely and is the natural foundation for the fused TT-attention kernel
(v0.4) that will compete with FlashAttention-3 on the (memory,
throughput, RULER accuracy) Pareto frontier.

## Matrices the size of small datasets

Building a TT-matrix doesn't require materialising a dense intermediate.
`TtMatrix::from_cores` accepts cores directly, which lets us construct
matrices whose dense form is physically impossible. Same `gpu_smart_ms`
kernel as above; same RTX 3090 (24 GB).

| rows    | cols    | d  | rank | dense size  | tt size  | cpu_tt_ms | gpu_smart_ms |     CPU↔GPU diff |
|--------:|--------:|---:|-----:|------------:|---------:|----------:|-------------:|-----------------:|
|  65 536 |  65 536 |  8 |   16 |     16.0 GB |  56.5 KB |     1 764 |          3.6 |          2.7e-8 |
| 262 144 | 262 144 |  9 |   16 |    256.0 GB |  64.5 KB |     8 300 |         32.8 |          1.9e-8 |
| 1 048 576 | 1 048 576 | 10 | 16 | **4 096.0 GB** | **72.5 KB** | 38 274 | **121** | **1.8e-8** |

The bottom row: a **1 048 576 × 1 048 576 matrix** (over 10¹² entries,
**4 TB dense**) collapses into **72.5 KB** of TT cores, and `y = M x`
runs in **121 ms** on a single consumer GPU — bit-equivalent to the
CPU result within f32 round-off (1.8 × 10⁻⁸). The dense form literally
cannot exist on any single computer; the TT form is smaller than this
README.

This is the property — not the speedup ratio — that the v0.4 fused
TT-attention kernel is built on. A long-context KV cache lives in the
"dense doesn't fit but TT does" regime by default.

## Layout

```
crates/
├─ tt-core/      Pure Rust TT operations (no GPU). The substrate.
├─ tt-cuda/      cudarc-based GPU kernels. v0.2 home.
└─ tt-bench/     Criterion benchmark harness + baselines.
kernels/        Hand-rolled .cu files (v0.2: tt_matvec.cu, tt_attention.cu).
policy/         Per-layer rank selection — offline BO + online bandits (v0.3).
```

## License

Apache-2.0. See `LICENSE`.
