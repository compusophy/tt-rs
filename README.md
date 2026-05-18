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
cargo test -p tt-cuda --release -- --ignored           # GPU smoke test
cargo bench -p tt-bench                                # criterion benchmarks
```

Hardware target for v0.3/v0.4: NVIDIA Ampere or newer with ≥16 GB VRAM.
Verified development hardware: RTX 3090 (Ampere, SM 8.6, 24 GB),
CUDA Toolkit 11.7 on Windows.

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
