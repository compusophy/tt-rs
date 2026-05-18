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

**v0.1 (this commit)**
- `tt-core` complete: TT-SVD decomposition with rank+tolerance truncation,
  exact reconstruction, single-element evaluation, in-place TT-norm,
  re-rounding. All operations CPU, f64 SVD with f32 storage.
- 5 unit tests prove correctness on random tensors, exact rank-1
  factorizations, single-element lookups, ‖T‖², and rank truncation.
- `tt-bench` (criterion harness) for decomposition and storage ratio.
- `tt-cuda` scaffold; kernels land in v0.2.

**v0.2 (planned)**
- TT-aware matvec on CPU (no dense reconstruction)
- CUDA TT-matvec kernel via `cudarc`, fp16 with tensor-core wmma
- Benchmark vs `tntorch` (target: 50× on `from_dense`, 100× on `round`)
- PyO3 bindings (`tt-py`) so it drops into HF Transformers pipelines

**v0.3 (the ambition)**
- Fused `tt_attention_fwd` CUDA kernel
- KV cache replacement in a `candle`-based Llama-3 inference loop
- Headline benchmark vs FlashAttention-3 on the (memory, throughput,
  RULER accuracy) Pareto frontier
- Offline + online policy for per-layer rank selection

## Run

```sh
cargo test --workspace --release      # unit tests across the workspace
cargo bench -p tt-bench               # decomposition + storage benchmarks
```

Hardware target for v0.2/0.3: NVIDIA Ampere or newer with ≥16 GB VRAM.
Verified development hardware: RTX 3090 (Ampere, SM 8.6, 24 GB).

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
