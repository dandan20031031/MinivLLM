# Fork notes

This is a fork of [Wenyueh/MinivLLM](https://github.com/Wenyueh/MinivLLM), an
educational reimplementation of vLLM. It tracks upstream `main` and adds the
changes below.

## Branches

| Branch | Contents |
|---|---|
| `main` | Unmodified mirror of upstream |
| `fix/cuda-graph-decode` | Upstream CUDA-graph fixes, submitted as [PR #105](https://github.com/Wenyueh/MinivLLM/pull/105) |
| `feat/windows-support` | Windows / sm_120 port: gloo fallback, `triton-windows`, CUDA 12.8 torch index |
| `bench/decode-profiling` | Decode profiling and batch-scaling benchmarks, plus their reports |

## What was changed

### Decode CUDA graph fixes (PR #105)

The upstream decode CUDA-graph path could not be used at all: enabling
`enforce_eager=False` raised inside `ModelRunner.__init__`. Five defects were
fixed:

1. `capture_cudagraph()` read `config['max_num_seqs']`, which `main.py` never
   defines → `KeyError`. Now falls back to `max_num_sequences`.
2. The graph output buffer was allocated with `vocab_size`, but `self.model()`
   returns the last hidden state (`hidden_size`); logits are produced
   afterwards by `compute_logits()`.
3. `graph.pool()` was called inside the `with torch.cuda.graph(...)` block, which
   raises `CUDAGraph::pool() without a preceding successful capture`.
4. Graph lookup iterated `dict.keys()`, and since graphs are registered in
   descending order (`16, 8, 4, 2, 1`), a batch of 3 matched the 16-row graph and
   computed 13 wasted rows.
5. Padding rows kept the `slot_mapping` of the previous replay, so
   `store_kvcache` could write stale KV into recycled physical blocks.

Measured on an RTX 5080 with Qwen3-0.6B (fp32 weights, 3 prompts):

| batch | before fix #4 | after fix #4 |
|---:|---:|---:|
| 3 | 425.7 tok/s | 590.1 tok/s |
| 8 | 1063.1 tok/s | 1304.6 tok/s |
| 16 | 1890.5 tok/s | 1890.8 tok/s |

Aggregate throughput scales to 1890.8 tok/s at batch 16, i.e. 3.2× over batch 3
and 5.4× over the eager baseline (110.1 tok/s).

### Windows support

- `nccl` is unavailable on Windows and is only needed for multi-GPU collectives;
  the backend now falls back to `gloo` when `sys.platform == 'win32'` or
  `world_size == 1`.
- `triton` has no Windows wheel; `triton-windows` is used instead.
- PyPI's default `torch` wheel is CPU-only on Windows, so the project pins the
  PyTorch cu128 index (required for sm_120 / RTX 50-series).
- `vllm` has no Windows build, so the dependency is restricted to non-Windows
  platforms; `benchmark_tps.py` imports it lazily and skips that comparison when
  it is absent.

### Tooling

- `benchmark_tps.py`: sweeps several batch sizes, reports aggregate and
  per-sequence throughput, takes the median of repeated runs with a cooling gap,
  and can toggle `enforce_eager` via environment variables.
- `profile_decode.py`: runs one `generate()` under `torch.profiler` and prints
  the kernels ranked by CUDA time.
- `check_model.py`: prints the model structure and parameter shapes produced by
  the real `ModelRunner` construction path.

Both scripts read the checkpoint location from `MODEL_PATH` (defaults to the
Hugging Face id `Qwen/Qwen3-0.6B`).

## Reports

- `docs/MinivLLM-Benchmark-报告.md` — end-to-end and kernel-level baselines
- `docs/MinivLLM-优化实验报告.md` — batch scaling, CUDA-graph boundary, and the
  five defects above
