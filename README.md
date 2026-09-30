# solkit

Speed-of-light (SOL) analysis for PyTorch reference implementations — a clean,
dependency-light rebuild of the core of [NVlabs/SOLAR](https://github.com/NVlabs/SOLAR).

Give it any eager torch code (a callable, an `nn.Module`, or a SOL-Bench /
KernelBench file) and it traces one forward pass on meta tensors, counts MACs
and DRAM traffic per op — mixed-precision exact — and evaluates a two-resource
roofline against an architecture config:

```python
import solkit

report = solkit.analyze(model, *inputs, arch="RTX_5060_Ti")
print(report.summary())
print(f"SOL = {report.sol_ms * 1e3:.1f} us ({report.bottleneck}-bound)")
```

```
$ solkit analyze examples/gqa_bench.py --arch RTX_5060_Ti --per-op
```

## Why

SOLAR pioneered this analysis but installs heavily: it requires a patched
`torchview` (upstream commit + two source patches), and pulls `openai`,
`pytest`, `matplotlib`, `pandas`, `huggingface-hub`... as hard runtime
dependencies. solkit keeps the core idea with **two dependencies: torch and
pyyaml**:

- No patch dependencies — tracing uses in-tree PyTorch APIs
  (`TorchDispatchMode` on meta tensors), never torchview.
- Mixed-precision exact — every tensor is charged at its own dtype width and
  every contraction runs on the tensor-core pipe of its own dtype (int8 MACs
  at int8 rate, fp16 bytes at fp16 width). SOLAR applies one precision to the
  whole graph.
- Causal-exact SDPA — `scaled_dot_product_attention` is intercepted at the
  Python API level and charged analytically (on meta tensors it would
  otherwise decompose into two full bmms and over-count causal attention
  ~2x).
- No LLM-generated handlers — unknown ops warn loudly and are countable via a
  public registry; nothing is ever guessed silently.

## Cost model

Two DRAM traffic models bound the same compute work:

- **unfused** — every op as its own kernel, all tensor reads/writes cross
  DRAM. What a naive one-op-per-kernel execution costs.
- **fused** (the SOL denominator) — only external traffic: parameters and
  call inputs read from DRAM (each element at most once, deduplicated by
  storage and access region — overlapping slices count once), outputs written
  back once. Intermediates are assumed to stay on-chip. This is optimistic
  when the working set exceeds L2; treat it as a speed limit, not a promise.

View ops (`view`, `permute`, `transpose`, slices, ...) move no data and are
free in both models. Elementwise/reduction ops contribute `other_ops`
(informational) and bytes, but not SOL compute — like SOLAR, on the grounds
that elementwise work is memory-bound in practice.

```
compute_cycles = Σ_dtype  MACs_dtype / MAC_per_cycle_dtype
memory_cycles  = bytes / DRAM_byte_per_cycle
SOL            = max(compute, memory) / freq
```

## Arch configs

`solkit/configs/arch/*.yaml` use the SOLAR schema (`freq_GHz`,
`DRAM_byte_per_cycle`, `MAC_per_cycle_<prec>_tc`, ...) with two differences:

- rates are **dense** peaks (SOLAR's H100 file uses sparse numbers, 2x dense);
- an optional `measured:` block overrides theoretical values — put your
  ncu-calibrated numbers there instead of editing datasheet values.

To derive values for a new GPU: capacities and SM count come from
`torch.cuda.get_device_properties`; DRAM/L2 bandwidth from ncu by dividing an
absolute rate by its `pct_of_peak_sustained` (e.g. `dram__bytes.sum.per_second
/ dram__throughput.avg.pct_of_peak_sustained_elapsed`); MAC rates from the
per-SM throughput table for your compute capability × SM count, cross-checked
with a pipe-saturated GEMM (`sm__pipe_tensor_cycles_active` pct back-calc).

Pass your own config with `arch_config="path/to/arch.yaml"`.

## Extending

Register a cost for any aten op (or override a built-in):

```python
@solkit.register_cost("aten::linalg_cross.default")
def _(tensors, kwargs, outs):
    return solkit.OpCost(macs=3 * outs[0].numel(), mac_dtype=tensors[0].dtype)
```

## Limitations

- Eager only: ops inside `torch.compile` regions don't dispatch and are
  invisible. Write the reference with plain torch ops.
- Custom CUDA/Triton kernels are opaque black boxes (same as SOLAR).
- Data-dependent control flow follows the single traced branch (same as
  SOLAR; values are uninitialized meta data).
- The roofline models two resources (tensor-core compute, DRAM bandwidth) —
  no L2 bandwidth, occupancy, or latency effects.

## Install

```
pip install solkit        # once published
```

From source:

```
uv venv && uv sync        # or: pip install -e .
pytest                    # CPU-only; no GPU needed
```

## License

Apache-2.0. The cost-model semantics and arch-config schema are informed by
NVlabs/SOLAR (Apache-2.0); no code is derived from it.
