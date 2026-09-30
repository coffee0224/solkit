"""solkit: speed-of-light analysis for PyTorch reference implementations.

Trace any callable or ``nn.Module`` on meta tensors, count MACs and DRAM
traffic per op (mixed-precision exact), and evaluate a two-resource roofline
against an architecture config::

    import solkit

    report = solkit.analyze(model, *inputs, arch="RTX_5060_Ti")
    print(report.summary())
    print(f"SOL = {report.sol_ms * 1e3:.1f} us, {report.bottleneck}-bound")
"""

from __future__ import annotations

import copy
from typing import Any, List, Optional, Tuple

import torch
from torch.utils._pytree import tree_flatten

from .bytes_model import compute_bytes, fused_breakdown, macs_by_dtype
from .costs import OpCost, register_cost
from .counter import trace
from .report import Report
from .roofline import load_arch, solve_roofline

__version__ = "0.1.0"

__all__ = [
    "analyze",
    "register_cost",
    "OpCost",
    "list_archs",
    "load_arch",
    "Report",
    "__version__",
]


def list_archs() -> List[str]:
    """Names of the bundled architecture configs."""
    from .roofline import list_archs as _la

    return _la()


def _to_meta(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        if obj.device.type == "meta":
            return obj
        return torch.empty(obj.shape, dtype=obj.dtype, device="meta")
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_meta(x) for x in obj)
    if isinstance(obj, dict):
        return {k: _to_meta(v) for k, v in obj.items()}
    return obj


def _external_of_module(module: torch.nn.Module) -> List[Tuple[torch.Tensor, str]]:
    out = []
    for name, p in module.named_parameters(recurse=True):
        out.append((p, f"param:{name}"))
    for name, b in module.named_buffers(recurse=True):
        out.append((b, f"buffer:{name}"))
    return out


def analyze(
    fn: Any,
    *inputs: Any,
    arch: str = "RTX_5060_Ti",
    arch_config: Optional[str] = None,
) -> Report:
    """Analyze one forward pass of ``fn`` and return its SOL report.

    Args:
        fn: a callable or ``nn.Module``. Modules are deep-copied and moved to
            the meta device (weights are not read); their parameters/buffers
            are marked as external DRAM traffic.
        inputs: tensors (any device) or nested containers of tensors; they are
            converted to meta tensors and marked as external.
        arch: bundled architecture config name (see :func:`list_archs`).
        arch_config: path to an arch YAML; overrides ``arch`` when given.

    The trace is a single eager execution on meta tensors: data-dependent
    control flow follows the branch taken by the (uninitialized) values, and
    custom CUDA/Triton kernels are opaque — write the reference with plain
    torch ops.
    """
    cfg = load_arch(arch_config or arch)
    arch_name = str(cfg.get("name", arch_config or arch))

    external: List[Tuple[torch.Tensor, str]] = []
    if isinstance(fn, torch.nn.Module):
        module = copy.deepcopy(fn).to_empty(device="meta")
        module.eval()
        external.extend(_external_of_module(module))
        callable_ = module
    else:
        callable_ = fn

    meta_inputs = _to_meta(inputs)
    for i, t in enumerate(tree_flatten(meta_inputs)[0]):
        if isinstance(t, torch.Tensor):
            external.append((t, f"input:{i}"))

    counter, outputs = trace(callable_, meta_inputs, external=external)

    mbd = macs_by_dtype(counter)
    bytes_model = compute_bytes(counter, outputs)
    roofline = solve_roofline(cfg, mbd, bytes_model.unfused_bytes, bytes_model.fused_bytes)

    return Report(
        arch_name=arch_name,
        macs_by_dtype=mbd,
        total_macs=sum(mbd.values()),
        total_other_ops=sum(r.other_ops for r in counter.op_records),
        bytes=bytes_model,
        breakdown=fused_breakdown(counter, outputs),
        roofline=roofline,
        op_records=counter.op_records,
        warnings=counter.warnings,
    )
