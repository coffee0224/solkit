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
    "concrete",
    "register_cost",
    "OpCost",
    "list_archs",
    "load_arch",
    "Report",
    "__version__",
]


class concrete:  # noqa: N801 — used as a call-site marker: solkit.concrete(t)
    """Mark one input tensor to pass through the trace unconverted.

    ``analyze`` executes the callable on meta tensors, which carry shapes
    but no data: any read of a tensor *value* (``.item()``, ``int(t)``,
    ``bool(t)``, ``t.tolist()``) raises ``RuntimeError``. Wrap inputs whose
    contents steer control flow — varlen ``cu_seqlens``, routing indices —
    and they reach the callable as real tensors, so data-dependent loops
    and branches trace normally. They still count as external DRAM
    traffic, charged for the regions actually read.

    A concrete input must only feed value reads, never tensor ops: a real
    tensor cannot share an op with meta tensors (same-device rule).
    """

    __slots__ = ("tensor",)

    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor


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


def _unwrap_concrete(obj: Any) -> Any:
    if isinstance(obj, concrete):
        return obj.tensor
    if isinstance(obj, (list, tuple)):
        return type(obj)(_unwrap_concrete(x) for x in obj)
    if isinstance(obj, dict):
        return {k: _unwrap_concrete(v) for k, v in obj.items()}
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
            converted to meta tensors and marked as external. A tensor wrapped
            in :class:`concrete` is passed through unconverted, for code whose
            control flow reads tensor values (``.item()`` loop bounds).
        arch: bundled architecture config name (see :func:`list_archs`).
        arch_config: path to an arch YAML; overrides ``arch`` when given.

    The trace is a single eager execution on meta tensors: branching on
    tensor values raises (meta tensors carry no data — wrap those inputs in
    :class:`concrete`), and custom CUDA/Triton kernels are opaque — write
    the reference with plain torch ops.
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
        if isinstance(t, concrete):
            external.append((t.tensor, f"input:{i}"))
        elif isinstance(t, torch.Tensor):
            external.append((t, f"input:{i}"))

    counter, outputs = trace(callable_, _unwrap_concrete(meta_inputs), external=external)

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
