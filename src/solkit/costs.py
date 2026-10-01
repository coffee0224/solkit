"""Per-op cost formulas.

Every formula is a pure function of tensor shapes/dtypes — no data is touched,
so counting runs entirely on ``device="meta"`` tensors. A formula returns an
:class:`OpCost`; ops without a registered formula fall back to elementwise
accounting (output elements counted as ``other_ops``, zero MACs).

Formulas receive the op's positional ``args`` tuple unflattened (so schema
positions like ``groups`` or ``transposed`` stay addressable), its ``kwargs``,
and the flattened output tensors.

The public extension point is :func:`register_cost`::

    @solkit.register_cost("aten::linalg_cross.default")
    def _(args, kwargs, outs):
        return solkit.OpCost(macs=3 * outs[0].numel(), mac_dtype=args[0].dtype)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence

import torch


@dataclass
class OpCost:
    """Cost of a single op invocation.

    macs:      multiply-accumulate count (tensor-core work if mac_dtype set)
    mac_dtype: dtype governing which tensor-core pipe the MACs run on; None
               means the op does no contraction work.
    other_ops: non-contraction ALU ops (informational; excluded from SOL
               compute, mirroring the rationale that elementwise work is
               memory-bound in practice).
    """

    macs: int = 0
    mac_dtype: Optional[torch.dtype] = None
    other_ops: int = 0


CostFn = Callable[[tuple, dict, Sequence[torch.Tensor]], OpCost]

COST_FUNCS: Dict[str, CostFn] = {}

# Ops that only manipulate metadata (views/aliases). Zero compute, zero DRAM
# traffic: a view op executed as its own kernel would move no bytes.
VIEW_OPS = frozenset(
    f"aten.{n}.default"
    for n in [
        "view",
        "reshape",
        "_reshape_alias",
        "reshape_as",
        "permute",
        "transpose",
        "t",
        "expand",
        "expand_as",
        "unsqueeze",
        "squeeze",
        "flatten",
        "unflatten",
        "unfold",
        "narrow",
        "select",
        "slice",
        "detach",
        "_unsafe_view",
        "alias",
        "unbind",
        "split",
        "split_with_sizes",
        "tensor_split",
        "chunk",
        "swapdims",
        "swapaxes",
        "movedim",
        "diagonal",
        "view_as",
    ]
)

# Ops that never reach the GPU as kernels or carry no memory traffic.
# NOTE: `_local_scalar_dense` (tensor.item()) is deliberately NOT skipped:
# on concrete inputs it is the recorded evidence that a value was read from
# an external storage (its 8 B region feeds the fused model), at the cost of
# one table row per .item() call.
SKIP_OPS = frozenset(
    f"aten.{n}.default"
    for n in [
        "size",
        "dim",
        "numel",
        "stride",
        "is_contiguous",
        "is_floating_point",
        "empty.memory_format",
        "empty_strided",
        "scalar_tensor",
        "lift_fresh",
    ]
)


def norm_op(name: str) -> str:
    """Normalize an aten op name to the dispatch form ``aten.<op>.<overload>``.

    torch has printed ``OpOverload`` both as ``aten::mm.default`` and (in
    recent versions) ``aten.mm.default``; accept either everywhere.
    """
    return name.replace("::", ".", 1)


def register_cost(*op_names: str) -> Callable[[CostFn], CostFn]:
    """Register a cost formula for one or more aten op names.

    Op names use the ``aten.mm.default`` form (``aten::mm.default`` is
    accepted and normalized). A later registration replaces an earlier one,
    so users can override built-in formulas.
    """

    def deco(fn: CostFn) -> CostFn:
        for name in op_names:
            COST_FUNCS[norm_op(name)] = fn
        return fn

    return deco


def _prod(xs: Sequence[int]) -> int:
    out = 1
    for x in xs:
        out *= x
    return out


def _batched_macs(a: torch.Tensor, b: torch.Tensor) -> int:
    """MACs of a batched matmul with broadcast batch dims: (...,M,K)@(...,K,N)."""
    m, k = a.shape[-2], a.shape[-1]
    n = b.shape[-1]
    batch = 1
    for da, db in zip(reversed(a.shape[:-2]), reversed(b.shape[:-2])):
        batch *= max(da, db)
    return m * k * n * batch


def sdpa_macs(q: torch.Tensor, k: torch.Tensor, is_causal: bool) -> int:
    """MACs of scaled_dot_product_attention, causal-exact.

    Two contractions: ``q @ k^T`` and ``attn @ v``, each costing
    ``batch * T_q * T_k * d``. Causal semantics follow eager
    ``is_causal=True`` (top-left aligned): query row i attends to
    ``min(i + 1, T_k)`` keys. Flash kernels with a bottom-right alignment
    (KV-cache decode) see fewer keys; the eager reference is the ground
    truth for what the algorithm computes.
    """
    tq, d = q.shape[-2], q.shape[-1]
    tk = k.shape[-2]
    if is_causal:
        if tq <= tk:
            visible = tq * (tq + 1) // 2
        else:
            visible = tk * (tk + 1) // 2 + (tq - tk) * tk
    else:
        visible = tq * tk
    batch = _prod(q.shape[:-2])  # q carries the fully-expanded batch dims
    return 2 * batch * visible * d


def _mm_cost(args, kwargs, outs):
    a, b = args[0], args[1]
    return OpCost(macs=a.shape[0] * a.shape[1] * b.shape[1], mac_dtype=a.dtype)


def _addmm_cost(args, kwargs, outs):
    # addmm/addbmm-style: (bias, mat1, mat2); the bias contributes no MACs.
    a, b = args[1], args[2]
    return OpCost(macs=_batched_macs(a, b), mac_dtype=a.dtype)


def _bmm_cost(args, kwargs, outs):
    a, b = args[0], args[1]
    return OpCost(macs=_batched_macs(a, b), mac_dtype=a.dtype)


def _matmul_cost(args, kwargs, outs):
    a, b = args[0], args[1]
    return OpCost(macs=_batched_macs(a, b), mac_dtype=a.dtype)


def _int8_mm_cost(args, kwargs, outs):
    a, b = args[0], args[1]
    return OpCost(macs=a.shape[0] * a.shape[1] * b.shape[1], mac_dtype=torch.int8)


def _scaled_mm_cost(args, kwargs, outs):
    a, b = args[0], args[1]
    return OpCost(macs=a.shape[0] * a.shape[1] * b.shape[1], mac_dtype=a.dtype)


def _convolution_cost(args, kwargs, outs):
    # aten.convolution(input, weight, bias?, stride, padding, dilation,
    #                 transposed, output_padding, groups) — convNd lowers here.
    x, w = args[0], args[1]
    transposed = args[6]
    if transposed:
        # weight (in_ch, out_ch/groups, *kernel): each input element lands in
        # out_ch/groups * kernel output positions.
        macs = _prod(x.shape) * w.shape[1] * _prod(w.shape[2:])
    else:
        # weight (out_ch, in_ch/groups, *kernel): each output element costs
        # in_ch/groups * kernel MACs.
        macs = _prod(outs[0].shape) * w.shape[1] * _prod(w.shape[2:])
    return OpCost(macs=macs, mac_dtype=x.dtype)


def _convnd_cost(args, kwargs, outs):
    # aten.convNd(input, weight, bias?, stride, padding, dilation, groups)
    x, w = args[0], args[1]
    macs = _prod(outs[0].shape) * w.shape[1] * _prod(w.shape[2:])
    return OpCost(macs=macs, mac_dtype=x.dtype)


def _conv_tnd_cost(args, kwargs, outs):
    # aten.conv_transposeNd: weight (in_ch, out_ch/groups, *kernel); charge
    # per input element (defensive — usually lowers to aten.convolution).
    x, w = args[0], args[1]
    macs = _prod(x.shape) * w.shape[1] * _prod(w.shape[2:])
    return OpCost(macs=macs, mac_dtype=x.dtype)


register_cost("aten::mm.default")(_mm_cost)
register_cost("aten::bmm.default")(_bmm_cost)
register_cost("aten::baddbmm.default")(_addmm_cost)
register_cost("aten::addmm.default")(_addmm_cost)
register_cost("aten::addbmm.default")(_addmm_cost)
register_cost("aten::matmul.default")(_matmul_cost)
register_cost("aten::convolution.default")(_convolution_cost)
register_cost("aten::conv1d.default", "aten::conv2d.default", "aten::conv3d.default")(_convnd_cost)
register_cost(
    "aten::conv_transpose1d.default",
    "aten::conv_transpose2d.default",
    "aten::conv_transpose3d.default",
)(_conv_tnd_cost)
register_cost("aten::_int_mm.default")(_int8_mm_cost)
register_cost("aten::_scaled_mm.default")(_scaled_mm_cost)

# Substrings that suggest a contraction variant we have no formula for; ops
# matching these trigger a warning instead of silently counting as elementwise.
_CONTRACTION_HINTS = ("mm", "matmul", "conv", "einsum", "attention", "dot")


def looks_like_contraction(op_name: str) -> bool:
    return any(h in op_name for h in _CONTRACTION_HINTS)
