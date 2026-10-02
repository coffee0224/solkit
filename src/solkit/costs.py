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

import string
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
# traffic: a view op executed as its own kernel would move no bytes. Matched
# by base name, overload-insensitive: indexing dispatches as
# aten.select.int / aten.slice.Tensor, .transpose(-1, -2) as
# aten.transpose.int, t.size(0) as aten.size.int — never the .default forms
# that a full-name match would catch.
VIEW_OPS = frozenset(
    "aten." + n
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


def is_view_op(name: str) -> bool:
    return name.rsplit(".", 1)[0] in VIEW_OPS


# Ops that never reach the GPU as kernels or carry no memory traffic.
# Matched by base name like VIEW_OPS (aten.size.int, aten.stride.int, ...).
# NOTE: `_local_scalar_dense` (tensor.item()) is deliberately NOT skipped:
# on concrete inputs it is the recorded evidence that a value was read from
# an external storage (its 8 B region feeds the fused model), at the cost of
# one table row per .item() call.
SKIP_OPS = frozenset(
    "aten." + n
    for n in [
        "size",
        "dim",
        "numel",
        "stride",
        "is_contiguous",
        "is_floating_point",
        "empty",
        "empty_strided",
        "scalar_tensor",
        "lift_fresh",
    ]
)


def is_skip_op(name: str) -> bool:
    return name.rsplit(".", 1)[0] in SKIP_OPS


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


def einsum_cost(equation: str, operands: Sequence[torch.Tensor]) -> OpCost:
    """Analytic cost of ``torch.einsum(equation, *operands)`` from the equation.

    eager lowers einsum at dispatch time (permute/reshape/bmm — or a plain
    broadcast mul when the summed dim has size 1), so the equation itself
    never reaches ``__torch_dispatch__``; this charges the contraction the
    *algorithm* performs instead:

    - two or more operands with at least one summed index (present in the
      inputs, absent from the output): ``MACs = prod(output dims) *
      prod(summed dims)`` at the promoted input dtype. Degenerate
      contractions (summed dim of size 1 — rank-1 outer updates, the GDN
      delta-rule pattern) are charged as contractions: the SOL denominator
      assumes a fused kernel can batch them into real GEMMs, which a chunked
      formulation does.
    - two or more operands, no summed index: elementwise product.
    - one operand with a summed index: reduction (adds, no MACs).
    - one operand, no summed index: permutation/diagonal — free.

    Raises ``ValueError`` on anything not parsed exactly (repeated labels in
    one operand, mismatched ellipsis ranks, label size conflicts); callers
    should then fall back to decomposition accounting with a warning.
    """
    eq = equation.replace(" ", "")
    if eq.count("->") > 1:
        raise ValueError("multiple '->'")
    if "->" in eq:
        in_part, out_part = eq.split("->")
    else:
        in_part, out_part = eq, None
    specs = in_part.split(",")
    if len(specs) != len(operands):
        raise ValueError("operand count mismatch")

    used = {c for c in eq if c.isalpha()}
    ell_ranks = {t.ndim - (len(s) - 3) for s, t in zip(specs, operands) if "..." in s}
    if len(ell_ranks) > 1:
        raise ValueError("mismatched ellipsis ndim")
    n_ell = ell_ranks.pop() if ell_ranks else 0
    pool = [c for c in string.ascii_letters if c not in used]
    if n_ell > len(pool):
        raise ValueError("too many ellipsis dims for the unused letters")

    sizes: Dict[str, int] = {}
    op_labels: list = []
    for spec, t in zip(specs, operands):
        if spec.count("...") > 1:
            raise ValueError("multiple ellipses in one operand")
        i = spec.find("...")
        labels = list(spec) if i < 0 else list(spec[:i]) + pool[:n_ell] + list(spec[i + 3 :])
        if len(labels) != t.ndim:
            raise ValueError(f"subscripts {spec!r} do not match dim {t.ndim}")
        if len(set(labels)) != len(labels):
            raise ValueError(f"repeated label in {spec!r} (diagonal) not supported")
        op_labels.append(labels)
        for lbl, d in zip(labels, t.shape):
            n = int(d)
            prev = sizes.get(lbl)
            if prev is not None and prev != n and prev != 1 and n != 1:
                raise ValueError(f"label {lbl!r} size conflict {prev} vs {n}")
            sizes[lbl] = max(prev or n, n)

    if out_part is None:
        counts: Dict[str, int] = {}
        for labels in op_labels:
            for lbl in labels:
                counts[lbl] = counts.get(lbl, 0) + 1
        out_labels = sorted(lb for lb, c in counts.items() if c == 1)
    elif "..." in out_part:
        i = out_part.find("...")
        out_labels = list(out_part[:i]) + pool[:n_ell] + list(out_part[i + 3 :])
    else:
        out_labels = list(out_part)
    if any(lb not in sizes for lb in out_labels):
        raise ValueError("output label missing from inputs")

    out_numel = _prod(sizes[lb] for lb in out_labels)
    summed = [sizes[lb] for lb in sizes if lb not in set(out_labels)]
    if len(operands) >= 2 and summed:
        dt = operands[0].dtype
        for t in operands[1:]:
            dt = torch.promote_types(dt, t.dtype)
        return OpCost(macs=out_numel * _prod(summed), mac_dtype=dt)
    if len(operands) >= 2:
        return OpCost(other_ops=out_numel)  # elementwise product
    if summed:
        return OpCost(other_ops=out_numel * _prod(summed))  # reduction
    return OpCost()  # permutation / diagonal: materialized copy, no ALU


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
