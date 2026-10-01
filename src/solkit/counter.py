"""Dispatch-level op counting.

:class:`SOLCounter` is a ``TorchDispatchMode`` that records every aten op
executed under it, together with the shapes/dtypes/storages of the tensors
involved. Tracing runs on ``device="meta"`` tensors so no memory is allocated
and no kernel launches.

Because composite aten ops decompose before ``__torch_dispatch__``
(``F.linear`` arrives as ``aten::addmm``, ``a @ b`` as ``aten::mm``, and —
crucially — ``scaled_dot_product_attention`` arrives as its math fallback of
two full-size ``bmm``s, which over-counts causal attention by ~2x), an outer
:class:`_FnHook` (``TorchFunctionMode``) intercepts SDPA at the Python API
level, charges it analytically (causal-exact) and suppresses the dispatch
accounting of its decomposition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from .costs import (
    COST_FUNCS,
    OpCost,
    is_skip_op,
    is_view_op,
    looks_like_contraction,
    norm_op,
    sdpa_macs,
)

SDPA_OP = "sdpa.analytic"


@dataclass
class StorageRef:
    """A DRAM-resident (external) buffer marked before tracing.

    ``keep`` holds a strong reference so ``id(storage)`` stays valid for the
    whole trace — meta storages report ``data_ptr() == 0``, so Python object
    identity is the only usable key.
    """

    label: str
    nbytes: int
    keep: Any = None


@dataclass
class ReadRegion:
    """Bytes of an external storage touched by one op, as (start, end)."""

    storage_id: int
    start: int
    end: int


@dataclass
class OpRecord:
    name: str
    in_shapes: List[str] = field(default_factory=list)
    out_shapes: List[str] = field(default_factory=list)
    macs: int = 0
    mac_dtype: Optional[torch.dtype] = None
    other_ops: int = 0
    in_bytes: int = 0
    out_bytes: int = 0
    reads: List[ReadRegion] = field(default_factory=list)


def _flat_tensors(obj: Any) -> List[torch.Tensor]:
    return [t for t in tree_flatten(obj)[0] if isinstance(t, torch.Tensor)]


def _tbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def _fmt_shape(t: torch.Tensor) -> str:
    dt = str(t.dtype).removeprefix("torch.")
    return f"{tuple(t.shape)}{dt}"


class SOLCounter(TorchDispatchMode):
    """Record every aten op with cost and memory-footprint bookkeeping."""

    def __init__(self) -> None:
        self.op_records: List[OpRecord] = []
        self.external: Dict[int, StorageRef] = {}
        self.warnings: List[str] = []
        self._suppress = 0  # >0 while inside an analytically-charged op
        self._storage_keepalive: Dict[int, Any] = {}

    # -- provenance -------------------------------------------------------

    def mark_external(self, tensor: torch.Tensor, label: str) -> None:
        st = tensor.untyped_storage()
        sid = id(st)
        self._storage_keepalive[sid] = st
        if sid not in self.external:
            self.external[sid] = StorageRef(label=label, nbytes=st.nbytes(), keep=st)

    # -- recording --------------------------------------------------------

    def _record(
        self,
        name: str,
        in_tensors: List[torch.Tensor],
        out_tensors: List[torch.Tensor],
        cost: OpCost,
    ) -> None:
        rec = OpRecord(
            name=name,
            in_shapes=[_fmt_shape(t) for t in in_tensors[:4]],
            out_shapes=[_fmt_shape(t) for t in out_tensors[:4]],
            macs=cost.macs,
            mac_dtype=cost.mac_dtype,
            other_ops=cost.other_ops,
        )
        if not is_view_op(name):
            rec.in_bytes = sum(_tbytes(t) for t in in_tensors)
            rec.out_bytes = sum(_tbytes(t) for t in out_tensors)
        for t in in_tensors:
            sid = id(t.untyped_storage())
            if sid in self.external:
                es = t.element_size()
                rec.reads.append(
                    ReadRegion(
                        storage_id=sid,
                        start=t.storage_offset() * es,
                        end=t.storage_offset() * es + _tbytes(t),
                    )
                )
        self.op_records.append(rec)

    def _default_cost(self, name: str, outs: List[torch.Tensor]) -> OpCost:
        if looks_like_contraction(name) and "sdpa" not in name:
            self.warnings.append(
                f"op '{name}' looks like a contraction but has no cost formula; "
                f"counted as elementwise (0 MACs). Register one with "
                f"@solkit.register_cost('{name}')."
            )
        return OpCost(other_ops=sum(t.numel() for t in outs))

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        name = norm_op(str(func))
        if self._suppress == 0 and not is_skip_op(name):
            tensors = _flat_tensors(args) + _flat_tensors(kwargs)
            outs = _flat_tensors(out)
            if is_view_op(name):
                cost = OpCost()  # pure metadata: no compute, no traffic
            else:
                cost_fn = COST_FUNCS.get(name)
                # Formulas get the unflattened args so schema positions
                # (groups, transposed, ...) stay addressable.
                cost = cost_fn(args, kwargs, outs) if cost_fn else self._default_cost(name, outs)
            self._record(name, tensors, outs, cost)
        return out

    # -- analytical SDPA ----------------------------------------------------

    def record_sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
        is_causal: bool,
        out: torch.Tensor,
    ) -> None:
        cost = OpCost(macs=sdpa_macs(q, k, is_causal), mac_dtype=q.dtype)
        in_tensors = [q, k, v] + ([mask] if mask is not None else [])
        self._record(SDPA_OP, in_tensors, [out], cost)


class _FnHook(TorchFunctionMode):
    """Outer mode: charge SDPA analytically, suppress its decomposition."""

    def __init__(self, counter: SOLCounter) -> None:
        self.counter = counter

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is F.scaled_dot_product_attention:
            q, k, v = args[0], args[1], args[2]
            mask = args[3] if len(args) > 3 else kwargs.get("attn_mask")
            is_causal = args[5] if len(args) > 5 else kwargs.get("is_causal", False)
            self.counter._suppress += 1
            try:
                out = func(*args, **kwargs)
            finally:
                self.counter._suppress -= 1
            self.counter.record_sdpa(q, k, v, mask, bool(is_causal), out)
            return out
        return func(*args, **kwargs)


def trace(
    fn: Any,
    inputs: Tuple[Any, ...],
    external: Optional[List[Tuple[torch.Tensor, str]]] = None,
) -> Tuple[SOLCounter, List[torch.Tensor]]:
    """Run ``fn(*inputs)`` under the counter on meta tensors.

    ``inputs`` must already be meta tensors. ``external`` marks storages that
    live in DRAM (params, model inputs). Returns the counter and the flattened
    output tensors.
    """
    counter = SOLCounter()
    for tensor, label in external or []:
        counter.mark_external(tensor, label)
    with _FnHook(counter):
        with counter:
            out = fn(*inputs)
    return counter, _flat_tensors(out)
