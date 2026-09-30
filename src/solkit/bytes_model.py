"""Memory traffic accounting: ``unfused`` and ``fused`` models.

Both models count DRAM bytes; intermediate tensors that never leave the GPU
are treated differently:

- **unfused**: every op runs in isolation as its own kernel, so all tensor
  reads and writes cross DRAM. This is what a naive one-kernel-per-op
  execution would cost.
- **fused** (the SOL denominator): only *external* traffic counts — external
  storages are model parameters/buffers and the call inputs, which must be
  read from DRAM (each element at most once, deduplicated by storage and
  access region), plus the returned outputs, which must be written back.
  Intermediates are assumed to stay on-chip, which is optimistic when the
  working set exceeds L2 (same assumption SOLAR makes).

Pure-view ops are excluded from both models (they move no data).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch

from .counter import SOLCounter, _tbytes


def _union_bytes(intervals: List[tuple], cap: int) -> int:
    """Total bytes covered by (start, end) intervals, capped at ``cap``."""
    if not intervals:
        return 0
    intervals = sorted(intervals)
    total = 0
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s > cur_e:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    total += cur_e - cur_s
    return min(total, cap)


@dataclass
class BytesModel:
    unfused_bytes: int = 0
    fused_read_bytes: int = 0
    fused_write_bytes: int = 0

    @property
    def fused_bytes(self) -> int:
        return self.fused_read_bytes + self.fused_write_bytes

    @property
    def fused_spill_bytes(self) -> int:
        """Bytes of intermediates an unfused execution would move; informational."""
        return max(0, self.unfused_bytes - self.fused_bytes)


@dataclass
class BytesBreakdown:
    """Fused-model traffic attributed by external tensor kind."""

    by_label: Dict[str, int] = field(default_factory=dict)

    def total(self) -> int:
        return sum(self.by_label.values())


def compute_bytes(counter: SOLCounter, outputs: List[torch.Tensor]) -> BytesModel:
    unfused = 0
    reads: Dict[int, List[tuple]] = {}
    for rec in counter.op_records:
        unfused += rec.in_bytes + rec.out_bytes
        for r in rec.reads:
            reads.setdefault(r.storage_id, []).append((r.start, r.end))

    read_bytes = 0
    for sid, ivals in reads.items():
        ref = counter.external.get(sid)
        if ref is not None:
            read_bytes += _union_bytes(ivals, ref.nbytes)

    # Outputs: written back to DRAM exactly once. Views of external inputs
    # (in-place functions) already counted as reads; count them as writes too,
    # since read + write are separate DRAM transactions.
    write_bytes = 0
    for t in outputs:
        write_bytes += _tbytes(t)

    return BytesModel(
        unfused_bytes=unfused,
        fused_read_bytes=read_bytes,
        fused_write_bytes=write_bytes,
    )


def fused_breakdown(counter: SOLCounter, outputs: List[torch.Tensor]) -> BytesBreakdown:
    """Attribute fused-model bytes to param/input/output labels."""
    reads: Dict[str, List[tuple]] = {}
    caps: Dict[str, int] = {}
    for rec in counter.op_records:
        for r in rec.reads:
            ref = counter.external.get(r.storage_id)
            if ref is not None:
                reads.setdefault(ref.label, []).append((r.start, r.end))
                caps[ref.label] = ref.nbytes
    by_label = {label: _union_bytes(ivals, caps[label]) for label, ivals in reads.items()}
    by_label["<outputs>"] = sum(_tbytes(t) for t in outputs)
    return BytesBreakdown(by_label=by_label)


def macs_by_dtype(counter: SOLCounter) -> Dict[torch.dtype, int]:
    out: Dict[torch.dtype, int] = {}
    for rec in counter.op_records:
        if rec.macs and rec.mac_dtype is not None:
            out[rec.mac_dtype] = out.get(rec.mac_dtype, 0) + rec.macs
    return out
