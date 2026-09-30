"""Analysis result: totals, roofline models, per-op table."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml

from .bytes_model import BytesBreakdown, BytesModel
from .counter import OpRecord
from .roofline import RooflineModel


@dataclass
class Report:
    arch_name: str
    macs_by_dtype: Dict[torch.dtype, int]
    total_macs: int
    total_other_ops: int
    bytes: BytesModel
    breakdown: BytesBreakdown
    roofline: Dict[str, RooflineModel]
    op_records: List[OpRecord]
    warnings: List[str] = field(default_factory=list)

    # Convenience accessors ------------------------------------------------

    @property
    def sol_ms(self) -> float:
        """SOL runtime of the fused model — the speed limit to compare against."""
        return self.roofline["fused"].runtime_ms

    @property
    def unfused_ms(self) -> float:
        """SOL runtime assuming one kernel per op (naive torch execution)."""
        return self.roofline["unfused"].runtime_ms

    @property
    def bottleneck(self) -> str:
        return self.roofline["fused"].bottleneck

    @property
    def total_flops(self) -> int:
        return 2 * self.total_macs

    def arithmetic_intensity(self, model: str = "fused") -> float:
        b = self.roofline[model].memory_bytes
        return self.total_macs / b if b else float("inf")

    def per_op(self) -> List[Dict[str, Any]]:
        return [
            {
                "op": r.name,
                "macs": r.macs,
                "mac_dtype": str(r.mac_dtype).removeprefix("torch.")
                if r.mac_dtype is not None
                else "",
                "other_ops": r.other_ops,
                "in_bytes": r.in_bytes,
                "out_bytes": r.out_bytes,
                "inputs": r.in_shapes,
                "outputs": r.out_shapes,
            }
            for r in self.op_records
        ]

    # Output ----------------------------------------------------------------

    def summary(self) -> str:
        lines = [
            f"arch:        {self.arch_name}",
            f"total MACs:  {self.total_macs:,}"
            + (
                "  ("
                + ", ".join(
                    f"{str(d).removeprefix('torch.')}:{m:,}" for d, m in self.macs_by_dtype.items()
                )
                + ")"
                if len(self.macs_by_dtype) > 1
                else ""
            ),
            f"other ops:   {self.total_other_ops:,} (elementwise, not in SOL compute)",
            f"DRAM fused:  {self.bytes.fused_bytes:,} B  "
            f"(reads {self.bytes.fused_read_bytes:,} + writes {self.bytes.fused_write_bytes:,})",
            f"DRAM unfused: {self.bytes.unfused_bytes:,} B",
        ]
        for name, m in self.roofline.items():
            lines.append(
                f"{name:>9}: SOL {m.runtime_ms * 1e3:9.3f} us  "
                f"bottleneck={m.bottleneck:<7} AI={self.arithmetic_intensity(name):.1f} MAC/B"
            )
        if self.warnings:
            lines.append("warnings:")
            lines.extend(f"  - {w}" for w in self.warnings)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arch": self.arch_name,
            "workload": {
                "total_macs": self.total_macs,
                "macs_by_dtype": {
                    str(d).removeprefix("torch."): m for d, m in self.macs_by_dtype.items()
                },
                "total_other_ops": self.total_other_ops,
            },
            "memory": {
                "fused_read_bytes": self.bytes.fused_read_bytes,
                "fused_write_bytes": self.bytes.fused_write_bytes,
                "fused_bytes": self.bytes.fused_bytes,
                "unfused_bytes": self.bytes.unfused_bytes,
                "breakdown": self.breakdown.by_label,
            },
            "roofline": {
                name: {
                    "runtime_ms": m.runtime_ms,
                    "compute_cycles": int(m.compute_cycles),
                    "memory_cycles": int(m.memory_cycles),
                    "total_cycles": int(m.total_cycles),
                    "bottleneck": m.bottleneck,
                    "arithmetic_intensity_mac_per_byte": self.arithmetic_intensity(name),
                }
                for name, m in self.roofline.items()
            },
            "ops": self.per_op(),
            "warnings": self.warnings,
        }

    def to_yaml(self, path: str | Path) -> Path:
        p = Path(path)
        p.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False))
        return p
