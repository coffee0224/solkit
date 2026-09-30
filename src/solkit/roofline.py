"""Architecture configs and the roofline solver.

Arch YAML files use the SOLAR schema (``freq_GHz``, ``DRAM_byte_per_cycle``,
``SRAM_capacity``, ``SRAM_byte_per_cycle``, ``MAC_per_cycle_<prec>_tc`` ...),
so SOLAR config files work unmodified. solkit additionally honors an optional
``measured:`` block whose keys override the theoretical values — put your
ncu-calibrated numbers there instead of editing the datasheet values.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml

_DTYPE_MAC_KEY = {
    torch.float16: "MAC_per_cycle_fp16_tc",
    torch.bfloat16: "MAC_per_cycle_bf16_tc",
    torch.float32: "MAC_per_cycle_fp32_sm",
    torch.int8: "MAC_per_cycle_int8_tc",
    torch.uint8: "MAC_per_cycle_int8_tc",
    torch.float8_e4m3fn: "MAC_per_cycle_fp8_tc",
    torch.float8_e4m3fnuz: "MAC_per_cycle_fp8_tc",
    torch.float8_e5m2: "MAC_per_cycle_fp8_tc",
    torch.float8_e5m2fnuz: "MAC_per_cycle_fp8_tc",
}

_FALLBACK_KEY = "MAC_per_cycle_fp32_sm"


class ArchNotFoundError(KeyError):
    def __str__(self) -> str:
        return (
            f"unknown arch config {self.args[0]!r}; available: "
            f"{', '.join(list_archs())} (or pass a YAML path to arch_config)"
        )


def list_archs() -> List[str]:
    return sorted(p.stem for p in resources.files("solkit.configs.arch").iterdir())


def load_arch(name_or_path: str) -> Dict[str, Any]:
    p = Path(name_or_path)
    if p.exists():
        cfg = yaml.safe_load(p.read_text())
    else:
        f = resources.files("solkit.configs.arch") / f"{name_or_path}.yaml"
        if not f.is_file():
            raise ArchNotFoundError(name_or_path)
        cfg = yaml.safe_load(f.read_text())
    if not isinstance(cfg, dict) or "freq_GHz" not in cfg:
        raise ValueError(f"arch config {name_or_path!r} is not a valid arch YAML")
    measured = cfg.pop("measured", None) or {}
    if not isinstance(measured, dict):
        raise ValueError("'measured' block must be a mapping")
    cfg.update(measured)
    return cfg


@dataclass
class RooflineModel:
    name: str
    memory_bytes: int
    compute_cycles: float
    memory_cycles: float
    freq_ghz: float

    @property
    def total_cycles(self) -> float:
        return max(self.compute_cycles, self.memory_cycles)

    @property
    def runtime_ms(self) -> float:
        return self.total_cycles / (self.freq_ghz * 1e6)

    @property
    def bottleneck(self) -> str:
        return "compute" if self.compute_cycles >= self.memory_cycles else "memory"


def solve_roofline(
    arch: Dict[str, Any],
    macs_by_dtype: Dict[torch.dtype, int],
    unfused_bytes: int,
    fused_bytes: int,
) -> Dict[str, RooflineModel]:
    """Compute SOL cycles/time for the unfused and fused traffic models."""
    freq = float(arch.get("freq_GHz", 1.0))
    dram_bw = float(arch.get("DRAM_byte_per_cycle", 1.0))
    fallback = float(arch.get(_FALLBACK_KEY, 1.0))

    compute_cycles = 0.0
    for dt, macs in macs_by_dtype.items():
        key = _DTYPE_MAC_KEY.get(dt)
        rate = float(arch.get(key)) if key and key in arch else None
        if rate is None:
            rate = fallback
        compute_cycles += macs / rate

    return {
        "unfused": RooflineModel(
            "unfused", unfused_bytes, compute_cycles, unfused_bytes / dram_bw, freq
        ),
        "fused": RooflineModel("fused", fused_bytes, compute_cycles, fused_bytes / dram_bw, freq),
    }
