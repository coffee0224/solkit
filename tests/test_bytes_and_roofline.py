"""Fused-model dedup, invariants, and roofline math tests."""

import pytest
import torch

import solkit
from solkit.roofline import load_arch, solve_roofline


def test_fused_dedups_repeated_reads():
    x = torch.randn(1000, dtype=torch.float16)

    def twice(t):
        return t.exp() + t.sin()

    r = solkit.analyze(twice, x)
    # x read by exp and by sin, but DRAM delivers it once
    assert r.bytes.fused_read_bytes == 2000
    assert r.bytes.unfused_bytes > r.bytes.fused_bytes


def test_fused_dedups_overlapping_slices():
    x = torch.randn(1000, dtype=torch.float16)

    def slices(t):
        a = t[:800].exp()
        b = t[200:].sin()
        return a.sum() + b.sum()

    r = solkit.analyze(slices, x)
    # [0,800) and [200,1000) overlap: union = 1000 elements = 2000 B, not 3200
    assert r.bytes.fused_read_bytes == 2000


def test_fused_dedups_views_vs_base():
    x = torch.randn(64, 64, dtype=torch.float16)

    def base_and_view(t):
        return t.sum() + t.view(4096).exp().sum()

    r = solkit.analyze(base_and_view, x)
    assert r.bytes.fused_read_bytes == 64 * 64 * 2


def test_outputs_written_once():
    a = torch.randn(4, 8, dtype=torch.float16)
    b = torch.randn(8, 5, dtype=torch.float16)
    r = solkit.analyze(lambda x, y: x @ y, a, b)
    assert r.bytes.fused_write_bytes == 4 * 5 * 2


def test_multi_output_writes():
    a = torch.randn(4, dtype=torch.float16)
    r = solkit.analyze(lambda t: (t.exp(), t.sin()), a)
    assert r.bytes.fused_write_bytes == 16


def _tiny_arch():
    return {
        "name": "Tiny",
        "freq_GHz": 1.0,
        "DRAM_byte_per_cycle": 100.0,
        "MAC_per_cycle_fp16_tc": 1000.0,
        "MAC_per_cycle_fp32_sm": 100.0,
    }


def test_roofline_compute_bound():
    m = solve_roofline(_tiny_arch(), {torch.float16: 1_000_000}, 1000, 1000)
    f = m["fused"]
    assert f.compute_cycles == 1000.0
    assert f.memory_cycles == 10.0
    assert f.bottleneck == "compute"
    assert f.runtime_ms == pytest.approx(0.001)  # 1000 cycles @ 1 GHz


def test_roofline_memory_bound():
    m = solve_roofline(_tiny_arch(), {torch.float16: 1000}, 1_000_000, 1_000_000)
    f = m["fused"]
    assert f.bottleneck == "memory"
    assert f.memory_cycles == 10_000.0
    assert f.runtime_ms == pytest.approx(0.01)  # 10k cycles @ 1 GHz


def test_roofline_mixed_dtype_cycles_add():
    arch = _tiny_arch()
    arch["MAC_per_cycle_int8_tc"] = 2000.0
    m = solve_roofline(arch, {torch.float16: 100_000, torch.int8: 200_000}, 0, 0)
    assert m["fused"].compute_cycles == 100.0 + 100.0


def test_measured_override(tmp_path):
    cfg = tmp_path / "arch.yaml"
    cfg.write_text(
        "name: T\n"
        "freq_GHz: 1\n"
        "DRAM_byte_per_cycle: 100\n"
        "MAC_per_cycle_fp16_tc: 1000\n"
        "measured:\n  DRAM_byte_per_cycle: 80\n"
    )
    a = load_arch(str(cfg))
    assert a["DRAM_byte_per_cycle"] == 80


def test_load_arch_unknown_lists_available():
    try:
        load_arch("nope")
    except KeyError as e:
        assert "RTX_5060_Ti" in str(e)
        assert "H100_PCIe" in str(e)
    else:
        raise AssertionError("expected ArchNotFoundError")


def test_unfused_ge_fused_invariant():
    a = torch.randn(64, 64, dtype=torch.float16)
    r = solkit.analyze(lambda t: (t @ t).exp(), a)
    assert r.bytes.unfused_bytes >= r.bytes.fused_bytes
    assert r.unfused_ms >= r.sol_ms


def test_report_yaml_roundtrip(tmp_path):
    a = torch.randn(4, 8, dtype=torch.float16)
    b = torch.randn(8, 5, dtype=torch.float16)
    r = solkit.analyze(lambda x, y: x @ y, a, b)
    p = r.to_yaml(tmp_path / "r.yaml")
    import yaml

    data = yaml.safe_load(p.read_text())
    assert data["workload"]["total_macs"] == 160
    assert data["roofline"]["fused"]["bottleneck"] in ("compute", "memory")
    assert data["memory"]["breakdown"]["input:0"] == 64
    assert any(row["op"] == "aten.mm.default" for row in data["ops"])
