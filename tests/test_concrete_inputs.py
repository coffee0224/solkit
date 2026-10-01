"""concrete() inputs — data-dependent control flow becomes traceable."""

import pytest
import torch

import solkit


def _varlen_scan(x, cu_seqlens):
    # Loop bound comes from a tensor *value*: raises on meta tensors.
    n = int(cu_seqlens[-1].item())
    out = x.clone()
    for i in range(n):
        out = out + x[i]
    return out


def test_value_read_on_meta_raises():
    x = torch.randn(4, 8)
    cu = torch.tensor([0, 4], dtype=torch.int64)
    with pytest.raises(RuntimeError):
        solkit.analyze(_varlen_scan, x, cu)


def test_concrete_unblocks_data_dependent_loop():
    x = torch.randn(4, 8)  # fp32: 4*8*4 = 128 B, every row touched
    cu = torch.tensor([0, 4], dtype=torch.int64)  # 2 * 8 = 16 B
    r = solkit.analyze(_varlen_scan, x, solkit.concrete(cu))
    assert r.total_macs == 0
    # x fully read; cu charged at input-tensor granularity — the select for
    # cu[-1] reads the whole 16 B input, which matches how a real kernel
    # consumes the full cu_seqlens array.
    assert r.bytes.fused_read_bytes == 128 + 16
    assert r.bytes.fused_write_bytes == 128
    assert r.breakdown.by_label["input:1"] == 16


def test_concrete_inside_containers():
    x = torch.randn(4, 8)
    cu = torch.tensor([0, 4], dtype=torch.int64)
    r = solkit.analyze(
        lambda d: _varlen_scan(d["x"], d["cu"]),
        {"x": x, "cu": solkit.concrete(cu)},
    )
    assert r.bytes.fused_read_bytes == 128 + 16
    assert r.breakdown.by_label["input:1"] == 16
