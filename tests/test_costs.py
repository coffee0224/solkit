"""Cost formula tests — exact expected values on known shapes."""

import torch

import solkit
from solkit.costs import sdpa_macs


def test_mm_macs_and_unfused_bytes():
    a = torch.randn(4, 8, dtype=torch.float16)
    b = torch.randn(8, 5, dtype=torch.float16)
    r = solkit.analyze(lambda x, y: x @ y, a, b)
    assert r.total_macs == 4 * 8 * 5
    assert r.macs_by_dtype == {torch.float16: 160}
    # unfused: read a (64 B) + read b (80 B) + write out (40 B)
    assert r.bytes.unfused_bytes == 64 + 80 + 40
    # fused: reads a+b (external), writes out
    assert r.bytes.fused_read_bytes == 64 + 80
    assert r.bytes.fused_write_bytes == 40


def test_linear_module_counts_addmm_params_external():
    lin = torch.nn.Linear(8, 4, bias=True).to(torch.float16)
    x = torch.randn(2, 8, dtype=torch.float16)
    r = solkit.analyze(lin, x)
    assert r.total_macs == 2 * 8 * 4
    # fused reads: param weight (32) + bias (8) + input (32) = 72 B
    assert r.bytes.fused_read_bytes == 4 * 8 * 2 + 4 * 2 + 2 * 8 * 2
    assert r.breakdown.by_label.get("param:weight") == 64
    assert r.breakdown.by_label.get("param:bias") == 8
    assert r.breakdown.by_label.get("input:0") == 32


def test_bmm_batched():
    a = torch.randn(3, 4, 8, dtype=torch.float16)
    b = torch.randn(3, 8, 5, dtype=torch.float16)
    r = solkit.analyze(lambda x, y: torch.bmm(x, y), a, b)
    assert r.total_macs == 3 * 4 * 8 * 5


def test_sdpa_causal_exact():
    q = torch.randn(1, 2, 16, 8)
    k = torch.randn(1, 2, 16, 8)
    v = torch.randn(1, 2, 16, 8)
    causal = solkit.analyze(
        lambda a, b, c: torch.nn.functional.scaled_dot_product_attention(a, b, c, is_causal=True),
        q,
        k,
        v,
    )
    full = solkit.analyze(
        lambda a, b, c: torch.nn.functional.scaled_dot_product_attention(a, b, c, is_causal=False),
        q,
        k,
        v,
    )
    # Tq=Tk=16: causal sees 16*17/2=136 of 256 full positions; x2 matmuls
    assert causal.total_macs == 2 * (1 * 2 * 136 * 8)
    assert full.total_macs == 2 * (1 * 2 * 256 * 8)
    assert causal.macs_by_dtype == {torch.float32: 2 * (1 * 2 * 136 * 8)}


def test_sdpa_macs_decode_shape():
    # Tq=4 queries against Tk=64 keys, causal top-left: all rows see <= 4 keys
    q = torch.randn(1, 1, 4, 8)
    k = torch.randn(1, 1, 64, 8)
    assert sdpa_macs(q, k, True) == 2 * (4 * 5 // 2) * 8


def test_int8_mm_uses_int8_pipe():
    a = torch.randint(-127, 127, (4, 8), dtype=torch.int8)
    b = torch.randint(-127, 127, (8, 5), dtype=torch.int8)
    r = solkit.analyze(lambda x, y: torch._int_mm(x, y), a, b)
    assert r.total_macs == 4 * 8 * 5
    assert r.macs_by_dtype == {torch.int8: 160}


def test_conv2d():
    x = torch.randn(1, 3, 8, 8)
    w = torch.randn(16, 3, 3, 3)
    r = solkit.analyze(lambda i, wt: torch.nn.functional.conv2d(i, wt), x, w)
    out_elems = 16 * 6 * 6  # no padding -> 8-3+1 = 6
    assert r.total_macs == out_elems * (3 * 3 * 3)


def test_mixed_precision_split():
    # int8 contraction + fp16 elementwise on an fp16 tensor: MACs stay int8
    a = torch.randint(-127, 127, (4, 8), dtype=torch.int8)
    b = torch.randint(-127, 127, (8, 5), dtype=torch.int8)
    c = torch.randn(4, 5, dtype=torch.float16)

    def fn(x, y, z):
        return torch._int_mm(x, y).to(torch.float16) * z

    r = solkit.analyze(fn, a, b, c)
    assert r.total_macs == 160
    assert r.macs_by_dtype == {torch.int8: 160}
    # int8 bytes at 1 B/elt, fp16 at 2 B/elt in the same trace
    assert r.bytes.fused_read_bytes == 32 + 40 + 40


def test_view_ops_free():
    x = torch.randn(8, 8, dtype=torch.float16)
    base = solkit.analyze(lambda t: t.exp(), x)
    viewed = solkit.analyze(lambda t: t.view(64).exp(), x)
    assert viewed.bytes.unfused_bytes == base.bytes.unfused_bytes
    assert viewed.total_other_ops == base.total_other_ops


def test_unknown_contraction_warns():
    from solkit.costs import COST_FUNCS

    saved = COST_FUNCS.pop("aten.mm.default")
    try:
        a = torch.randn(4, 8, dtype=torch.float16)
        b = torch.randn(8, 5, dtype=torch.float16)
        r = solkit.analyze(lambda x, y: x @ y, a, b)
        assert r.total_macs == 0
        assert any("aten.mm" in w for w in r.warnings)
    finally:
        COST_FUNCS["aten.mm.default"] = saved


def test_register_cost_extension():
    @solkit.register_cost("aten::mul.Tensor")
    def _(tensors, kwargs, outs):
        return solkit.OpCost(macs=7, mac_dtype=torch.float16)

    a = torch.randn(4, dtype=torch.float16)
    r = solkit.analyze(lambda x: x * x, a)
    assert r.total_macs == 7
