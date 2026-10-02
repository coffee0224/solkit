"""Analytic einsum charging — intercepted at the API level like SDPA."""

import torch

import solkit


def _e(eq, *shapes, dtype=torch.float32):
    ts = [torch.randn(s, dtype=dtype) for s in shapes]
    return solkit.analyze(lambda *xs: torch.einsum(eq, *xs), *ts)


def test_matmul_equation():
    r = _e("ij,jk->ik", (4, 8), (8, 5))
    assert r.total_macs == 4 * 8 * 5
    assert r.macs_by_dtype == {torch.float32: 160}


def test_batched_with_ellipsis():
    r = _e("...ij,...jk->...ik", (2, 3, 4, 8), (2, 3, 8, 5))
    assert r.total_macs == 2 * 3 * 4 * 8 * 5


def test_implicit_output_mode():
    r = _e("ij,jk", (4, 8), (8, 5))
    assert r.total_macs == 4 * 8 * 5


def test_three_operand_chain():
    # "ij,jk,kl->il": path-independent multiplication count M*K*N*L
    r = _e("ij,jk,kl->il", (2, 3), (3, 4), (4, 5))
    assert r.total_macs == 2 * 3 * 4 * 5


def test_size1_contraction_is_charged():
    # GDN delta-rule rank-1 update: summed dim l has size 1, eager lowers to
    # broadcast mul (0 MACs at dispatch level); the analytic charge keeps the
    # contraction.
    r = _e("hkl,hlv->hkv", (16, 128, 1), (16, 1, 128))
    assert r.total_macs == 16 * 128 * 128
    assert any(row["op"] == "einsum:hkl,hlv->hkv" for row in r.per_op())


def test_dot_product():
    r = _e("i,i->", (7,), (7,))
    assert r.total_macs == 7


def test_mixed_precision_promotes():
    r = _e("hkl,hlv->hkv", (2, 4, 1), (2, 1, 4), dtype=torch.bfloat16)
    assert r.macs_by_dtype == {torch.bfloat16: 2 * 4 * 4}


def test_broadcast_label_takes_max_size():
    r = _e("hkl,hlv->hkv", (4, 8, 1), (1, 1, 6))  # h broadcast 4 vs 1
    assert r.total_macs == 4 * 8 * 6


def test_elementwise_no_summed_index():
    r = _e("ij,ij->ij", (3, 4), (3, 4))
    assert r.total_macs == 0
    assert r.total_other_ops == 12


def test_single_operand_reduction():
    r = _e("ij->i", (3, 4))
    assert r.total_macs == 0
    assert r.total_other_ops == 12


def test_single_operand_permutation_free():
    r = _e("ij->ji", (3, 4))
    assert r.total_macs == 0
    assert r.total_other_ops == 0


def test_list_form():
    a, b = torch.randn(4, 8), torch.randn(8, 5)
    r = solkit.analyze(lambda: torch.einsum("ij,jk->ik", [a, b]))
    assert r.total_macs == 4 * 8 * 5


def test_unfused_bytes_one_kernel():
    r = _e("ij,jk->ik", (4, 8), (8, 5))
    # one charged op: read a (128) + b (160), write out (80)
    assert r.bytes.unfused_bytes == 128 + 160 + 80
    assert r.bytes.fused_read_bytes == 128 + 160
