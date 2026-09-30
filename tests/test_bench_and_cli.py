"""End-to-end: bench file loading (KernelBench + SolBench v3) and CLI."""

import torch

import solkit
from solkit.bench_loader import load_bench_model

KERNELBENCH_FILE = """
import torch

HIDDEN = 16

class Model(torch.nn.Module):
    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.fc1 = torch.nn.Linear(hidden, 4 * hidden)
        self.fc2 = torch.nn.Linear(4 * hidden, hidden)

    def forward(self, x):
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))

def get_inputs():
    return (torch.randn(2, HIDDEN, dtype=torch.float16),)
"""

SOLBENCH_V3_FILE = """
import torch

_axes = {"B": 2, "T": 8, "D": 16}
_param_order = ["x", "w"]

def _ref_get_inputs(_axes, device):
    return {
        "x": torch.randn(_axes["B"], _axes["T"], _axes["D"], device=device),
        "w": torch.randn(_axes["D"], _axes["D"], device=device),
    }

class ReferenceModel(torch.nn.Module):
    def forward(self, x, w):
        return torch.nn.functional.gelu(x @ w)
"""


def test_kernelbench_load_and_analyze(tmp_path):
    p = tmp_path / "kernelbench.py"
    p.write_text(KERNELBENCH_FILE)
    model, inputs = load_bench_model(p)
    assert isinstance(model, torch.nn.Module)
    assert inputs[0].device.type == "meta"
    r = solkit.analyze(model, *inputs)
    assert r.total_macs == 2 * 16 * 64 + 2 * 64 * 16
    assert "param:fc1.weight" in r.breakdown.by_label


def test_solbench_v3_load(tmp_path):
    p = tmp_path / "solbench.py"
    p.write_text(SOLBENCH_V3_FILE)
    model, inputs = load_bench_model(p)
    assert isinstance(model, torch.nn.Module)
    shapes = [tuple(t.shape) for t in inputs]
    assert shapes == [(2, 8, 16), (16, 16)]
    r = solkit.analyze(model, *inputs)
    assert r.total_macs == 2 * 8 * 16 * 16
    assert r.bytes.fused_read_bytes == 2 * 8 * 16 * 4 + 16 * 16 * 4


def test_cli_analyze(tmp_path, capsys):
    from solkit.__main__ import main

    p = tmp_path / "kernelbench.py"
    p.write_text(KERNELBENCH_FILE)
    out = tmp_path / "report.yaml"
    rc = main(["analyze", str(p), "--out", str(out)])
    assert rc == 0
    captured = capsys.readouterr()
    assert "SOL" in captured.out
    assert "bottleneck" in captured.out
    assert out.exists()


def test_cli_archs(capsys):
    from solkit.__main__ import main

    assert main(["archs"]) == 0
    out = capsys.readouterr().out
    assert "RTX_5060_Ti" in out
    assert "H100_PCIe" in out


def test_gqa_example_end_to_end():
    # the shipped example: linear projections + broadcast (GQA) causal SDPA
    from pathlib import Path

    model, inputs = load_bench_model(Path(__file__).parent.parent / "examples" / "gqa_bench.py")
    r = solkit.analyze(model, *inputs)
    assert r.total_macs > 0
    assert any("sdpa" in rec.name for rec in r.op_records)
    assert r.sol_ms > 0
