#!/usr/bin/env python3
"""Derive a solkit arch config from the GPU in this machine.

Three tiers of fidelity, all automated:

1. Static properties (name, SM count, L2/DRAM capacity) come from
   ``torch.cuda.get_device_properties`` -- exact.
2. Sustained SM clock -- sampled via pynvml or ``nvidia-smi`` while
   benchmarks hold the GPU busy. The boost clock is NOT the right
   divisor; under load consumer cards settle 10-20%% lower.
3. Throughput rates (DRAM bandwidth, MAC rate per dtype) -- CUDA-event
   timed microbenchmarks: a large device-to-device copy, and big square
   GEMMs per dtype (tensor-core pipes where applicable). These are
   *achieved* numbers: lower bounds on the dense peak, in practice
   within ~5-15%% of it.

For exact peaks, wrap the ``--probe`` modes in ncu and back-calculate
(the script prints the commands); put the results in the YAML's
``measured:`` block.

Usage:
    python tools/derive_arch.py                    # YAML to stdout
    python tools/derive_arch.py --out my_gpu.yaml
    python tools/derive_arch.py --freq 2.58        # skip clock sampling
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import threading
import time
from datetime import date

import torch

GEMM_DTYPES = ["fp32", "tf32", "fp16", "bf16", "int8", "fp8"]


def _read_sm_clock_mhz() -> int | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


class ClockMonitor(threading.Thread):
    """Sample the SM clock in the background while benchmarks run."""

    def __init__(self, interval_s: float = 0.15):
        super().__init__(daemon=True)
        self._halt = threading.Event()  # not `_stop`: Thread owns that name
        self._interval = interval_s
        self.samples: list[int] = []

    def run(self) -> None:
        while not self._halt.is_set():
            mhz = _read_sm_clock_mhz()
            if mhz:
                self.samples.append(mhz)
            self._halt.wait(self._interval)

    def stop(self) -> int | None:
        self._halt.set()
        self.join()
        return int(statistics.median(self.samples)) if self.samples else None


def bench(fn, warmup: int = 3, iters: int = 10) -> float:
    """Median wall time of ``fn()`` in seconds, CUDA-event timed."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1e3)
    return statistics.median(times)


def _set_fp32_precision(tf32: bool):
    m = torch.backends.cuda.matmul
    if hasattr(m, "fp32_precision"):  # torch >= 2.9
        old = m.fp32_precision
        m.fp32_precision = "tf32" if tf32 else "ieee"
        return old
    old = m.allow_tf32
    m.allow_tf32 = tf32
    return old


def _restore_fp32_precision(old) -> None:
    m = torch.backends.cuda.matmul
    if hasattr(m, "fp32_precision"):
        if old in ("tf32", "ieee"):
            m.fp32_precision = old
    else:
        m.allow_tf32 = old


def gemm_fn(name: str, n: int):
    """Build one square-GEMM launch for dtype ``name``; None if unsupported."""
    dev = "cuda"
    if name == "fp32":
        a = torch.randn(n, n, device=dev)
        b = torch.randn(n, n, device=dev)
        old = _set_fp32_precision(False)
        return lambda: a @ b, lambda: _restore_fp32_precision(old)
    if name == "tf32":
        a = torch.randn(n, n, device=dev)
        b = torch.randn(n, n, device=dev)
        old = _set_fp32_precision(True)
        return lambda: a @ b, lambda: _restore_fp32_precision(old)
    if name in ("fp16", "bf16"):
        dt = {"fp16": torch.float16, "bf16": torch.bfloat16}[name]
        a = torch.randn(n, n, device=dev, dtype=dt)
        b = torch.randn(n, n, device=dev, dtype=dt)
        return (lambda: a @ b), (lambda: None)
    if name == "int8":
        if not hasattr(torch, "_int_mm"):
            return None
        a = torch.randint(-127, 128, (n, n), device=dev, dtype=torch.int8)
        b = torch.randint(-127, 128, (n, n), device=dev, dtype=torch.int8)
        return (lambda: torch._int_mm(a, b)), (lambda: None)
    if name == "fp8":
        if torch.cuda.get_device_capability() < (8, 9) or not hasattr(torch, "_scaled_mm"):
            return None
        dt = torch.float8_e4m3fn
        a = torch.randn(n, n, device=dev, dtype=torch.float16).to(dt)
        b = torch.randn(n, n, device=dev, dtype=torch.float16).to(dt)  # column-major for _scaled_mm
        one = torch.ones((), device=dev, dtype=torch.float32)

        def run():
            torch._scaled_mm(a, b.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16)

        return run, (lambda: None)
    raise ValueError(name)


def measure(size: int, iters: int) -> dict:
    """Absolute achieved rates: DRAM GB/s (copy) and MACs/s per dtype (GEMM)."""
    out: dict = {}

    src = torch.empty(1 << 30, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    # an elementwise kernel, not copy_ (which is a DMA memcpy: no kernel for
    # ncu to see, and often a slightly different path)
    t = bench(lambda: torch.mul(src, 2, out=dst), iters=iters)
    out["dram_copy_gbps"] = 2 * src.numel() / t / 1e9

    for name in GEMM_DTYPES:
        n = size if name != "fp32" else size // 2
        try:
            pair = gemm_fn(name, n)
            if pair is None:
                out[name] = None
                continue
            fn, cleanup = pair
            t = bench(fn, iters=iters)
            cleanup()
            out[name] = n**3 / t  # MACs/s for a square GEMM
        except Exception as e:  # unsupported dtype/kernel on this card
            out[name] = None
            print(f"# {name}: skipped ({type(e).__name__}: {e})")
    return out


def probe(kind: str, size: int) -> None:
    """One kernel launch for ncu to wrap, then exit."""
    if kind == "copy":
        src = torch.randint(0, 255, (1 << 30,), device="cuda", dtype=torch.uint8)
        dst = torch.empty_like(src)
        for _ in range(2):
            torch.mul(src, 2, out=dst)
        torch.cuda.synchronize()
        return
    n = size if kind != "gemm-fp32" else size // 2
    fn, cleanup = gemm_fn(kind.removeprefix("gemm-"), n)
    fn()
    cleanup()
    torch.cuda.synchronize()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=None, help="write the YAML here (else stdout)")
    ap.add_argument("--freq", type=float, default=None, help="skip clock sampling; use this GHz")
    ap.add_argument("--size", type=int, default=8192, help="GEMM edge size (default 8192)")
    ap.add_argument("--iters", type=int, default=10, help="timed iterations per benchmark")
    ap.add_argument("--probe", default=None, help="single-kernel mode for ncu: copy | gemm-<dtype>")
    args = ap.parse_args(argv)

    if not torch.cuda.is_available():
        print("need a CUDA GPU", file=sys.stderr)
        return 2
    torch.cuda.init()

    if args.probe:
        probe(args.probe, args.size)
        return 0

    props = torch.cuda.get_device_properties(0)
    print(f"measuring on {props.name} (CC {props.major}.{props.minor}) ...", file=sys.stderr)

    # Clocks ramp above the sustained point for the first seconds of load,
    # so sample across the whole settle + benchmark window and take the median.
    warm_a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    freq = args.freq
    mon = None
    if freq is None:
        mon = ClockMonitor()
        mon.start()
        t0 = time.monotonic()
        while time.monotonic() - t0 < 2.0:
            warm_a @ warm_a
        torch.cuda.synchronize()

    res = measure(args.size, args.iters)

    if mon is not None:
        sampled = mon.stop()
        if sampled:
            freq = sampled / 1e3
            print(f"sustained SM clock: {sampled} MHz ({len(mon.samples)} samples)", file=sys.stderr)
        else:
            print("could not sample clocks; pass --freq", file=sys.stderr)
            return 2

    def mac_line(val: float | None, yaml_key: str) -> str:
        if val is None:
            return f"# {yaml_key}: UNSUPPORTED/FAILED on this card"
        per_cycle = round(val / (freq * 1e9))
        return f"{yaml_key}: {per_cycle:d}    # achieved on {args.size}-cube GEMM (lower bound of dense peak)"

    lines = [
        f"# Derived {date.today()} by tools/derive_arch.py on {props.name} "
        f"(CC {props.major}.{props.minor}, {props.multi_processor_count} SMs).",
        "# Rates are *achieved* (microbenchmark, <= dense peak). For exact peaks",
        "# run the ncu back-calcs printed to stderr and move values into `measured:`.",
        f'name: "{props.name.replace(" ", "_")}"',
        f"SRAM_capacity: {getattr(props, 'L2_cache_size', 0)}    # L2, from get_device_properties",
        "# SRAM_byte_per_cycle: <optional> L2 bandwidth, informational only;",
        "#   ncu: lts__t_bytes.sum.per_second / lts__throughput.avg.pct_of_peak_sustained_elapsed / freq",
        f"DRAM_capacity: {props.total_memory}",
        f"DRAM_byte_per_cycle: {round(res['dram_copy_gbps'] / freq):d}    # D2D-copy achieved "
        f"({res['dram_copy_gbps']:.0f} GB/s); true peak ~5-15% higher",
        f"freq_GHz: {freq:.2f}    # sampled under load",
        mac_line(res["fp32"], "MAC_per_cycle_fp32_sm"),
        mac_line(res["tf32"], "MAC_per_cycle_tf32_tc"),
        mac_line(res["fp16"], "MAC_per_cycle_fp16_tc"),
        mac_line(res["bf16"], "MAC_per_cycle_bf16_tc"),
        mac_line(res["int8"], "MAC_per_cycle_int8_tc"),
        mac_line(res["fp8"], "MAC_per_cycle_fp8_tc"),
    ]
    yaml_text = "\n".join(lines) + "\n"

    if args.out:
        from pathlib import Path

        Path(args.out).write_text(yaml_text)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(yaml_text)

    py = sys.executable
    print(
        "\nExact-peak ncu back-calcs (value = rate / (pct/100), then / freq in cycles):\n"
        "  DRAM:  ncu --metrics dram__bytes.sum.per_second,dram__throughput.avg.pct_of_peak_sustained_elapsed "
        f"{py} tools/derive_arch.py --probe copy\n"
        "  fp16:  ncu --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed "
        f"{py} tools/derive_arch.py --probe gemm-fp16\n"
        "(repeat the fp16 line per dtype; copy bandwidth needs no clock lock, "
        "but match the freq you put in the YAML)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
