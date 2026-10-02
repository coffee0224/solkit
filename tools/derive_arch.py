#!/usr/bin/env python3
"""Derive a solkit arch config from the GPU in this machine.

Values are obtained in priority order:

1. **ncu counters, where they exist** (exact hardware peaks, immune to
   kernel quality): per-dtype tensor MAC rates from
   ``sm__ops_path_tensor_src_<dt>_dst_fp32.sum.peak_sustained`` (ops are
   counted as 2 per MAC, summed over SMs), DRAM and L2 peaks from
   ``dram__bytes`` / ``lts__t_bytes`` ``.sum.peak_sustained_elapsed.per_second``.
   One profiled GEMM yields all of them at once.
2. **CUDA-event microbenchmarks** for the rest (achieved numbers, lower
   bounds of the peaks): fp32 always (CUDA cores have no ops_path
   metric), any tensor dtype ncu did not report on this arch, and a D2D
   elementwise kernel for DRAM if the counter read failed.

Static properties come from ``torch.cuda.get_device_properties``; the
sustained SM clock (needed to turn bytes/s into bytes/cycle) is sampled
under GEMM load -- the boost clock is NOT the right divisor.

Usage:
    python tools/derive_arch.py                    # YAML to stdout
    python tools/derive_arch.py --out my_gpu.yaml
    python tools/derive_arch.py --no-ncu           # benchmarks only
    python tools/derive_arch.py --freq 2.58        # skip clock sampling
"""

from __future__ import annotations

import argparse
import csv
import io
import shutil
import statistics
import subprocess
import sys
import threading
import time
from datetime import date
from pathlib import Path

import torch

GEMM_DTYPES = ["fp32", "tf32", "fp16", "bf16", "int8", "fp8"]

# ncu "math ops" are counted as 2 per MAC (verified: a n^3-MAC GEMM reports
# 2*n^3 ops). .sum aggregates per-SM rates into a GPU-wide per-cycle figure.
NCU_TENSOR_METRIC = {
    "tf32": "sm__ops_path_tensor_src_tf32_dst_fp32.sum.peak_sustained",
    "fp16": "sm__ops_path_tensor_src_fp16_dst_fp32.sum.peak_sustained",
    "bf16": "sm__ops_path_tensor_src_bf16_dst_fp32.sum.peak_sustained",
    "int8": "sm__ops_path_tensor_src_int8.sum.peak_sustained",
    "fp8": "sm__ops_path_tensor_src_fp8_dst_fp32.sum.peak_sustained",
}
NCU_DRAM = "dram__bytes.sum.peak_sustained_elapsed.per_second"
NCU_L2 = "lts__t_bytes.sum.peak_sustained_elapsed.per_second"


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


def measure(size: int, iters: int, dtypes: list[str], with_dram: bool) -> dict:
    """Absolute achieved rates for the dtypes ncu could not cover."""
    out: dict = {"dram_bps": None}

    if with_dram:
        src = torch.empty(1 << 30, dtype=torch.uint8, device="cuda")
        dst = torch.empty_like(src)
        # an elementwise kernel, not copy_ (which is a DMA memcpy: no kernel for
        # ncu to see, and often a slightly different path)
        t = bench(lambda: torch.mul(src, 2, out=dst), iters=iters)
        out["dram_bps"] = 2 * src.numel() / t

    for name in dtypes:
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


def ncu_peaks(size: int) -> dict:
    """Hardware peaks straight from ncu counters; partial dict on failure.

    peak_sustained metrics are unit properties, so one profiled GEMM
    reports every path's peak regardless of which pipe the kernel uses.
    """
    ncu = shutil.which("ncu")
    if ncu is None:
        return {}
    metrics = list(NCU_TENSOR_METRIC.values()) + [NCU_DRAM, NCU_L2]
    cmd = [
        ncu, "--csv", "--target-processes", "all", "--clock-control", "none",
        "--metrics", ",".join(metrics),
        sys.executable, str(Path(__file__).resolve()), "--probe", "gemm-bf16",
        "--size", str(size),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except (subprocess.TimeoutExpired, OSError) as e:
        print(f"# ncu unavailable ({type(e).__name__}); falling back to benchmarks", file=sys.stderr)
        return {}

    want = set(metrics)
    vals: dict[str, float] = {}
    for row in csv.reader(io.StringIO(proc.stdout)):
        if len(row) < 3 or row[-3] not in want:
            continue
        try:
            vals[row[-3]] = float(row[-1].replace(",", ""))
        except ValueError:
            continue
    if not vals and proc.returncode != 0:
        print(f"# ncu failed (exit {proc.returncode}); falling back to benchmarks", file=sys.stderr)

    out: dict = {}
    for dt, metric in NCU_TENSOR_METRIC.items():
        if metric in vals:
            out[dt] = vals[metric] / 2  # ops are 2 per MAC
    if NCU_DRAM in vals:
        out["dram_bps"] = vals[NCU_DRAM]
    if NCU_L2 in vals:
        out["l2_bps"] = vals[NCU_L2]
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
    ap.add_argument("--no-ncu", action="store_true", help="never call ncu; benchmarks only")
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

    peaks = {} if args.no_ncu else ncu_peaks(args.size)
    got = [dt for dt in GEMM_DTYPES if dt in peaks]
    if got:
        print(f"ncu: direct peaks for {', '.join(got)} + DRAM/L2", file=sys.stderr)

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

    need_dram = peaks.get("dram_bps") is None
    need_gemm = [d for d in GEMM_DTYPES if d not in peaks]  # fp32 is never in peaks
    res = measure(args.size, args.iters, need_gemm, need_dram)

    if mon is not None:
        sampled = mon.stop()
        if sampled:
            freq = sampled / 1e3
            print(f"sustained SM clock: {sampled} MHz ({len(mon.samples)} samples)", file=sys.stderr)
        else:
            print("could not sample clocks; pass --freq", file=sys.stderr)
            return 2

    def src(dt: str) -> str:
        return "ncu" if dt in peaks else "bench"

    def mac_line(val: float | None, yaml_key: str, src: str) -> str:
        if val is None:
            return f"# {yaml_key}: UNSUPPORTED/FAILED on this card"
        # ncu peak_sustained is already per-cycle; benchmarks report MACs/s.
        per_cycle = round(val) if src == "ncu" else round(val / (freq * 1e9))
        if src == "ncu":
            return f"{yaml_key}: {per_cycle:d}    # ncu peak_sustained (hardware peak)"
        return (
            f"{yaml_key}: {per_cycle:d}    # GEMM achieved on this stack "
            f"(lower bound of peak; no ncu metric)"
        )

    dram_bps = peaks.get("dram_bps") or res["dram_bps"]
    if dram_bps:
        dram_src = "ncu peak" if peaks.get("dram_bps") else "D2D-copy achieved (~85-92% of peak)"
        dram_line = (
            f"DRAM_byte_per_cycle: {round(dram_bps / (freq * 1e9)):d}    # {dram_src} "
            f"({dram_bps / 1e9:.0f} GB/s)"
        )
    else:
        dram_line = "# DRAM_byte_per_cycle: FAILED (no ncu metric, copy bench failed)"

    l2_bps = peaks.get("l2_bps")
    if l2_bps:
        sram_line = (
            f"SRAM_byte_per_cycle: {round(l2_bps / (freq * 1e9)):d}    # ncu peak "
            f"({l2_bps / 1e9:.0f} GB/s), informational"
        )
    else:
        sram_line = (
            "# SRAM_byte_per_cycle: <optional> L2 bandwidth, informational only;\n"
            "#   ncu: lts__t_bytes.sum.peak_sustained_elapsed.per_second / freq"
        )

    lines = [
        f"# Derived {date.today()} by tools/derive_arch.py on {props.name} "
        f"(CC {props.major}.{props.minor}, {props.multi_processor_count} SMs).",
        "# MAC/clk and byte/clk peaks: ncu counters where available (exact),",
        "# CUDA-event benchmarks elsewhere (achieved, <= peak).",
        f'name: "{props.name.replace(" ", "_")}"',
        f"SRAM_capacity: {getattr(props, 'L2_cache_size', 0)}    # L2, from get_device_properties",
        sram_line,
        f"DRAM_capacity: {props.total_memory}",
        dram_line,
        f"freq_GHz: {freq:.2f}    # sampled under load",
        mac_line(peaks.get("fp32") or res.get("fp32"), "MAC_per_cycle_fp32_sm", "bench"),
        mac_line(peaks.get("tf32") or res.get("tf32"), "MAC_per_cycle_tf32_tc", src("tf32")),
        mac_line(peaks.get("fp16") or res.get("fp16"), "MAC_per_cycle_fp16_tc", src("fp16")),
        mac_line(peaks.get("bf16") or res.get("bf16"), "MAC_per_cycle_bf16_tc", src("bf16")),
        mac_line(peaks.get("int8") or res.get("int8"), "MAC_per_cycle_int8_tc", src("int8")),
        mac_line(peaks.get("fp8") or res.get("fp8"), "MAC_per_cycle_fp8_tc", src("fp8")),
    ]
    yaml_text = "\n".join(lines) + "\n"

    if args.out:
        Path(args.out).write_text(yaml_text)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(yaml_text)

    if not peaks:
        py = sys.executable
        print(
            "\nncu not used; for exact peaks instead of achieved numbers, install ncu and rerun,\n"
            "or wrap the probes manually:\n"
            "  ncu --metrics sm__ops_path_tensor_src_fp16_dst_fp32.sum.peak_sustained "
            f"{py} tools/derive_arch.py --probe gemm-fp16\n",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
