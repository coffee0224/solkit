"""Command line interface: ``solkit analyze <file>``."""

from __future__ import annotations

import argparse
import sys

from . import __version__, analyze, list_archs
from .bench_loader import load_bench_model


def _cmd_analyze(args: argparse.Namespace) -> int:
    model, inputs = load_bench_model(args.file)
    report = analyze(model, *inputs, arch=args.arch, arch_config=args.arch_config)
    print(report.summary())
    if args.per_op:
        print("\nper-op:")
        for row in report.per_op():
            macs = f"{row['macs']:,}" if row["macs"] else "-"
            print(
                f"  {row['op']:<44} {macs:>14}  {row['mac_dtype']:<8} "
                f"in {row['in_bytes']:>12,}  out {row['out_bytes']:>12,}"
            )
    if args.out:
        p = report.to_yaml(args.out)
        print(f"\nwrote {p}", file=sys.stderr)
    return 0


def _cmd_archs(_args: argparse.Namespace) -> int:
    for name in list_archs():
        print(name)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="solkit", description="Speed-of-light analysis for torch reference code"
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_an = sub.add_parser("analyze", help="analyze a SOL-Bench/KernelBench file")
    p_an.add_argument("file", help="bench .py defining Model/ReferenceModel + get_inputs()")
    p_an.add_argument(
        "--arch", default="RTX_5060_Ti", help=f"arch config name: {', '.join(list_archs())}"
    )
    p_an.add_argument("--arch-config", default=None, help="path to an arch YAML (overrides --arch)")
    p_an.add_argument("--out", default=None, help="write the full report YAML here")
    p_an.add_argument("--per-op", action="store_true", help="print the per-op table")
    p_an.set_defaults(func=_cmd_analyze)

    p_ar = sub.add_parser("archs", help="list bundled architecture configs")
    p_ar.set_defaults(func=_cmd_archs)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
