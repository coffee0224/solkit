"""Load SOL-Bench / KernelBench benchmark files.

A bench file defines a ``Model`` (or ``ReferenceModel``) ``nn.Module`` plus a
``get_inputs()`` function (KernelBench convention), or a SolBench-v3 style
``_ref_get_inputs(_axes, device)`` with module-level ``_axes``/``_param_order``
constants. Inputs are allocated directly on the meta device so no real memory
is needed to analyze a workload.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType
from typing import Any, List, Tuple

import torch


def _load_module(path: str | Path) -> ModuleType:
    p = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(f"solkit_bench_{p.stem}", p)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import bench file {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _find_model_class(mod: ModuleType) -> type:
    for name in ("Model", "ReferenceModel"):
        cls = getattr(mod, name, None)
        if isinstance(cls, type) and issubclass(cls, torch.nn.Module):
            return cls
    raise ValueError("bench file must define 'Model' or 'ReferenceModel' (an nn.Module subclass)")


def _instantiate(mod: ModuleType, cls: type) -> torch.nn.Module:
    if hasattr(mod, "get_init_inputs"):
        try:
            init_inputs = mod.get_init_inputs()
            return cls(*init_inputs) if init_inputs else cls()
        except TypeError:
            pass
    return cls()


def _meta_inputs(mod: ModuleType, path: Path) -> List[Any]:
    # SolBench v3: _ref_get_inputs(_axes, device) with _axes/_param_order
    # declared as literals in the source.
    ref_get = getattr(mod, "_ref_get_inputs", None)
    if callable(ref_get):
        src = path.read_text()
        axes_m = re.search(r"_axes\s*=\s*(\{[^}]*\})", src)
        if axes_m:
            axes = eval(axes_m.group(1))  # noqa: S307 — bench-file literal
            got = ref_get(axes, torch.device("meta"))
            if isinstance(got, dict):
                order_m = re.search(r"_param_order\s*=\s*(\[[^\]]*\])", src)
                order = eval(order_m.group(1)) if order_m else list(got)
                return [got[k] for k in order if k in got]
            return list(got)

    if not hasattr(mod, "get_inputs"):
        raise ValueError("bench file must define get_inputs() or _ref_get_inputs()")

    # get_inputs(): allocate on meta by patching the common factory functions.
    orig = {n: getattr(torch, n) for n in ("randn", "zeros", "ones", "empty", "randint")}

    def meta_factory(name):
        base = orig[name]

        def wrapper(*a, **kw):
            kw.pop("device", None)
            kw.pop("pin_memory", None)
            return base(*a, device="meta", **kw)

        return wrapper

    try:
        for n in orig:
            setattr(torch, n, meta_factory(n))
        out = mod.get_inputs()
    finally:
        for n, f in orig.items():
            setattr(torch, n, f)
    if isinstance(out, torch.Tensor):
        return [out]
    return list(out)


def load_bench_model(path: str | Path) -> Tuple[torch.nn.Module, List[Any]]:
    """Return ``(model, inputs)`` from a SOL-Bench/KernelBench file.

    The model keeps its real device/weights; :func:`solkit.analyze` copies it
    to the meta device itself. Inputs are meta tensors.
    """
    p = Path(path)
    mod = _load_module(p)
    cls = _find_model_class(mod)
    model = _instantiate(mod, cls)
    inputs = _meta_inputs(mod, p)
    return model, inputs
