
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto-generate receptive-field (RF) markdown for a PyTorch model by importing
the real model, running a dummy forward pass to determine the actual execution
order, and then computing layer-by-layer RF using rf_utils.

Designed to live in: <repo_root>/utils/rf_autogen.py
Expects rf_utils.py alongside it: <repo_root>/utils/rf_utils.py

Usage examples (run from repo root):
  python utils/rf_autogen.py --model-file models/net.py --model-class Net --title "RF: Net (32x32)"

This module is also importable from update_readme.py (repo root).
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn

# Ensure we can import siblings (rf_utils.py) when executed directly
UTILS_DIR = Path(__file__).parent.resolve()
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

try:
    from rf_utils import LayerSpec, rf_trace, rf_markdown_table
except Exception as e:
    raise SystemExit(f"Failed to import rf_utils from {UTILS_DIR}: {e}")


def dynamic_import_from_path(py_path: Path):
    py_path = Path(py_path).resolve()
    spec = importlib.util.spec_from_file_location(py_path.stem, py_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {py_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[py_path.stem] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def build_model(module, class_name: str, ctor_kv: dict | None = None):
    ctor_kv = ctor_kv or {}
    cls = getattr(module, class_name, None)
    if cls is None:
        raise AttributeError(f"Class {class_name} not found in module {module.__name__}")
    try:
        return cls()
    except TypeError:
        return cls(**ctor_kv)


def _int(x):
    if isinstance(x, (tuple, list)):
        return int(x[0])
    return int(x)


def collect_exec_order(model: nn.Module, sample: torch.Tensor):
    visited = []
    name_map = {id(m): n for n, m in model.named_modules()}

    def pre_hook(mod, inp):
        visited.append((name_map.get(id(mod), mod.__class__.__name__), mod))

    handles = []
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)):
            handles.append(m.register_forward_pre_hook(pre_hook))

    model.eval()
    with torch.no_grad():
        _ = model(sample)

    for h in handles:
        h.remove()
    return visited


def layerspecs_from_exec_order(visited) -> List[LayerSpec]:
    layers: List[LayerSpec] = []
    for name, mod in visited:
        if isinstance(mod, nn.Conv2d):
            layers.append(LayerSpec(name=name, k=_int(mod.kernel_size), stride=_int(mod.stride), dilation=_int(mod.dilation)))
        elif isinstance(mod, (nn.MaxPool2d, nn.AvgPool2d)):
            stride = getattr(mod, 'stride', None)
            layers.append(LayerSpec(name=name, k=_int(mod.kernel_size), stride=_int(stride) if stride else _int(mod.kernel_size), dilation=1))
    return layers


def rf_markdown_for_model(model_file: Path, class_name: str, title: str, input_shape: Tuple[int,int,int], ctor: dict | None = None) -> str:
    module = dynamic_import_from_path(model_file)
    model = build_model(module, class_name, ctor_kv=(ctor or {}))

    c, h, w = input_shape
    sample = torch.zeros(1, c, h, w)
    visited = collect_exec_order(model, sample)
    layers = layerspecs_from_exec_order(visited)
    trace = rf_trace(layers)
    md = rf_markdown_table(trace, title if title else f"RF: {class_name}")
    return md


def parse_kv(s: str) -> dict:
    d = {}
    if not s:
        return d
    parts = s.split(',')
    for p in parts:
        if '=' not in p:
            raise argparse.ArgumentTypeError("--ctor expects comma-separated key=value pairs, e.g., num_classes=10")
        k, v = p.split('=', 1)
        try:
            v_cast = int(v)
        except ValueError:
            try:
                v_cast = float(v)
            except ValueError:
                v_cast = v
        d[k.strip()] = v_cast
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-file', required=True, type=Path, help='Path (relative to repo root) to .py that defines the model class')
    ap.add_argument('--model-class', required=True, type=str, help='Class name to instantiate')
    ap.add_argument('--title', required=False, type=str, default='', help='Markdown title for the RF table')
    ap.add_argument('--input', nargs=3, type=int, default=[3, 32, 32], metavar=('C', 'H', 'W'), help='Input tensor shape (CxHxW)')
    ap.add_argument('--ctor', type=parse_kv, default={}, help='Constructor kwargs: key=value,key=value')
    ap.add_argument('--out', type=Path, default=None, help='Optional path to save markdown')
    args = ap.parse_args()

    # Resolve model file from repo root (two levels up from utils/ if run via utils/rf_autogen.py)
    repo_root = Path(__file__).resolve().parents[1]
    model_path = (repo_root / args.model_file).resolve()

    md = rf_markdown_for_model(model_path, args.model_class, args.title, tuple(args.input), ctor=args.ctor)
    if args.out:
        out_path = (repo_root / args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding='utf-8')
        print(f"Wrote RF markdown -> {out_path}")
    else:
        print(md)


if __name__ == '__main__':
    main()
