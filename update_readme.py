
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inject latest metrics, plots, and **auto-computed Receptive Field (RF)** tables into README.md.

Layout assumption:
- This script lives at repo root: <repo_root>/update_readme.py
- RF helpers live in: <repo_root>/utils/rf_autogen.py and <repo_root>/utils/rf_utils.py

Also embeds (if present):
- results/plots/test_samples_grid.png
- results/plots/augmented_samples_grid.png
- results/plots/acc_curves.png
- results/plots/loss_curves.png
- results/plots/cm.png
- results/plots/test_samples/*.png
- results/plots/augmented_samples/*.png
Blocks updated/created in README.md:
- <!-- RESULTS --> ... <!-- /RESULTS -->
- <!-- PLOTS --> ... <!-- /PLOTS -->
- <!-- TILE_GALLERY --> ... <!-- /TILE_GALLERY -->
- <!-- VISUALIZATION --> ... <!-- /VISUALIZATION -->
Usage:
  python update_readme.py \

    --rf models/net.py:Net:"RF: Net (32x32)" \

    --rf models/net_dilated.py:NetDilated:"RF: NetDilated (32x32)" \

    --rf-input 3 32 32 \

    --rf-ctor num_classes=10
"""
from __future__ import annotations

from pathlib import Path
import re
import pandas as pd
from datetime import datetime
import glob
import argparse
import sys

ROOT = Path(__file__).parent.resolve()
README = ROOT / "README.md"
LOG = ROOT / "results" / "train_log.csv"
MODEL_SUMMARY = ROOT / "results" / "model_summary.txt"

# Ensure utils/ is importable, then import rf_autogen symbols
UTILS_DIR = (ROOT / "utils").resolve()
if UTILS_DIR.exists() and str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

try:
    import utils.rf_autogen as _rf
    rf_markdown_for_model = _rf.rf_markdown_for_model
    parse_kv = _rf.parse_kv
except Exception as e:
    rf_markdown_for_model = None  # type: ignore
    parse_kv = None  # type: ignore
    _RF_IMPORT_ERROR = str(e)
else:
    _RF_IMPORT_ERROR = ""

PLOTS = [
    ("Test samples (grid)", "results/plots/test_samples_grid.png"),
    ("Augmented train samples (grid)", "results/plots/augmented_samples_grid.png"),
    ("Accuracy curves", "results/plots/acc_curves.png"),
    ("Loss curves", "results/plots/loss_curves.png"),
    ("Confusion matrix (normalized)", "results/plots/cm.png"),
]
TILES = [
    ("Test sample tiles", "results/plots/test_samples"),
    ("Augmented sample tiles", "results/plots/augmented_samples"),
]


def _block(md: str, tag: str, content: str) -> str:
    start = f"<!-- {tag} -->"
    end = f"<!-- /{tag} -->"
    pattern = re.compile(rf"<!-- {tag} -->.*?<!-- /{tag} -->", re.S)
    block = f"{start}\n{content.rstrip()}\n{end}"
    return pattern.sub(block, md) if pattern.search(md) else md.rstrip() + "\n\n" + block + "\n"


def _format_results_table(df: pd.DataFrame) -> str:
    wide = {"epoch","train_loss","train_acc","test_loss","test_acc"}.issubset(set(df.columns))
    if wide:
        best_idx = df["test_acc"].idxmax()
        best_row = df.loc[best_idx]
        last_tr = df.iloc[-1]
        rows = [
            "| Metric | Value |",
            "|---|---:|",
            f"| **Best Test Acc** | **{float(best_row['test_acc'])*100:.2f}%** (epoch {int(best_row['epoch'])}) |",
            f"| Best Test Loss | {float(best_row['test_loss']):.4f} |",
            f"| Last Train Acc | {float(last_tr['train_acc'])*100:.2f}% (epoch {int(last_tr['epoch'])}) |",
            f"| Last Train Loss | {float(last_tr['train_loss']):.4f} |",
            f"| Updated | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |",
        ]
        return "\n".join(rows)

    long = {"epoch","phase","loss","acc"}.issubset(set(df.columns))
    if long:
        test = df[df["phase"] == "test"]
        if test.empty:
            return "_No test metrics yet._"
        best = test.loc[test["acc"].idxmax()]
        train = df[df["phase"] == "train"]
        last_tr = train.iloc[-1] if not train.empty else None
        rows = [
            "| Metric | Value |",
            "|---|---:|",
            f"| **Best Test Acc** | **{float(best['acc'])*100:.2f}%** (epoch {int(best['epoch'])}) |",
            f"| Best Test Loss | {float(best['loss']):.4f} |",
        ]
        if last_tr is not None:
            rows.append(f"| Last Train Acc | {float(last_tr['acc'])*100:.2f}% (epoch {int(last_tr['epoch'])}) |")
            rows.append(f"| Last Train Loss | {float(last_tr['loss']):.4f} |")
        rows.append(f"| Updated | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |")
        return "\n".join(rows)

    return "_Unrecognized train_log.csv schema (expected wide or long)._"


def _format_plots_section() -> str:
    lines = []
    for title, rel in PLOTS:
        p = ROOT / rel
        if p.exists():
            lines.append(f"<p><strong>{title}</strong><br>")
            lines.append(f'<img src="{rel}" alt="{title}" width="720"></p>\n')
    return "\n".join(lines).rstrip() or "_No plots found yet._"


def _format_tile_gallery() -> str:
    groups = []
    for label, folder in TILES:
        pf = ROOT / folder
        if not pf.exists():
            continue
        imgs = sorted(glob.glob(str(pf / "*.png")))[:8]
        if not imgs:
            continue
        row = [f"<h4>{label}</h4>", "<p>"]
        for im in imgs:
            rel = Path(im).resolve().relative_to(ROOT.resolve()).as_posix()
            row.append(f'<img src="{rel}" width="128" style="margin:4px;">')
        row.append("</p>")
        groups.append("\n".join(row))
    return "\n\n".join(groups) or "_No per-image tiles found._"


def _format_visualization_usage() -> str:
    return (
        "**Augmentation Visualization**\n\n"
        "Use the helper to render a crisp grid and per-image tiles (nearest-neighbor, no blur):\n\n"
        "```bash\n"
        "python visualize_augmentations.py --save results/plots/aug_demo.png\n"
        "```\n"
        "_Why_: sanity-check your Albumentations pipeline and confirm computed mean/std are used.\n"
    )


def _format_model_summary() -> str:
    if MODEL_SUMMARY.exists():
        txt = MODEL_SUMMARY.read_text(encoding="utf-8")
        return (
            "<details><summary><b>Model summary (click to expand)</b></summary>\n\n"
            "```\n" + txt.rstrip() + "\n```\n\n"
            "</details>"
        )
    return "_No model summary found. Train once to generate it (results/model_summary.txt)._"


# ------------------------- RF INTEGRATION -------------------------

def _parse_rf_specs(args) -> list[dict]:
    specs = []
    for spec in (args.rf or []):
        parts = spec.split(":", 2)
        if len(parts) < 2:
            raise SystemExit(f"--rf expects 'path.py:ClassName:Title' (Title optional). Got: {spec}")
        path = parts[0].strip()
        cls = parts[1].strip()
        title = parts[2].strip() if len(parts) == 3 else f"RF: {cls}"
        specs.append({"path": path, "cls": cls, "title": title})
    return specs


def _format_rf_section(args) -> str:
    if rf_markdown_for_model is None:
        msg = "_rf_autogen import failed; skipping RF section._"
        if 'RF_IMPORT_ERROR' in globals() and _RF_IMPORT_ERROR:
            msg += f" Error: {_RF_IMPORT_ERROR}"
        return msg

    rf_specs = _parse_rf_specs(args)
    if not rf_specs:
        return "_No RF models specified yet. Re-run with --rf path.py:Class:Title to include a table._"

    pieces = []
    ctor = parse_kv(args.rf_ctor) if parse_kv and args.rf_ctor else {}
    for s in rf_specs:
        model_path = (ROOT / s["path"]).resolve()
        md = rf_markdown_for_model(
            model_path, s["cls"], s["title"],
            (args.rf_input[0], args.rf_input[1], args.rf_input[2]), ctor=ctor
        )
        pieces.append(md)
    return "\n\n".join(pieces)


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument('--rf', action='append', default=[], help="Repeatable: 'path.py:ClassName:Title' (Title optional)")
    ap.add_argument('--rf-input', nargs=3, type=int, default=[3, 32, 32], metavar=('C','H','W'), help='Input tensor shape for RF (CxHxW)')
    ap.add_argument('--rf-ctor', type=str, default="", help="Constructor kwargs, e.g., 'num_classes=10,channels=64' (no spaces)")
    return ap


def main():
    if not README.exists():
        raise SystemExit("README.md not found. Create one first.")
    args = _build_argparser().parse_args()

    md = README.read_text(encoding="utf-8")

    # RESULTS
    if LOG.exists():
        try:
            df = pd.read_csv(LOG)
            results_md = _format_results_table(df)
        except Exception as e:
            results_md = f"_Could not parse train_log.csv: {e}_"
    else:
        results_md = "_No train_log.csv yet. Run training first._"
    md = _block(md, "RESULTS", results_md)

    # PLOTS
    plots_md = _format_plots_section()
    md = _block(md, "PLOTS", plots_md)

    # TILE GALLERY
    tiles_md = _format_tile_gallery()
    md = _block(md, "TILE_GALLERY", tiles_md)

    # VISUALIZATION
    vis_md = _format_visualization_usage()
    md = _block(md, "VISUALIZATION", vis_md)

    # MODEL SUMMARY
    model_md = _format_model_summary()
    md = _block(md, "MODEL_SUMMARY", model_md)

    # RF (auto-computed from your real model)
    try:
        rf_md = _format_rf_section(args)
    except Exception as e:
        rf_md = f"_RF extraction failed: {e}_"
    md = _block(md, "RF", rf_md)

    README.write_text(md, encoding="utf-8")
    print("README.md updated.")


if __name__ == "__main__":
    main()
