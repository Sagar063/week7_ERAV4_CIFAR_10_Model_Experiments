#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inject latest metrics and plots into README.md (schema-agnostic).

Supported train_log.csv schemas:
1) New (wide) schema: epoch,train_loss,train_acc,test_loss,test_acc,lr
2) Old (long) schema: epoch,phase,loss,acc

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
    python update_readme.py
"""
from pathlib import Path
import re
import pandas as pd
from datetime import datetime
import glob

ROOT = Path(__file__).parent
README = ROOT / "README.md"
LOG = ROOT / "results" / "train_log.csv"

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
    """
    Produce a small summary table, handling both schemas.
    """
    # New schema?
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

    # Old schema? (long format with phase)
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

    # Unknown schema
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
    # Show up to 8 tiles from each directory
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
            row.append(f'<img src="{Path(im).as_posix()}" width="128" style="margin:4px;">')
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

def main():
    if not README.exists():
        raise SystemExit("README.md not found. Create one first.")

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

    # PLOTS (grids, curves, confusion matrix)
    plots_md = _format_plots_section()
    md = _block(md, "PLOTS", plots_md)

    # TILE GALLERY (optional)
    tiles_md = _format_tile_gallery()
    md = _block(md, "TILE_GALLERY", tiles_md)

    # VISUALIZATION usage (commands + purpose)
    vis_md = _format_visualization_usage()
    md = _block(md, "VISUALIZATION", vis_md)

    README.write_text(md, encoding="utf-8")
    print("README.md updated.")

if __name__ == "__main__":
    main()
