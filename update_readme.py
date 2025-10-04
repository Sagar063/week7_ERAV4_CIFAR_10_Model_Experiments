#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inject latest metrics and plots into README.md.

- Expects:
    results/train_log.csv
    results/plots/test_samples.png (optional)
    results/plots/augmented_samples.png (optional)
    results/plots/acc_curves.png (optional)

- Updates or appends blocks:
    <!-- RESULTS --> ... <!-- /RESULTS -->
    <!-- PLOTS --> ... <!-- /PLOTS -->

Usage:
    python update_readme.py
"""
from pathlib import Path
import re
import pandas as pd
from datetime import datetime

ROOT = Path(__file__).parent
README = ROOT / "README.md"
LOG = ROOT / "results" / "train_log.csv"
PLOTS = {
    "Test samples": "results/plots/test_samples.png",
    "Augmented train samples": "results/plots/augmented_samples.png",
    "Accuracy curves": "results/plots/acc_curves.png",
}

def _block(md: str, tag: str, content: str) -> str:
    """Replace or append a <!-- TAG --> ... <!-- /TAG --> block."""
    start = f"<!-- {tag} -->"
    end = f"<!-- /{tag} -->"
    pattern = re.compile(rf"<!-- {tag} -->.*?<!-- /{tag} -->", re.S)
    block = f"{start}\n{content.rstrip()}\n{end}"
    if pattern.search(md):
        return pattern.sub(block, md)
    else:
        # append with two newlines
        return md.rstrip() + "\n\n" + block + "\n"

def _format_results_table(df: pd.DataFrame) -> str:
    # last row for each phase; pick best test
    test = df[df["phase"] == "test"]
    if test.empty:
        return "_No test metrics yet._"
    best = test.loc[test["acc"].idxmax()]
    best_ep = int(best["epoch"])
    best_acc = float(best["acc"]) * 100.0
    best_loss = float(best["loss"])

    # also show last train row (for context)
    train = df[df["phase"] == "train"]
    last_tr = train.iloc[-1] if not train.empty else None

    rows = []
    rows.append("| Metric | Value |")
    rows.append("|---|---:|")
    rows.append(f"| **Best Test Acc** | **{best_acc:.2f}%** (epoch {best_ep}) |")
    rows.append(f"| Best Test Loss | {best_loss:.4f} |")
    if last_tr is not None:
        rows.append(f"| Last Train Acc | {float(last_tr['acc'])*100:.2f}% (epoch {int(last_tr['epoch'])}) |")
        rows.append(f"| Last Train Loss | {float(last_tr['loss']):.4f} |")
    rows.append(f"| Updated | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |")
    return "\n".join(rows)

def _format_plots_section() -> str:
    lines = []
    for title, rel in PLOTS.items():
        p = ROOT / rel
        if p.exists():
            # Use HTML to control width nicely in GitHub
            lines.append(f"<p><strong>{title}</strong><br>")
            lines.append(f'<img src="{rel}" alt="{title}" width="600"></p>\n')
    if not lines:
        return "_No plots found yet._"
    return "\n".join(lines).rstrip()

def main():
    if not README.exists():
        raise SystemExit("README.md not found. Create one first.")

    md = README.read_text(encoding="utf-8")

    # RESULTS block
    if LOG.exists():
        df = pd.read_csv(LOG)
        results_md = _format_results_table(df)
    else:
        results_md = "_No train_log.csv yet. Run training first._"

    md = _block(md, "RESULTS", results_md)

    # PLOTS block
    plots_md = _format_plots_section()
    md = _block(md, "PLOTS", plots_md)

    README.write_text(md, encoding="utf-8")
    print("README.md updated.")

if __name__ == "__main__":
    main()
