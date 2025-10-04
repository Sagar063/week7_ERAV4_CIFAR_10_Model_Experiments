
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Auto-update README.md with:
- Combined accuracy/loss plots (base vs dilated)
- Summary table (per model)
- Visual sample grids (test, augmented)
- Model summaries
- RF tables for both models (auto-computed from live classes)
- Confusion matrices and key classification report stats

This script assumes:
- repo root contains: README.md, model.py, utils/rf_utils.py, utils/rf_autogen.py
- per-model results are saved under results/base_model and results/dilated_model
- train logs: results/<model>/train_log.csv  with columns: epoch, train_acc, test_acc, train_loss, test_loss

Placeholders supported in README (use any/all):
  Blocks with start/end:
    <!-- COMBINED_SUMMARY_TABLE --> ... <!-- /COMBINED_SUMMARY_TABLE -->
    <!-- COMBINED_ACC_PLOT --> ... <!-- /COMBINED_ACC_PLOT -->
    <!-- COMBINED_LOSS_PLOT --> ... <!-- /COMBINED_LOSS_PLOT -->
    <!-- TEST_SAMPLES_GRID --> ... <!-- /TEST_SAMPLES_GRID -->
    <!-- AUGMENTED_SAMPLES_GRID --> ... <!-- /AUGMENTED_SAMPLES_GRID -->
    <!-- MODEL_SUMMARY_BASE --> ... <!-- /MODEL_SUMMARY_BASE -->
    <!-- MODEL_SUMMARY_DILATED --> ... <!-- /MODEL_SUMMARY_DILATED -->
    <!-- CM_BASE --> ... <!-- /CM_BASE -->
    <!-- CM_DILATED --> ... <!-- /CM_DILATED -->
    <!-- CLS_REPORT_BASE --> ... <!-- /CLS_REPORT_BASE -->
    <!-- CLS_REPORT_DILATED --> ... <!-- /CLS_REPORT_DILATED -->
    <!-- RF_NET --> ... <!-- /RF_NET -->
    <!-- RF_DILATED --> ... <!-- /RF_DILATED -->

  (If only a single-line marker exists, the content will be inserted right after it.)

Usage examples:
  python update_readme.py \
    --rf model.py:Net:"RF: Net (32x32 input)" \
    --rf model.py:NetDilated:"RF: NetDilated (32x32 input)" \
    --rf-input 3 32 32 --rf-ctor num_classes=10
"""

import argparse, re, sys, json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.resolve()
README = ROOT / "README.md"

# Default model result folders
BASE_DIR = ROOT / "results" / "base_model"
DILATED_DIR = ROOT / "results" / "dilated_model"

# Combined artifacts destination
COMBINED_DIR = ROOT / "results" / "combined"
COMBINED_PLOTS = COMBINED_DIR / "plots"

# Make utils importable for RF
import sys as _sys
UTILS_DIR = (ROOT / "utils").resolve()
if str(UTILS_DIR) not in _sys.path:
    _sys.path.insert(0, str(UTILS_DIR))

# Try import RF generator
_rf_err = ""
try:
    import utils.rf_autogen as _rf
except Exception as _e:
    _rf_err = f"{_e}"
    _rf = None


# ----------------------- README helpers -----------------------

def _ensure_tags(md: str, tag: str) -> str:
    """Ensure there's a block for tag; if not, create it at end."""
    start = f"<!-- {tag} -->"
    end = f"<!-- /{tag} -->"
    if start in md and end in md:
        return md
    if start in md and end not in md:
        # single-line marker, add an end right after
        md = md.replace(start, f"{start}\n{end}")
        return md
    # neither present — append a new empty block
    return md.rstrip() + f"\n\n{start}\n{end}\n"

def _set_block(md: str, tag: str, content: str) -> str:
    """Set/replace a block delimited by <!-- TAG --> ... <!-- /TAG -->"""
    md = _ensure_tags(md, tag)
    start = f"<!-- {tag} -->"
    end = f"<!-- /{tag} -->"
    pattern = re.compile(rf"<!-- {re.escape(tag)} -->.*?<!-- /{re.escape(tag)} -->", re.S)
    block = f"{start}\n{content.rstrip()}\n{end}"
    return pattern.sub(block, md)

def _img(path: Path, title: str, width: int = 800) -> str:
    rel = path.resolve().relative_to(ROOT).as_posix() if path.exists() else path.as_posix()
    return f'<p><strong>{title}</strong><br><img src="{rel}" alt="{title}" width="{width}"></p>'

def _read_text_safe(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        return f"_Could not read {path}: {e}_"


# ----------------------- Data load -----------------------
def _first_epoch_over(df: pd.DataFrame, threshold=0.85, col="test_acc"):
    hits = df.loc[df[col] >= threshold, "epoch"]
    if len(hits):
        return int(hits.iloc[0])
    return "N/A"


def _load_log(d: Path):
    csvp = d / "train_log.csv"
    if not csvp.exists():
        return None
    try:
        df = pd.read_csv(csvp)
        # sanity
        needed = {"epoch","train_acc","test_acc","train_loss","test_loss"}
        if not needed.issubset(df.columns):
            return None
        df = df.sort_values("epoch").drop_duplicates(subset=["epoch"], keep="last")
        return df
    except Exception:
        return None

def _best_epoch(df: pd.DataFrame, col="test_acc"):
    idx = int(df[col].idxmax())
    row = df.loc[idx]
    return int(row["epoch"]), float(row[col])

def _params_from_summary(path: Path):
    txt = _read_text_safe(path)
    # Expect first line "Trainable parameters: N"
    for line in txt.splitlines():
        line = line.strip()
        if line.lower().startswith("trainable parameters"):
            # extract digits
            import re as _re
            m = _re.search(r"(\d[\d_,]*)", line)
            if m:
                return int(m.group(1).replace(",","").replace("_",""))
    return None

def _cls_report_stats(path: Path):
    """Return macro/weighted precision, recall, f1 if available."""
    try:
        df = pd.read_csv(path, index_col=0)
        rows = {}
        for key in ["macro avg","weighted avg"]:
            if key in df.columns or key in df.index:
                # scikit's output_dict=True writes metrics as columns (labels) and rows as [precision, recall, f1-score, support]
                if key in df.columns:
                    col = df[key]
                    rows[key] = {
                        "precision": float(col.get("precision", np.nan)),
                        "recall": float(col.get("recall", np.nan)),
                        "f1": float(col.get("f1-score", np.nan)),
                    }
                else:
                    row = df.loc[key]
                    rows[key] = {
                        "precision": float(row.get("precision", np.nan)),
                        "recall": float(row.get("recall", np.nan)),
                        "f1": float(row.get("f1-score", np.nan)),
                    }
        return rows
    except Exception:
        return {}

# ----------------------- Plots -----------------------

def _plot_combined_acc(df_base, df_dil, outpath: Path):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9,4.5))
    if df_base is not None:
        plt.plot(df_base["epoch"], df_base["train_acc"], label="Base Train", linestyle="-")
        plt.plot(df_base["epoch"], df_base["test_acc"],  label="Base Val",   linestyle="--")
    if df_dil is not None:
        plt.plot(df_dil["epoch"], df_dil["train_acc"],  label="Dilated Train", linestyle="-")
        plt.plot(df_dil["epoch"], df_dil["test_acc"],   label="Dilated Val",   linestyle="--")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy (Train vs Val) — Base vs Dilated")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(outpath, dpi=150); plt.close()

def _plot_combined_loss(df_base, df_dil, outpath: Path):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9,4.5))
    if df_base is not None:
        plt.plot(df_base["epoch"], df_base["train_loss"], label="Base Train", linestyle="-")
        plt.plot(df_base["epoch"], df_base["test_loss"],  label="Base Val",   linestyle="--")
    if df_dil is not None:
        plt.plot(df_dil["epoch"], df_dil["train_loss"],  label="Dilated Train", linestyle="-")
        plt.plot(df_dil["epoch"], df_dil["test_loss"],   label="Dilated Val",   linestyle="--")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss (Train vs Val) — Base vs Dilated")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(outpath, dpi=150); plt.close()



def render_aug_demo(img_path="results/augmentations_plots/aug_demo.png"):
    p = Path(img_path)
    if p.exists():
        return (
            f'<p><strong>Augmentation Demo</strong><br>\n'
            f'<img src="{img_path}" alt="Augmentation Demo" width="900">\n'
            f'</p>'
        )
    # if not found, show a gentle hint (keeps README tidy)
    return (
        "<p><em>Augmentation image not found.</em> "
        "Run: <code>python visualize_augmentations.py --data ./data --n-samples 16</code> "
        "to generate <code>results/augmentations_plots/aug_demo.png</code>.</p>"
    )

# ----------------------- RF -----------------------

def _rf_markdown_for(path: Path, cls: str, title: str, input_shape, ctor_kv):
    if _rf is None:
        return f"_RF disabled (rf_autogen import failed: {_rf_err})_"
    try:
        md = _rf.rf_markdown_for_model(path, cls, title, input_shape, ctor=ctor_kv)
        return md
    except Exception as e:
        return f"_RF extraction failed for {cls}: {e}_"


# ----------------------- Main flow -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rf', action='append', default=[], help="Repeatable: 'path.py:ClassName:Title' (Title optional)")
    ap.add_argument('--rf-input', nargs=3, type=int, default=[3,32,32], metavar=('C','H','W'))
    ap.add_argument('--rf-ctor', type=str, default="", help="Constructor kwargs: 'k=v,k=v' (no spaces)")

    ap.add_argument('--base-dir', default=str(BASE_DIR), help="Path to base model results dir")
    ap.add_argument('--dilated-dir', default=str(DILATED_DIR), help="Path to dilated model results dir")
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    dil_dir = Path(args.dilated_dir)

    # Load logs
    df_b = _load_log(base_dir)
    df_d = _load_log(dil_dir)

    # Combined plots
    COMBINED_PLOTS.mkdir(parents=True, exist_ok=True)
    acc_png = COMBINED_PLOTS / "combined_acc.png"
    loss_png = COMBINED_PLOTS / "combined_loss.png"
    _plot_combined_acc(df_b, df_d, acc_png)
    _plot_combined_loss(df_b, df_d, loss_png)

    # Summary table rows
    rows = []
    for name, d, df in [
        ("base_model", base_dir, df_b),
        ("dilated_model", dil_dir, df_d),
    ]:
        if d.exists() and df is not None and len(df):
            first85 = _first_epoch_over(df, 0.85, "test_acc")
            best_ep, best_acc = _best_epoch(df, "test_acc")
            final_acc = float(df.iloc[-1]["test_acc"])
            epochs = int(df["epoch"].max())
            params = _params_from_summary(d / "model_summary.txt")
            # lr / optimizer from defaults; override with ckpt args if last/best exists
            opt = "SGD"
            lr = None
            ckpt_last = d / "checkpoints" / "last.pth"
            if ckpt_last.exists():
                try:
                    import torch
                    ck = torch.load(str(ckpt_last), map_location="cpu")
                    a = ck.get("args", {})
                    lr = a.get("lr", lr)
                except Exception:
                    pass
            row = {
                "exp_name": name,
                "params": params if params is not None else "N/A",
                "best_test_acc": round(best_acc*100, 2) if best_acc is not None else "N/A",
                "best_epoch": best_ep if best_ep is not None else "N/A",
                "final_test_acc": round(final_acc*100, 2) if final_acc is not None else "N/A",
                "epochs": epochs if epochs is not None else "N/A",
                "epoch>85%": first85, # not persisted
                "augment": True,
                "optimizer": opt,
                "lr": lr if lr is not None else "default(0.1)",
                "use_steplr": False,
            }
            rows.append(row)

    # Build summary markdown table
    if rows:
        cols = ["exp_name","params","best_test_acc","best_epoch","final_test_acc","epochs","epoch>85%","augment","optimizer","lr","use_steplr"]
        header = "| " + " | ".join(cols) + " |"
        sep = "|" + "|".join(["---"]*len(cols)) + "|"
        lines = [header, sep]
        for r in rows:
            vals = [str(r.get(c,"")) for c in cols]
            lines.append("| " + " | ".join(vals) + " |")
        summary_md = "\n".join(lines)
    else:
        summary_md = "_No results found yet. Train both models to populate this table._"

    # Visual grids (take from whichever exists)
    test_grid = None
    aug_grid = None
    for d in [base_dir, dil_dir]:
        p1 = d / "plots" / "test_samples_grid.png"
        p2 = d / "plots" / "augmented_samples_grid.png"
        if test_grid is None and p1.exists():
            test_grid = p1
        if aug_grid is None and p2.exists():
            aug_grid = p2
    test_md = _img(test_grid, "Test Samples (grid)") if test_grid else "_No test grid found._"
    aug_md = _img(aug_grid, "Augmented Train Samples (grid)") if aug_grid else "_No augmented grid found._"

    # Model summaries
    base_sum_txt = _read_text_safe(base_dir / "model_summary.txt")
    dil_sum_txt  = _read_text_safe(dil_dir / "model_summary.txt")
    base_sum_md = "<details><summary><b>Model summary (Base)</b></summary>\n\n```\n" + base_sum_txt.strip() + "\n```\n</details>"
    dil_sum_md  = "<details><summary><b>Model summary (Dilated)</b></summary>\n\n```\n" + dil_sum_txt.strip() + "\n```\n</details>"

    # Confusion matrices and reports (macro/weighted)
    cm_base = base_dir / "plots" / "cm.png"
    cm_dil  = dil_dir / "plots" / "cm.png"
    cm_base_md = _img(cm_base, "Confusion Matrix — Base") if cm_base.exists() else "_Base confusion matrix not found._"
    cm_dil_md  = _img(cm_dil,  "Confusion Matrix — Dilated") if cm_dil.exists() else "_Dilated confusion matrix not found._"

    rep_base = _cls_report_stats(base_dir / "classification_report.csv")
    rep_dil  = _cls_report_stats(dil_dir / "classification_report.csv")

    def _rep_table(rep):
        if not rep:
            return "_Classification report not found._"
        rows = []
        rows.append("**Classification Report**\n")
        header = "| Average Type | Precision | Recall | F1-score |"
        sep    = "|---|---:|---:|---:|"
        rows += [header, sep]
        for key, label in [("macro avg", "**Macro Avg**"), ("weighted avg", "**Weighted Avg**")]:
            r = rep.get(key, {})
            pr = r.get("precision", np.nan)
            rc = r.get("recall", np.nan)
            f1 = r.get("f1", np.nan)
            rows.append(f"| {label} | {pr:.3f} | {rc:.3f} | {f1:.3f} |")
        return "\n".join(rows)


    rep_base_md = _rep_table(rep_base)
    rep_dil_md  = _rep_table(rep_dil)

    # RF per model (if requested OR default to model.py)
    ctor = {}
    if args.rf_ctor:
        # parse k=v,k=v
        for pair in args.rf_ctor.split(","):
            if not pair.strip():
                continue
            k, v = pair.split("=", 1)
            k = k.strip(); v = v.strip()
            try:
                v_cast = int(v)
            except ValueError:
                try:
                    v_cast = float(v)
                except ValueError:
                    v_cast = v
            ctor[k] = v_cast

    rf_specs = []
    if args.rf:
        for spec in args.rf:
            parts = spec.split(":", 2)
            if len(parts) < 2: continue
            path = (ROOT / parts[0].strip()).resolve()
            cls = parts[1].strip()
            title = parts[2].strip() if len(parts) == 3 else f"RF: {cls}"
            rf_specs.append((path, cls, title))
    else:
        # sensible default to your model.py
        rf_specs = [
            (ROOT / "model.py", "Net", "RF: Net (32x32 input)"),
            (ROOT / "model.py", "NetDilated", "RF: NetDilated (32x32 input)"),
        ]

    rf_md_map = {}
    for pth, cls, title in rf_specs:
        rf_md_map[cls] = _rf_markdown_for(pth, cls, title, tuple(args.rf_input), ctor)

    # Read README
    if not README.exists():
        README.write_text("# README\n", encoding="utf-8")
    md = README.read_text(encoding="utf-8")

    # Inject combined items
    md = _set_block(md, "COMBINED_SUMMARY_TABLE", summary_md)
    md = _set_block(md, "COMBINED_ACC_PLOT", _img(acc_png, "Accuracy — Base vs Dilated"))
    md = _set_block(md, "COMBINED_LOSS_PLOT", _img(loss_png, "Loss — Base vs Dilated"))

    # Visual grids
    md = _set_block(md, "TEST_SAMPLES_GRID", test_md)
    md = _set_block(md, "AUGMENTED_SAMPLES_GRID", aug_md)

    # Model summaries
    md = _set_block(md, "MODEL_SUMMARY_BASE", base_sum_md)
    md = _set_block(md, "MODEL_SUMMARY_DILATED", dil_sum_md)

    # Confusion matrices + reports
    md = _set_block(md, "CM_BASE", cm_base_md)
    md = _set_block(md, "CM_DILATED", cm_dil_md)
    md = _set_block(md, "CLS_REPORT_BASE", rep_base_md)
    md = _set_block(md, "CLS_REPORT_DILATED", rep_dil_md)

    # RF tables
    # Add helper headings as requested
    rf_base_block = rf_md_map.get("Net", "_RF for Net missing._")
    rf_dil_block  = rf_md_map.get("NetDilated", "_RF for NetDilated missing._")

    md = _set_block(md, "RF_NET", rf_base_block)
    md = _set_block(md, "RF_DILATED", rf_dil_block)

    # Write back
    README.write_text(md, encoding="utf-8")
    print("README.md updated.")
    print(f"- Combined plots -> {COMBINED_PLOTS}")
    print("- Blocks filled: COMBINED_SUMMARY_TABLE, COMBINED_ACC_PLOT, COMBINED_LOSS_PLOT,")
    print("                TEST_SAMPLES_GRID, AUGMENTED_SAMPLES_GRID,")
    print("                MODEL_SUMMARY_BASE, MODEL_SUMMARY_DILATED,")
    print("                CM_BASE, CM_DILATED, CLS_REPORT_BASE, CLS_REPORT_DILATED,")
    print("                RF_NET, RF_DILATED")

if __name__ == "__main__":
    main()
