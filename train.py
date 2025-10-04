#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, time, math, random, json
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

# metrics
from sklearn.metrics import confusion_matrix, classification_report

# IMPORTANT: import as a module so we can override its globals
import dataset.cifar10 as cifar_ds
from model import Net, NetDilated


# -------------------- Repro & device helpers --------------------
def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def get_device(arg):
    if arg == "cpu": return torch.device("cpu")
    if arg == "cuda": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------- Vis utils --------------------
# def show_grid(images, labels, classes, title, mean, std, save_path=None, max_n=16):
#     """Visualize a batch, denormalizing with the provided mean/std."""
#     n = min(len(images), max_n)
#     cols = 8
#     rows = math.ceil(n / cols)
#     fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6))
#     axes = axes.ravel()

#     mean = np.array(mean).reshape(3, 1, 1)
#     std = np.array(std).reshape(3, 1, 1)

#     for i in range(cols * rows):
#         ax = axes[i]
#         ax.axis("off")
#         if i < n:
#             img = images[i].cpu().numpy()  # C,H,W (normalized)
#             img = (img * std + mean)       # back to [0,1]
#             img = np.clip(img, 0, 1).transpose(1, 2, 0)
#             ax.imshow(img)
#             ax.set_title(classes[labels[i]], fontsize=8)

#     plt.suptitle(title)
#     plt.tight_layout()
#     if save_path:
#         Path(save_path).parent.mkdir(parents=True, exist_ok=True)
#         plt.savefig(save_path, dpi=150)
#     plt.close(fig)

# -------------------- Vis utils --------------------


def _denorm_to_numpy(img_t, mean, std):
    """(C,H,W) tensor in [normalized] -> numpy uint8 (H,W,C)"""
    mean = np.array(mean).reshape(3,1,1)
    std  = np.array(std).reshape(3,1,1)
    x = img_t.cpu().numpy()
    x = (x * std + mean)       # back to 0..1
    x = np.clip(x, 0.0, 1.0)
    x = (x * 255.0 + 0.5).astype(np.uint8).transpose(1, 2, 0)  # HWC uint8
    return x

def save_grid_and_tiles(images, labels, classes, title,
                        mean, std, grid_path=None, tiles_dir=None,
                        max_n=16, upscale=4):
    """
    Save a crisp grid (nearest neighbor) and optional per-image tiles.
    - images: Tensor batch [B,C,H,W] already normalized
    - upscale: integer scale (e.g., 4 makes 32x32 -> 128x128)
    """
    n = min(len(images), max_n)
    cols = 8
    rows = int(np.ceil(n / cols))

    # Prepare denorm’d numpy images
    imgs_np = [_denorm_to_numpy(images[i], mean, std) for i in range(n)]

    # Save per-image tiles if requested
    if tiles_dir:
        tiles_dir = Path(tiles_dir)
        tiles_dir.mkdir(parents=True, exist_ok=True)
        for i, arr in enumerate(imgs_np):
            im = Image.fromarray(arr)
            if upscale > 1:
                im = im.resize((arr.shape[1]*upscale, arr.shape[0]*upscale), Image.NEAREST)
            im.save(tiles_dir / f"img_{i:02d}_{classes[labels[i]]}.png")

    # Build and save the grid (nearest, no smoothing)
    fig_w, fig_h = cols * 2.0, rows * 2.0  # light footprint; scaling handled by imshow
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axes = np.atleast_2d(axes).ravel()

    for i in range(rows * cols):
        ax = axes[i]
        ax.axis("off")
        if i < n:
            arr = imgs_np[i]
            if upscale > 1:
                # Matplotlib nearest + small figure is enough; but optionally pre-upscale
                arr = np.array(Image.fromarray(arr).resize((arr.shape[1]*upscale, arr.shape[0]*upscale), Image.NEAREST))
            ax.imshow(arr, interpolation="nearest")
            ax.set_title(classes[labels[i]], fontsize=8)

    plt.suptitle(title)
    plt.tight_layout()
    if grid_path:
        Path(grid_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(grid_path, dpi=150)
    plt.close(fig)


# -------------------- Mean/Std (train split only) --------------------
def compute_cifar10_stats(data_dir="./data", batch_size=512, num_workers=2):
    """
    Compute CIFAR-10 per-channel mean & std (0..1 scale) from the TRAIN split only.
    """
    from torchvision import datasets, transforms

    ds = datasets.CIFAR10(root=data_dir, train=True, download=True,
                          transform=transforms.ToTensor())  # ensures [0,1]
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    n_pixels = 0
    sum_ = torch.zeros(3, dtype=torch.float64)
    sumsq_ = torch.zeros(3, dtype=torch.float64)

    for xb, _ in tqdm(loader, desc="Computing CIFAR-10 mean/std", leave=False):
        xb = xb.to(dtype=torch.float64)     # [B, C, H, W] in [0,1]
        b, c, h, w = xb.shape
        n_pixels += b * h * w
        sum_ += xb.sum(dim=[0, 2, 3])
        sumsq_ += (xb * xb).sum(dim=[0, 2, 3])

    mean = (sum_ / n_pixels)
    var = (sumsq_ / n_pixels) - mean * mean
    std = var.clamp(min=0).sqrt()

    mean = mean.to(torch.float32).tolist()
    std = std.to(torch.float32).tolist()
    return mean, std


def load_stats_if_available(path: str):
    try:
        with open(path, "r") as f:
            obj = json.load(f)
        if "mean" in obj and "std" in obj and len(obj["mean"]) == 3 and len(obj["std"]) == 3:
            return obj["mean"], obj["std"]
    except Exception:
        pass
    return None, None

def save_stats(path: str, mean, std):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"mean": mean, "std": std}, f, indent=2)


# -------------------- Model summary --------------------


def show_model_summary(model, input_size=(3,32,32), device_str="cpu"):
    print("\n" + "="*60)
    print("🧾 Model Summary")
    print("-"*60)
    try:
        # Prefer torchinfo so batch isn’t -1
        from torchinfo import summary as ti_summary
        print(ti_summary(model, input_size=(1, *input_size), device=device_str))
    except Exception:
        try:
            from torchsummary import summary as ts_summary
            ts_summary(model, input_size=input_size, device=device_str)
        except Exception:
            print("Install torchinfo or torchsummary for layer-wise summary.")
    print("="*60 + "\n")


# -------------------- Train / Eval with live progress --------------------
def train_one_epoch(model, loader, device, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="train", leave=False)
    for xb, yb in pbar:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        bs = xb.size(0)
        running_loss += loss.item() * bs
        pred = out.argmax(1)
        correct += (pred == yb).sum().item()
        total += bs

        pbar.set_postfix(loss=f"{running_loss/total:.4f}", acc=f"{correct/total:.4f}")
    return running_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_true, all_pred = [], []
    pbar = tqdm(loader, desc="eval", leave=False)
    for xb, yb in pbar:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        loss = criterion(out, yb)

        bs = xb.size(0)
        running_loss += loss.item() * bs
        pred = out.argmax(1)
        correct += (pred == yb).sum().item()
        total += bs

        all_true.append(yb.cpu().numpy())
        all_pred.append(pred.cpu().numpy())

        pbar.set_postfix(loss=f"{running_loss/total:.4f}", acc=f"{correct/total:.4f}")

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    return running_loss / total, correct / total, y_true, y_pred


# -------------------- Confusion matrix plot --------------------
def plot_confusion_matrix(cm, classes, normalize=True, title="Confusion matrix", save_path="results/plots/cm.png"):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True).clip(min=1e-12)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # annotate cells
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


# -------------------- Main --------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="./data", type=str)
    p.add_argument("--epochs", default=25, type=int)
    p.add_argument("--batch-size", default=128, type=int)
    p.add_argument("--lr", default=1e-3, type=float)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--workers", default=2, type=int)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--step-size", default=10, type=int)
    p.add_argument("--gamma", default=0.5, type=float)
    p.add_argument("--model", default="basic", choices=["basic", "dilated"])
    # caching options for stats
    p.add_argument("--stats-cache", default="results/cifar10_stats.json", type=str)
    p.add_argument("--recompute-stats", action="store_true",
                   help="Force recomputing CIFAR-10 mean/std even if cache exists.")
    args = p.parse_args()

    seed_all(args.seed)
    device = get_device(args.device)

    # 1) Load or compute stats from TRAIN split only
    mean, std = (None, None)
    if not args.recompute_stats:
        mean, std = load_stats_if_available(args.stats_cache)

    if mean is None or std is None:
        mean, std = compute_cifar10_stats(args.data, batch_size=512, num_workers=args.workers)
        save_stats(args.stats_cache, mean, std)

    # Pretty print
    print("\n" + "=" * 60)
    print("📊 CIFAR-10 Channel Statistics")
    print("-" * 60)
    print(f"  ▪ Mean (R, G, B): ({mean[0]:.6f}, {mean[1]:.6f}, {mean[2]:.6f})")
    print(f"  ▪ Std  (R, G, B): ({std[0]:.6f},  {std[1]:.6f},  {std[2]:.6f})")
    print("-" * 60)
    print("  (Each value corresponds to a channel: Red, Green, Blue)")
    print("=" * 60 + "\n")

    # 2) Inject these into dataset module globals so its Albumentations uses them
    cifar_ds.CIFAR10_MEAN = tuple(mean)
    cifar_ds.CIFAR10_STD = tuple(std)

    # 3) Build transforms/datasets using those stats
    t_train = cifar_ds.get_train_transforms()
    t_test = cifar_ds.get_test_transforms()
    ds_train = cifar_ds.AlbumentationsCIFAR10(args.data, train=True, download=True, transform=t_train)
    ds_test = cifar_ds.AlbumentationsCIFAR10(args.data, train=False, download=True, transform=t_test)
    classes = ds_train.ds.classes

    train_loader = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )
    test_loader = DataLoader(
        ds_test, batch_size=max(args.batch_size, 256), shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    # 4) Model / Optim / Sched
    model = Net().to(device) if args.model == "basic" else NetDilated().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params}")

    # 4a) Layer-wise summary (if torchsummary/torchinfo available)
    show_model_summary(model, input_size=(3, 32, 32), device_str=("cuda" if device.type == "cuda" else "cpu"))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # 5) Quick visualization (denormalized)
    # xb0, yb0 = next(iter(test_loader))
    # show_grid(xb0, yb0, classes, "Test samples (normalized -> shown denorm)", mean, std,
    #           save_path="results/plots/test_samples.png")
    # xb1, yb1 = next(iter(train_loader))
    # show_grid(xb1, yb1, classes, "Augmented train samples (denorm)", mean, std,
    #           save_path="results/plots/augmented_samples.png")
    # 5) Quick visualization (crisp grid + individual tiles)
    xb0, yb0 = next(iter(test_loader))
    save_grid_and_tiles(
        xb0, yb0, classes,
        title="Test samples (normalized -> shown denorm)",
        mean=mean, std=std,
        grid_path="results/plots/test_samples_grid.png",
        tiles_dir="results/plots/test_samples",   # NEW: per-image tiles
        max_n=16, upscale=4                       # upscale 4x => 128x128
    )

    xb1, yb1 = next(iter(train_loader))
    save_grid_and_tiles(
        xb1, yb1, classes,
        title="Augmented train samples (denorm)",
        mean=mean, std=std,
        grid_path="results/plots/augmented_samples_grid.png",
        tiles_dir="results/plots/augmented_samples",  # NEW: per-image tiles
        max_n=16, upscale=4
    )


    # 6) Training loop with history
    log_path = Path("results") / "train_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    history_rows = []
    best_acc, best_ep = -1.0, -1
    start_time = time.time()

    for epoch in tqdm(range(1, args.epochs + 1), desc="[train] epochs", leave=True):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, device, criterion, optimizer)
        te_loss, te_acc, y_true, y_pred = evaluate(model, test_loader, device, criterion)
        scheduler.step()

        tqdm.write(f"Epoch {epoch:03d}/{args.epochs} | "
                   f"Train: loss={tr_loss:.4f} acc={tr_acc*100:.2f}% | "
                   f"Test:  loss={te_loss:.4f} acc={te_acc*100:.2f}%")

        history_rows.append({
            "epoch": epoch,
            "train_loss": tr_loss, "train_acc": tr_acc,
            "test_loss": te_loss,  "test_acc": te_acc,
            "lr": optimizer.param_groups[0]["lr"],
        })

        if te_acc > best_acc:
            best_acc, best_ep = te_acc, epoch
            Path("results").mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "acc": best_acc, "epoch": best_ep},
                       "results/best.pth")

    total_time = time.time() - start_time
    pd.DataFrame(history_rows).to_csv(log_path, index=False)

    # 7) Curves
    df = pd.DataFrame(history_rows)
    Path("results/plots").mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 4))
    plt.plot(df["epoch"], df["train_acc"], label="train_acc")
    plt.plot(df["epoch"], df["test_acc"], label="test_acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy curves")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig("results/plots/acc_curves.png", dpi=150); plt.close(fig)

    fig = plt.figure(figsize=(8, 4))
    plt.plot(df["epoch"], df["train_loss"], label="train_loss")
    plt.plot(df["epoch"], df["test_loss"], label="test_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss curves")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig("results/plots/loss_curves.png", dpi=150); plt.close(fig)

    # 8) Confusion matrix & classification report (last epoch predictions)
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, classes=classes, normalize=True,
                          title="CIFAR-10 Confusion (normalized)",
                          save_path="results/plots/cm.png")

    cls_rep = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    pd.DataFrame(cls_rep).to_csv("results/classification_report.csv")

    print(f"\nBest test acc: {best_acc*100:.2f}% at epoch {best_ep}")
    print(f"Total train time: {total_time:.1f}s")
    print("Saved:")
    print("  - results/train_log.csv")
    print("  - results/plots/acc_curves.png")
    print("  - results/plots/loss_curves.png")
    print("  - results/plots/cm.png")
    print("  - results/classification_report.csv")
    print("  - results/best.pth")


if __name__ == "__main__":
    main()
