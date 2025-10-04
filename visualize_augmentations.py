#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize on-the-fly augmentations for CIFAR-10 training images.

Usage:
  python visualize_augmentations.py
  python visualize_augmentations.py --save results/plots/aug_demo.png
  python visualize_augmentations.py --recompute-stats
"""

import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from dataset.cifar10 import AlbumentationsCIFAR10, get_train_transforms
import dataset.cifar10 as cifar_ds  # we'll override its globals

# ---------- stats helpers ----------
def compute_cifar10_stats(data_dir="./data", batch_size=512, num_workers=2):
    ds = datasets.CIFAR10(root=data_dir, train=True, download=True,
                          transform=transforms.ToTensor())  # [0,1]
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    n_pixels = 0
    sum_ = torch.zeros(3, dtype=torch.float64)
    sumsq_ = torch.zeros(3, dtype=torch.float64)

    for xb, _ in tqdm(loader, desc="Computing CIFAR-10 mean/std", leave=False):
        xb = xb.to(dtype=torch.float64)              # [B,C,H,W]
        b, c, h, w = xb.shape
        n_pixels += b * h * w
        sum_ += xb.sum(dim=[0,2,3])
        sumsq_ += (xb * xb).sum(dim=[0,2,3])

    mean = (sum_ / n_pixels)
    var = (sumsq_ / n_pixels) - mean * mean
    std = var.clamp(min=0).sqrt()

    return mean.to(torch.float32).tolist(), std.to(torch.float32).tolist()

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

# ---------- viz helpers ----------
def denormalize(t, mean, std):
    """Undo normalization for display (Tensor -> numpy image)."""
    mean = np.array(mean).reshape(3,1,1)
    std  = np.array(std).reshape(3,1,1)
    t = t.cpu().numpy()
    t = (t * std) + mean
    t = np.clip(t, 0, 1)
    return np.transpose(t, (1, 2, 0))

def visualize(save_path: Path | None, n_samples: int, data_dir: str,
              stats_cache: str, recompute_stats: bool, workers: int):
    # 1) get mean/std (from cache or compute)
    mean, std = (None, None)
    if not recompute_stats:
        mean, std = load_stats_if_available(stats_cache)
    if mean is None or std is None:
        mean, std = compute_cifar10_stats(data_dir=data_dir, num_workers=workers)
        save_stats(stats_cache, mean, std)

    # 2) inject into dataset module so transforms use them
    cifar_ds.CIFAR10_MEAN = tuple(mean)
    cifar_ds.CIFAR10_STD  = tuple(std)

    # 3) dataset with on-the-fly augs
    ds = AlbumentationsCIFAR10(root=data_dir, train=True, download=True,
                               transform=get_train_transforms())

    # random index + show several augmented views of the same base image
    idx = np.random.randint(0, len(ds))
    label = ds.ds.classes[ds.ds.targets[idx]]
    imgs = [ds[idx][0] for _ in range(n_samples)]

    # 4) plot grid
    rows = 2
    cols = int(np.ceil(n_samples / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = axes.flatten()

    for i in range(n_samples):
        ax = axes[i]
        ax.imshow(denormalize(imgs[i], mean, std))
        ax.axis("off")
        ax.set_title(f"Aug #{i+1}", fontsize=9)

    for j in range(n_samples, len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"On-the-fly Augmentations (Label: {label})", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Saved to: {save_path}")
        plt.close(fig)
    else:
        plt.show()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", type=str, default="results/plots/aug_demo.png",
                    help="Path to save the visualization image.")
    ap.add_argument("--data", type=str, default="./data")
    ap.add_argument("--stats-cache", type=str, default="results/cifar10_stats.json")
    ap.add_argument("--recompute-stats", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--n", type=int, default=8, help="number of augmented samples to show")
    args = ap.parse_args()

    visualize(Path(args.save), n_samples=args.n, data_dir=args.data,
              stats_cache=args.stats_cache, recompute_stats=args.recompute_stats,
              workers=args.workers)

if __name__ == "__main__":
    main()
