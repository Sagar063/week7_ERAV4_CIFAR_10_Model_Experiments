#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize on-the-fly augmentations for CIFAR-10 training images.

Usage:
    python visualize_augmentations.py
    python visualize_augmentations.py --save results/plots/aug_demo.png

"""

import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
from dataset.cifar10 import AlbumentationsCIFAR10, get_train_transforms, CIFAR10_MEAN, CIFAR10_STD

def denormalize(t):
    """Undo normalization for display (Tensor -> numpy image)."""
    mean = np.array(CIFAR10_MEAN).reshape(3,1,1)
    std = np.array(CIFAR10_STD).reshape(3,1,1)
    t = t.cpu().numpy()
    t = (t * std) + mean
    t = np.clip(t, 0, 1)
    return np.transpose(t, (1, 2, 0))

def visualize(save_path: Path = None, n_samples: int = 8):
    # Initialize dataset with on-the-fly augmentations
    ds = AlbumentationsCIFAR10(root="./data", train=True, download=True, transform=get_train_transforms())

    idx = np.random.randint(0, len(ds))
    label = ds.ds.classes[ds.ds.targets[idx]]
    print(f"Visualizing on-the-fly augmentations for sample index: {idx}, label: {label}")

    imgs = [ds[idx][0] for _ in range(n_samples)]

    # Plot grid (2x4 or appropriate shape)
    rows = 2
    cols = int(np.ceil(n_samples / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = axes.flatten()

    for i in range(n_samples):
        ax = axes[i]
        ax.imshow(denormalize(imgs[i]))
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", type=str, default="results/plots/aug_demo.png",
                        help="Optional path to save the visualization (default: results/plots/aug_demo.png).")
    args = parser.parse_args()

    visualize(Path(args.save))

if __name__ == "__main__":
    main()
