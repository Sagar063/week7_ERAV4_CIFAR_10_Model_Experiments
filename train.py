#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, time, math, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

from dataset.cifar10 import AlbumentationsCIFAR10, get_train_transforms, get_test_transforms, CIFAR10_MEAN, CIFAR10_STD
from model import Net, NetDilated

def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic=False; torch.backends.cudnn.benchmark=True

def show_grid(images, labels, classes, title, save_path=None, max_n=16):
    n = min(len(images), max_n)
    cols = 8
    rows = math.ceil(n/cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*1.6, rows*1.6))
    axes = axes.ravel()
    # unnormalize
    mean = np.array(CIFAR10_MEAN).reshape(3,1,1)
    std = np.array(CIFAR10_STD).reshape(3,1,1)
    for i in range(cols*rows):
        ax = axes[i]
        ax.axis("off")
        if i<n:
            img = images[i].cpu().numpy()  # C,H,W
            img = (img*std + mean)  # back to 0..1
            img = np.clip(img,0,1).transpose(1,2,0)
            ax.imshow(img)
            ax.set_title(classes[labels[i]], fontsize=8)
    plt.suptitle(title)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True) 
        plt.savefig(save_path, dpi=150)
    plt.close(fig)

def get_device(arg):
    if arg=="cpu": return torch.device("cpu")
    if arg=="cuda": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(model, loader, device, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for xb, yb in tqdm(loader, desc="train", leave=False):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()*xb.size(0)
        pred = out.argmax(1)
        correct += (pred==yb).sum().item()
        total += xb.size(0)
    return running_loss/total, correct/total

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for xb, yb in tqdm(loader, desc="eval", leave=False):
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        loss = criterion(out, yb)
        running_loss += loss.item()*xb.size(0)
        pred = out.argmax(1)
        correct += (pred==yb).sum().item()
        total += xb.size(0)
    return running_loss/total, correct/total

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="./data", type=str)
    p.add_argument("--epochs", default=25, type=int)
    p.add_argument("--batch-size", default=128, type=int)
    p.add_argument("--lr", default=1e-3, type=float)
    p.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
    p.add_argument("--workers", default=2, type=int)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--step-size", default=10, type=int)
    p.add_argument("--gamma", default=0.5, type=float)
    p.add_argument("--model", default="basic", choices=["basic","dilated"])
    args = p.parse_args()

    seed_all(args.seed)
    device = get_device(args.device)

    # datasets & loaders
    t_train = get_train_transforms()
    t_test = get_test_transforms()
    ds_train = AlbumentationsCIFAR10(args.data, train=True, download=True, transform=t_train)
    ds_test  = AlbumentationsCIFAR10(args.data, train=False, download=True, transform=t_test)

    classes = ds_train.ds.classes

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    test_loader  = DataLoader(ds_test,  batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # plot a few samples (raw test)
    xb0, yb0 = next(iter(test_loader))
    show_grid(xb0, yb0, classes, "Test samples (normalized)", save_path="results/plots/test_samples.png")

    # plot a few augmented samples (from train)
    xb1, yb1 = next(iter(train_loader))
    show_grid(xb1, yb1, classes, "Augmented train samples", save_path="results/plots/augmented_samples.png")

    model = Net().to(device) if args.model=="basic" else NetDilated().to(device)
    # print params
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    log_path = Path("results") / "train_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    df_log = []

    best_acc, best_ep = 0.0, -1

    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, device, criterion, optimizer)
        te_loss, te_acc = evaluate(model, test_loader, device, criterion)
        scheduler.step()

        dt = time.time()-t0
        print(f"Epoch {epoch:03d}: train acc {tr_acc:.4f} | test acc {te_acc:.4f} | time {dt:.1f}s")
        df_log.append({"epoch":epoch, "phase":"train", "loss":tr_loss, "acc":tr_acc})
        df_log.append({"epoch":epoch, "phase":"test",  "loss":te_loss, "acc":te_acc})

        if te_acc > best_acc:
            best_acc, best_ep = te_acc, epoch
            torch.save({"model":model.state_dict(), "acc":best_acc, "epoch":best_ep}, "results/best.pth")

    pd.DataFrame(df_log).to_csv(log_path, index=False)

    # simple accuracy curve
    import matplotlib.pyplot as plt
    df = pd.DataFrame(df_log)
    fig = plt.figure(figsize=(7,4))
    for phase in ["train","test"]:
        sub = df[df["phase"]==phase]
        plt.plot(sub["epoch"], sub["acc"], label=phase)
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy curves")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig("results/plots/acc_curves.png", dpi=150); plt.close(fig)

    print(f"Best test acc: {best_acc*100:.2f}% at epoch {best_ep}")

if __name__ == "__main__":
    main()


# '''
# # Basic (stride-2) model
# python train.py --epochs 25 --batch-size 128 --lr 0.001 --device auto --model basic

# # Dilated-only (no downsample) model
# python train.py --epochs 25 --batch-size 128 --lr 0.001 --device auto --model dilated
# '''