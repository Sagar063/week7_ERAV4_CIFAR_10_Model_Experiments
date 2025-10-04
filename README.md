# CIFAR-10: C1C2C3C4O (No MaxPool, 3× 3×3 stride-2), DW-Separable & Dilated

This repo trains a compact CNN on CIFAR-10 with the following constraints:
- **No MaxPooling** (we use exactly three `3×3, stride=2` layers to downsample).
- One or more **Depthwise Separable Convs** (not in block 1).
- One **Dilated Conv** in the last block.
- **GAP** (Global Average Pooling). A final FC after GAP is used to get logits for 10 classes.
- **On-the-fly augmentation** via Albumentations: HorizontalFlip, ShiftScaleRotate, CoarseDropout.
- Target: **≥85%** test accuracy, **<200k** params (this model is ~66k).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate  # (Linux/Mac)
pip install -r requirements.txt

python train.py --epochs 25 --batch-size 128 --lr 0.001 --device auto
```

Training artifacts go to `results/` (CSV log and plots). Run `update_readme.py` to inject latest results into this README.

## Receptive Field (RF) running tally

We track RF using: `RF_out = RF_in + (k-1)*d*jump`, `jump_out = jump_in * stride` (start RF=1, jump=1).

The model’s main spatial ops and cumulative RF:
- B1a: `3×3 s1 d1` → RF=3, jump=1  
- B1b: `3×3 s1 d1` → RF=5, jump=1  
- B1c: `3×3 s2 d1` → RF=7, jump=2  ← **(1st downsample)**
- B2a: `DW 3×3 s1 d1` → RF=11, jump=2  
- B2b: `PW 1×1 s1` → RF=11, jump=2  
- B2c: `DW 3×3 s2 d1` → RF=15, jump=4  ← **(2nd downsample)**
- B2d: `PW 1×1 s1` → RF=15, jump=4  
- B3a: `PW 1×1 s1` → RF=15, jump=4  
- B3b: `DW 3×3 s2 d1` → RF=23, jump=8  ← **(3rd downsample)**
- B3c: `PW 1×1 s1` → RF=23, jump=8  
- B4a: `DW 3×3 s1 d2` → RF=55, jump=8  ← **(dilated; RF > 44)**
- B4b: `PW 1×1 s1` → RF=55, jump=8  

After B4, the feature map is 4×4 (from 32→16→8→4), then **GAP** → 1×1, and a small FC → 10 classes.

> Why can FC after GAP be “optional”? In many modern CNNs, GAP is followed by a **linear classifier** (1×1 conv or FC) to map channels→classes. Strictly speaking, if the last conv already outputs `C=10` channels, you can compute logits by **GAP alone**. In practice, we keep a tiny linear layer after GAP for flexibility.


**Latest Test Accuracy:** 59.80% at epoch 2


**Latest Test Accuracy:** 59.80% at epoch 2

<!-- RESULTS -->
| Metric | Value |
|---|---:|
| **Best Test Acc** | **74.91%** (epoch 7) |
| Best Test Loss | 0.7063 |
| Last Train Acc | 68.68% (epoch 7) |
| Last Train Loss | 0.8833 |
| Updated | 2025-10-04 18:07:35 |
<!-- /RESULTS -->

<!-- PLOTS -->
<p><strong>Test samples (grid)</strong><br>
<img src="results/plots/test_samples_grid.png" alt="Test samples (grid)" width="720"></p>

<p><strong>Augmented train samples (grid)</strong><br>
<img src="results/plots/augmented_samples_grid.png" alt="Augmented train samples (grid)" width="720"></p>

<p><strong>Accuracy curves</strong><br>
<img src="results/plots/acc_curves.png" alt="Accuracy curves" width="720"></p>

<p><strong>Loss curves</strong><br>
<img src="results/plots/loss_curves.png" alt="Loss curves" width="720"></p>

<p><strong>Confusion matrix (normalized)</strong><br>
<img src="results/plots/cm.png" alt="Confusion matrix (normalized)" width="720"></p>
<!-- /PLOTS -->

<!-- TILE_GALLERY -->
<h4>Test sample tiles</h4>
<p>
<img src="results/plots/test_samples/img_00_cat.png" width="128" style="margin:4px;">
<img src="results/plots/test_samples/img_01_ship.png" width="128" style="margin:4px;">
<img src="results/plots/test_samples/img_02_ship.png" width="128" style="margin:4px;">
<img src="results/plots/test_samples/img_03_airplane.png" width="128" style="margin:4px;">
<img src="results/plots/test_samples/img_04_frog.png" width="128" style="margin:4px;">
<img src="results/plots/test_samples/img_05_frog.png" width="128" style="margin:4px;">
<img src="results/plots/test_samples/img_06_automobile.png" width="128" style="margin:4px;">
<img src="results/plots/test_samples/img_07_frog.png" width="128" style="margin:4px;">
</p>

<h4>Augmented sample tiles</h4>
<p>
<img src="results/plots/augmented_samples/img_00_bird.png" width="128" style="margin:4px;">
<img src="results/plots/augmented_samples/img_01_cat.png" width="128" style="margin:4px;">
<img src="results/plots/augmented_samples/img_02_truck.png" width="128" style="margin:4px;">
<img src="results/plots/augmented_samples/img_03_deer.png" width="128" style="margin:4px;">
<img src="results/plots/augmented_samples/img_04_ship.png" width="128" style="margin:4px;">
<img src="results/plots/augmented_samples/img_05_deer.png" width="128" style="margin:4px;">
<img src="results/plots/augmented_samples/img_06_frog.png" width="128" style="margin:4px;">
<img src="results/plots/augmented_samples/img_07_bird.png" width="128" style="margin:4px;">
</p>
<!-- /TILE_GALLERY -->

<!-- VISUALIZATION -->
**Augmentation Visualization**

Use the helper to render a crisp grid and per-image tiles (nearest-neighbor, no blur):

```bash
python visualize_augmentations.py --save results/plots/aug_demo.png
```
_Why_: sanity-check your Albumentations pipeline and confirm computed mean/std are used.
<!-- /VISUALIZATION -->

<!-- MODEL_SUMMARY -->
<details><summary><b>Model summary (click to expand)</b></summary>

```
Trainable parameters: 84322

==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
Net                                      [1, 10]                   --
├─ConvBlock: 1-1                         [1, 24, 32, 32]           --
│    └─Conv2d: 2-1                       [1, 24, 32, 32]           648
│    └─BatchNorm2d: 2-2                  [1, 24, 32, 32]           48
│    └─ReLU: 2-3                         [1, 24, 32, 32]           --
├─ConvBlock: 1-2                         [1, 32, 32, 32]           --
│    └─Conv2d: 2-4                       [1, 32, 32, 32]           6,912
│    └─BatchNorm2d: 2-5                  [1, 32, 32, 32]           64
│    └─ReLU: 2-6                         [1, 32, 32, 32]           --
├─ConvBlock: 1-3                         [1, 48, 16, 16]           --
│    └─Conv2d: 2-7                       [1, 48, 16, 16]           13,824
│    └─BatchNorm2d: 2-8                  [1, 48, 16, 16]           96
│    └─ReLU: 2-9                         [1, 48, 16, 16]           --
├─DWSeparable: 1-4                       [1, 64, 16, 16]           --
│    └─Conv2d: 2-10                      [1, 48, 16, 16]           432
│    └─Conv2d: 2-11                      [1, 64, 16, 16]           3,072
│    └─BatchNorm2d: 2-12                 [1, 64, 16, 16]           128
│    └─ReLU: 2-13                        [1, 64, 16, 16]           --
├─DWSeparable: 1-5                       [1, 80, 8, 8]             --
│    └─Conv2d: 2-14                      [1, 64, 8, 8]             576
│    └─Conv2d: 2-15                      [1, 80, 8, 8]             5,120
│    └─BatchNorm2d: 2-16                 [1, 80, 8, 8]             160
│    └─ReLU: 2-17                        [1, 80, 8, 8]             --
├─ConvBlock: 1-6                         [1, 96, 8, 8]             --
│    └─Conv2d: 2-18                      [1, 96, 8, 8]             7,680
│    └─BatchNorm2d: 2-19                 [1, 96, 8, 8]             192
│    └─ReLU: 2-20                        [1, 96, 8, 8]             --
├─DWSeparable: 1-7                       [1, 112, 4, 4]            --
│    └─Conv2d: 2-21                      [1, 96, 4, 4]             864
│    └─Conv2d: 2-22                      [1, 112, 4, 4]            10,752
│    └─BatchNorm2d: 2-23                 [1, 112, 4, 4]            224
│    └─ReLU: 2-24                        [1, 112, 4, 4]            --
├─DWSeparable: 1-8                       [1, 128, 4, 4]            --
│    └─Conv2d: 2-25                      [1, 112, 4, 4]            1,008
│    └─Conv2d: 2-26                      [1, 128, 4, 4]            14,336
│    └─BatchNorm2d: 2-27                 [1, 128, 4, 4]            256
│    └─ReLU: 2-28                        [1, 128, 4, 4]            --
├─ConvBlock: 1-9                         [1, 128, 4, 4]            --
│    └─Conv2d: 2-29                      [1, 128, 4, 4]            16,384
│    └─BatchNorm2d: 2-30                 [1, 128, 4, 4]            256
│    └─ReLU: 2-31                        [1, 128, 4, 4]            --
├─AdaptiveAvgPool2d: 1-10                [1, 128, 1, 1]            --
├─Linear: 1-11                           [1, 10]                   1,290
==========================================================================================
Total params: 84,322
Trainable params: 84,322
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 13.73
==========================================================================================
Input size (MB): 0.01
Forward/backward pass size (MB): 1.81
Params size (MB): 0.34
Estimated Total Size (MB): 2.16
==========================================================================================
```

</details>
<!-- /MODEL_SUMMARY -->
