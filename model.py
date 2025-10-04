import torch
import torch.nn as nn
import torch.nn.functional as F

# Depthwise Separable Conv block: DW 3x3 -> PW 1x1
class DWSeparable(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dilation=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dw(x)   # RF += (3-1)*d*jump; jump *= stride
        x = self.pw(x)   # RF unchanged
        x = self.bn(x)
        return self.act(x)

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, k=3, dilation=1):
        super().__init__()
        pad = dilation if k==3 else 0
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=stride, padding=pad, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Net(nn.Module):
    """
    C1 C2 C3 C4 O
    - No MaxPool; exactly three 3x3 with stride=2 (downsample x2 each time)
    - DW-Separable used (not in first block)
    - Dilated conv in last block
    - GAP + Linear(10)
    Total params ~84k (excluding BN affine which is tiny), RF > 44.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        # ------- C1 ------- (no depthwise here)
        self.c1a = ConvBlock(3, 24, k=3, stride=1, dilation=1)   # RF 3
        self.c1b = ConvBlock(24, 32, k=3, stride=1, dilation=1)  # RF 5
        self.c1c = ConvBlock(32, 48, k=3, stride=2, dilation=1)  # RF 7  (downsample #1)

        # ------- C2 ------- (uses DW separable & one stride-2 3x3 via DW)
        self.c2a = DWSeparable(48, 64, stride=1, dilation=1)     # RF 11
        self.c2b = DWSeparable(64, 80, stride=2, dilation=1)     # RF 15 (downsample #2)

        # ------- C3 ------- (bottleneck then DW stride-2)
        self.c3a = ConvBlock(80, 96, k=1, stride=1, dilation=1)  # RF 15 (1x1 doesn't change RF)
        self.c3b = DWSeparable(96, 112, stride=2, dilation=1)    # RF 23 (downsample #3)

        # ------- C4 ------- (dilated DW conv toward last block)
        self.c4a = DWSeparable(112, 128, stride=1, dilation=2)   # RF 55 (dilated)
        self.c4b = ConvBlock(128, 128, k=1, stride=1, dilation=1)# RF 55

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # block 1
        x = self.c1a(x)    # RF 3
        x = self.c1b(x)    # RF 5
        x = self.c1c(x)    # RF 7
        # block 2
        x = self.c2a(x)    # RF 11
        x = self.c2b(x)    # RF 15
        # block 3
        x = self.c3a(x)    # RF 15
        x = self.c3b(x)    # RF 23
        # block 4
        x = self.c4a(x)    # RF 55 (>44)
        x = self.c4b(x)    # RF 55
        # O
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

class NetDilated(nn.Module):
    """
    Dilation-only variant (bonus):
      - No MaxPool; no stride > 1 anywhere (all stride=1)
      - RF grown via progressively larger dilations (2, 4, 8)
      - Includes Depthwise-Separable conv (C2 and C4)
      - GAP + Linear(num_classes)

    RF sketch (start RF=1, jump=1; each 3x3 adds 2*d):
      C1: d=1,1   -> +2 +2            -> RF 5
      C2: d=2,2   -> +4 (DW) +4       -> RF 13
      C3: d=4,4   -> +8 +8            -> RF 29
      C4: d=8     -> +16 (DW)         -> RF 45  (>= 45 ✅)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        # ---- C1 ---- (keep modest widths)
        self.c1a = ConvBlock(3,  24, k=3, stride=1, dilation=1)    # RF 3
        self.c1b = ConvBlock(24, 32, k=3, stride=1, dilation=1)    # RF 5

        # ---- C2 ---- (DW-sep with d=2, then 1x1, then 3x3 d=2)
        self.c2a_dw = DWSeparable(32, 48, stride=1, dilation=2)    # RF 9  (+4)
        self.c2b_pw = ConvBlock(48, 48, k=1, stride=1, dilation=1) # RF 9
        self.c2c    = ConvBlock(48, 64, k=3, stride=1, dilation=2) # RF 13 (+4)

        # ---- C3 ---- (two 3x3 with d=4)
        self.c3a_pw = ConvBlock(64, 80, k=1, stride=1, dilation=1) # RF 13
        self.c3b    = ConvBlock(80, 80, k=3, stride=1, dilation=4) # RF 21 (+8)
        self.c3c    = ConvBlock(80, 96, k=3, stride=1, dilation=4) # RF 29 (+8)

        # ---- C4 ---- (DW-sep with strong dilation=8)
        self.c4a    = DWSeparable(96, 112, stride=1, dilation=8)   # RF 45 (+16)
        self.c4b_pw = ConvBlock(112,112, k=1, stride=1, dilation=1)# RF 45

        # ---- Head ----
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Linear(112, num_classes)  # GAP output 112-D

    def forward(self, x):
        # C1
        x = self.c1a(x); x = self.c1b(x)
        # C2
        x = self.c2a_dw(x); x = self.c2b_pw(x); x = self.c2c(x)
        # C3
        x = self.c3a_pw(x); x = self.c3b(x); x = self.c3c(x)
        # C4
        x = self.c4a(x); x = self.c4b_pw(x)
        # O
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.fc(x)