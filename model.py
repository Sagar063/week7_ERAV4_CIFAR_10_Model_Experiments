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
    Dilated-only variant (bonus): no MaxPool, no stride>1 anywhere.
    We expand receptive field via increasing dilations and keep spatial size at 32x32,
    then use GAP -> Linear(10). Includes a DW-separable conv (not in block 1).

    RF math (start RF=1, jump=1; each 3x3 adds 2*d):
      B1: d=[1,1] -> +2 +2 = +4 -> RF=5
      B2: d=[2,2] -> +4 +4 = +8 -> RF=13 (DW sep used here)
      B3: d=[4,4] -> +8 +8 = +16 -> RF=29
      B4: d=[8]   -> +16         -> RF=45 (>44)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        # Keep channels modest to stay well under 200k parameters.
        # Use 1x1 to compress/expand without affecting RF.
        # ------- C1 ------- (standard convs; no DW here)
        self.c1a = ConvBlock(3,  32, k=3, stride=1, dilation=1)   # RF 3
        self.c1b = ConvBlock(32, 48, k=3, stride=1, dilation=1)   # RF 5

        # ------- C2 ------- (DW-separable + dilation=2, then PW)
        self.c2a_dw = DWSeparable(48, 64, stride=1, dilation=2)   # RF 9  (+4)
        self.c2b_pw = ConvBlock(64, 64, k=1, stride=1, dilation=1)# RF 9
        self.c2c    = ConvBlock(64, 80, k=3, stride=1, dilation=2)# RF 13 (+4)

        # ------- C3 ------- (dilation=4)
        self.c3a_pw = ConvBlock(80, 96, k=1, stride=1, dilation=1)# RF 13
        self.c3b    = ConvBlock(96, 96, k=3, stride=1, dilation=4)# RF 21 (+8)
        self.c3c    = ConvBlock(96,112, k=3, stride=1, dilation=4)# RF 29 (+8)

        # ------- C4 ------- (strong dilation=8)
        self.c4a    = DWSeparable(112,128, stride=1, dilation=8)  # RF 45 (+16)

        # Output head
        self.c4b_pw = ConvBlock(128,128, k=1, stride=1, dilation=1)# RF 45
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Linear(128, num_classes)

    def forward(self, x):
        # block 1
        x = self.c1a(x)    # RF 3
        x = self.c1b(x)    # RF 5
        # block 2
        x = self.c2a_dw(x) # RF 9
        x = self.c2b_pw(x) # RF 9
        x = self.c2c(x)    # RF 13
        # block 3
        x = self.c3a_pw(x) # RF 13
        x = self.c3b(x)    # RF 21
        x = self.c3c(x)    # RF 29
        # block 4
        x = self.c4a(x)    # RF 45 (>44)
        x = self.c4b_pw(x) # RF 45
        # O
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.fc(x)