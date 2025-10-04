from typing import Optional, Callable
from torch.utils.data import Dataset
from torchvision import datasets
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# CIFAR-10 channel statistics
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


def get_train_transforms():
    """
    Albumentations training pipeline:
      - Horizontal flip
      - Affine (translate/scale/rotate)  [recommended over ShiftScaleRotate]
      - CoarseDropout with your installed version's signature:
          (num_holes_range, hole_height_range, hole_width_range, fill, fill_mask, p)
      - Normalize + ToTensorV2
    """
    # Albumentations "fill" expects 0..255 values; use per-channel dataset mean
    mean255 = tuple(int(m * 255) for m in CIFAR10_MEAN)

    return A.Compose([
        A.HorizontalFlip(p=0.5),

        # Preferred over ShiftScaleRotate (removes deprecation warning)
       A.Affine(
                translate_percent={"x": 0.0625, "y": 0.0625},
                scale=(0.9, 1.1),
                rotate=(-15, 15),
                fit_output=False,
                border_mode=cv2.BORDER_REFLECT_101,  # ✅ correct param
                p=0.5,
            ),


        # Matches your printed signature:
        # (num_holes_range, hole_height_range, hole_width_range, fill, fill_mask, p)
        A.CoarseDropout(
            num_holes_range=(1, 1),
            hole_height_range=(16, 16),  # absolute pixels
            hole_width_range=(16, 16),   # absolute pixels
            fill=mean255,                # per-channel mean in 0..255
            fill_mask=None,
            p=0.5,
        ),

        A.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ToTensorV2()
    ])


def get_test_transforms():
    """ Evaluation pipeline: only Normalize + ToTensorV2. """
    return A.Compose([
        A.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ToTensorV2()
    ])


class AlbumentationsCIFAR10(Dataset):
    """
    torchvision CIFAR-10 wrapped to apply Albumentations on-the-fly.
    """
    def __init__(self, root: str, train: bool, download: bool = True,
                 transform: Optional[Callable] = None):
        self.ds = datasets.CIFAR10(root=root, train=train, download=download)
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        img, label = self.ds[idx]      # PIL RGB
        img = np.array(img)            # -> numpy RGB [H,W,C], uint8 0..255

        if self.transform is not None:
            img = self.transform(image=img)["image"]  # -> torch.FloatTensor [C,H,W]

        return img, label
