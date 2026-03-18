"""Training and test image transform pipelines."""

import math

import numpy as np
import torch
from torchvision import transforms as T

from utils.transforms import OverlappingCropMultiChannel
from utils.utils import convert_to_list


def get_train_transforms(
    random_crop_size: int = 1500,
    crop_size: int = 500,
    out_size: int = 256,
    without_rotation: bool = False,
) -> T.Compose:
    """Augmented pipeline: flips, rotation (if with_rotation=True), colour jitter, tiling, resize, normalise."""
    T1 = [T.RandomVerticalFlip(), T.RandomHorizontalFlip()]

    T_rot = [T.RandomRotation(90)]

    T2 = [
        T.Lambda(lambda image: [
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1)(image_slice)
            for image_slice in convert_to_list(image)
        ]),
        T.Lambda(lambda image_list: torch.stack(
            [image_slice for image_slice in image_list]
        ).squeeze(1)),
        T.CenterCrop(math.ceil(1.018 * random_crop_size)),
        T.RandomCrop(random_crop_size),
        OverlappingCropMultiChannel(crop_size, crop_size, pad=False),
        T.Lambda(lambda crops: torch.stack([crop for crop in crops])),
        T.Resize(out_size),
        T.Normalize(mean=0.5, std=0.5),
    ]

    if without_rotation:
        return T.Compose(T1 + T2)
    else:
        return T.Compose(T1 + T_rot + T2)


def get_test_transforms(
    random_crop_size: int = 1500,
    crop_size: int = 500,
    out_size: int = 256,
) -> T.Compose:
    """Deterministic pipeline: centre-crop, tile, resize, normalise."""
    return T.Compose([
        T.CenterCrop(random_crop_size),
        OverlappingCropMultiChannel(crop_size, crop_size, pad=False),
        T.Lambda(lambda crops: torch.stack([crop for crop in crops])),
        T.Resize(out_size),
        T.Normalize(mean=0.5, std=0.5),
    ])
