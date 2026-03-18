"""Build train, validation, and test DataLoaders from a TrainingConfig."""

import os

import torch

from config import NUM_WORKERS, TrainingConfig
from data.dataset import ImageDataset
from data.class_params import get_class_merge_dict, get_moa_class_weights
from data.transforms import get_test_transforms, get_train_transforms
from utils.utils import get_class_weights


def load_data(cfg: TrainingConfig):
    """Return (train_loader, val_loader, test_loader, classes, class_weights).

    cfg.train_val_test_dir[1]: str/list = dedicated val dir; None = random split; absent = reuse train root.
    """
    root_dir = cfg.data_dir
    train_val_test_dir = cfg.train_val_test_dir
    merge_dict = get_class_merge_dict(cfg.use_e_coli_moa)

    if isinstance(train_val_test_dir[0], str):
        root_train = [os.path.join(root_dir, train_val_test_dir[0])]
    else:
        root_train = [os.path.join(root_dir, d) for d in train_val_test_dir[0]]

    train_dataset = ImageDataset(
        root=root_train,
        bit_depth=cfg.im_bit_depth,
        dropped_classes=cfg.dropped_classes,
        transform=get_train_transforms(random_crop_size=cfg.full_image_size, crop_size=cfg.crop_size, out_size=cfg.out_size, without_rotation=cfg.aug_wo_rand_rot),
        channels=cfg.channels,
        class_merge_dict=merge_dict,
        subsampling_factor=cfg.subsampling_factor,
    )

    val_dataset = None

    if isinstance(train_val_test_dir[1], str):
        root_val = [os.path.join(root_dir, train_val_test_dir[1])]
        val_dataset = ImageDataset(
            root=root_val,
            root_all=root_train,
            bit_depth=cfg.im_bit_depth,
            dropped_classes=cfg.dropped_classes,
            transform=get_train_transforms(random_crop_size=cfg.full_image_size, crop_size=cfg.crop_size, out_size=cfg.out_size, without_rotation=cfg.aug_wo_rand_rot),
            channels=cfg.channels,
            class_merge_dict=merge_dict,
        )
    elif isinstance(train_val_test_dir[1], list):
        root_val = [os.path.join(root_dir, d) for d in train_val_test_dir[1]]
        val_dataset = ImageDataset(
            root=root_val,
            root_all=root_train,
            bit_depth=cfg.im_bit_depth,
            dropped_classes=cfg.dropped_classes,
            transform=get_train_transforms(random_crop_size=cfg.full_image_size, crop_size=cfg.crop_size, out_size=cfg.out_size, without_rotation=cfg.aug_wo_rand_rot),
            channels=cfg.channels,
            class_merge_dict=merge_dict,
        )
    elif train_val_test_dir[1] is None:
        train_size = int((1 - cfg.val_split) * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )

    if val_dataset is None:
        val_dataset = ImageDataset(
            root=root_train,
            bit_depth=cfg.im_bit_depth,
            dropped_classes=cfg.dropped_classes,
            transform=get_train_transforms(random_crop_size=cfg.full_image_size, crop_size=cfg.crop_size, out_size=cfg.out_size, without_rotation=cfg.aug_wo_rand_rot),
            channels=cfg.channels,
            class_merge_dict=merge_dict,
        )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=NUM_WORKERS
    )

    if isinstance(train_val_test_dir[2], str):
        root_test = [os.path.join(root_dir, train_val_test_dir[2])]
    else:
        root_test = [os.path.join(root_dir, d) for d in train_val_test_dir[2]]

    test_dataset = ImageDataset(
        root=root_test,
        root_all=root_train,
        bit_depth=cfg.im_bit_depth,
        dropped_classes=cfg.dropped_classes,
        transform=get_test_transforms(random_crop_size=cfg.full_image_size, crop_size=cfg.crop_size, out_size=cfg.out_size),
        channels=cfg.channels,
        class_merge_dict=merge_dict,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=NUM_WORKERS
    )

    if cfg.use_class_weights and not cfg.use_e_coli_moa:
        class_weights = get_class_weights(root_train, cfg.dropped_classes, merge_dict)
    elif cfg.use_e_coli_moa:
        class_weights = get_moa_class_weights(test_dataset.classes)
    else:
        class_weights = [1 for __ in test_dataset.classes]

    return train_loader, val_loader, test_loader, test_dataset.classes, class_weights
