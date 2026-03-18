"""Shared utility functions: directory resolution, dict merging, class weights, conversion to list."""

import os
import random
import torch


def parse_train_val_test_dir(data_dir, train_dir, test_dir, val_dir, dose) -> list:
    """Resolve train/val/test dirs. val_dir=['None'] → None (random split); [] → auto-select one."""
    all_dir = os.listdir(data_dir)

    if dose:
        all_dir = [os.path.join(d, dose) for d in all_dir]

    if val_dir[0] == 'None':
        val_dir = None
    elif len(val_dir) == 0:
        val_dir = [random.choice([dir for dir in all_dir if dir not in test_dir])]

    if len(train_dir) == 0:
        if val_dir:
            train_dir = [dir for dir in all_dir if dir not in test_dir + val_dir]
        else:
            train_dir = [dir for dir in all_dir if dir not in test_dir]

    return [train_dir, val_dir, test_dir]


def intersect_dicts(class_merge_dict: dict, moa_dict: dict) -> dict:
    """Remap class_merge_dict values through moa_dict, then union with moa_dict."""
    intrsct_dict = dict()
    for k, v in class_merge_dict.items():
        if v in moa_dict.keys():
            intrsct_dict[k] = moa_dict[v]

    return {**intrsct_dict, **moa_dict}


def get_class_weights(root_train: list, dropped_classes: list, class_merge_dict: dict) -> list:
    """Return inverse-frequency class weights across all roots after dropping and merging."""
    class_dirs = []
    for dir in root_train:
        class_dirs += os.listdir(dir)
    class_dirs = [x for x in class_dirs if x not in dropped_classes]
    class_dirs = [
        class_merge_dict[x] if x in class_merge_dict.keys() else x for x in class_dirs
    ]
    class_weights = [1 / class_dirs.count(x) for x in sorted(list(set(class_dirs)))]

    return class_weights


def convert_to_list(x: torch.Tensor) -> list:
    """Return x split along dim 0, each slice unsqueezed to (1, H, W)."""
    return [x_i.unsqueeze(0) for x_i in x]
