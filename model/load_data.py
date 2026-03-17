import torchvision
import torch
import os
import math
import numpy as np

from transforms import get_train_transforms, get_test_transforms
from dataloaders import TiffDataset
from dataset_class_params import get_class_merge_dict, get_moa_class_weights
from utils import get_class_weights

def load_data(root_dir,
              train_val_test_dir=[],
              dropped_classes=[],
              batch_size=16,
              val_split=0.2,
              crop_size=500,
              out_size=256,
              channels=None,
              use_e_coli_moa=False,
              use_class_weights=False,
              subsampling_factor=1.,
              bit_depth=8):
    """
    Return train, val and test dataloaders
    """

    if len(train_val_test_dir) > 0:
        if isinstance(train_val_test_dir[0], str):
            root_train = [os.path.join(root_dir, train_val_test_dir[0])]
        else:
            root_train = [os.path.join(root_dir, dir) for dir in train_val_test_dir[0]]

    train_dataset = TiffDataset(root=root_train,
                                bit_depth=bit_depth,
                                dropped_classes=dropped_classes,
                                transform=get_train_transforms(crop_size=crop_size, out_size=out_size),
                                channels=channels,
                                class_merge_dict=get_class_merge_dict(use_e_coli_moa),
                                subsampling_factor=subsampling_factor)

    if len(train_val_test_dir) > 0:
        if isinstance(train_val_test_dir[1], str):
            root_val = [os.path.join(root_dir, train_val_test_dir[1])]
        elif isinstance(train_val_test_dir[1], list):
            root_val = [os.path.join(root_dir, dir) for dir in train_val_test_dir[1]]
        elif train_val_test_dir[1] == None:
            train_size = int((1 - val_split) * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    if 'val_dataset' not in locals():
        val_dataset = TiffDataset(root=root_train,
                                  bit_depth=bit_depth,
                                  dropped_classes=dropped_classes,
                                  transform=get_train_transforms(crop_size=crop_size, out_size=out_size),
                                  channels=channels,
                                  class_merge_dict=get_class_merge_dict(use_e_coli_moa))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=12)

    if len(train_val_test_dir) > 0:
        if isinstance(train_val_test_dir[2], str):
            root_test = [os.path.join(root_dir, train_val_test_dir[2])]
        else:
            root_test = [os.path.join(root_dir, dir) for dir in train_val_test_dir[2]]

    test_dataset = TiffDataset(root=root_test,
                               bit_depth=bit_depth,
                               dropped_classes=dropped_classes,
                               transform=get_test_transforms(crop_size=crop_size, out_size=out_size),
                               channels=channels,
                               class_merge_dict=get_class_merge_dict(use_e_coli_moa))

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=12)

    if use_class_weights and not use_e_coli_moa:
        class_weights = get_class_weights(root_train)
    elif use_e_coli_moa:
        class_weights = get_moa_class_weights(test_dataset.classes)
    else:
        class_weights = [1 for __ in test_dataset.classes]

    return train_loader, val_loader, test_loader, test_dataset.classes, class_weights
