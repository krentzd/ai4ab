from torchvision import datasets, transforms
import torchvision
import torch
import os
import math
import numpy as np

from dataloaders import TiffDataset, ClassSpecificImageFolder
from utils import OverlappingCrop, OverlappingCropMultiChannel

def convert_to_list(x):
    return [x_i.unsqueeze(0) for x_i in x]

def dataset_loader(type='tiff'):
    if type == 'tiff':
        return TiffDataset
    elif type == 'png':
        return ClassSpecificImageFolder

def load_data(root_dir, train_test_dir=[], dropped_classes=[], batch_size=32, val_split=0.2, crop_size=512, **kwargs):
    """
    Return train, val and test dataloaders
    """

    data_mean = 0.5
    data_std = 0.5

    # Generate n crops per image
    train_transforms_list = [transforms.RandomVerticalFlip(),
                             transforms.RandomHorizontalFlip(),
                             transforms.RandomRotation(90),
                             transforms.Lambda(lambda image: [transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1)(image_slice) for image_slice in convert_to_list(image)]), #Apply color jitter to channels individually, since they're uncorrelated
                             transforms.Lambda(lambda image_list: torch.stack([image_slice for image_slice in image_list]).squeeze(1)),
                             transforms.CenterCrop(math.ceil(np.sqrt(0.5) * 2160)),
                             transforms.RandomCrop(1500), # = Slightly smaller random crop
                             OverlappingCropMultiChannel(crop_size, crop_size, pad=False),
                             transforms.Lambda(lambda crops: torch.stack([crop for crop in crops])),
                             transforms.Resize((256, 256)),
                             transforms.Normalize(mean=data_mean, std=data_std)]

    train_transforms_list_png = [transforms.RandomVerticalFlip(),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.RandomRotation(90),
                                 transforms.CenterCrop(math.ceil(np.sqrt(0.5) * 2160)), # = 1528
                                 transforms.RandomCrop(1500), # = Slightly smaller random crop
                                 OverlappingCrop(crop_size, crop_size, pad=False),
                                 transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                 transforms.Resize((256, 256)),
                                 transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1),
                                 # transforms.Grayscale(num_output_channels=1),
                                 transforms.Normalize(mean=data_mean, std=data_std)]

    test_transforms_list = [transforms.CenterCrop(1500),
                            OverlappingCropMultiChannel(crop_size, crop_size, pad=False),
                            transforms.Lambda(lambda crops: torch.stack([crop for crop in crops])),
                            transforms.Resize((256, 256)),
                            transforms.Normalize(mean=data_mean, std=data_std)]

    test_transforms_list_png = [transforms.CenterCrop(1500),
                                OverlappingCrop(crop_size, crop_size, pad=False),
                                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                transforms.Resize((256, 256)),
                                # transforms.Grayscale(num_output_channels=1),
                                transforms.Normalize(mean=data_mean, std=data_std)]

    train_transforms =  transforms.Compose(train_transforms_list)
    test_transforms =  transforms.Compose(test_transforms_list)

    train_transforms_png =  transforms.Compose(train_transforms_list_png)
    test_transforms_png =  transforms.Compose(test_transforms_list_png)

    if len(train_test_dir) > 0:
        if isinstance(train_test_dir[0], str):
            root_train = [os.path.join(root_dir, train_test_dir[0])]
        else:
            print(train_test_dir[0])
            root_train = [os.path.join(root_dir, dir) for dir in train_test_dir[0]]

    full_dataset = dataset_loader(type=kwargs.get('data_type', 'tiff'))(root=root_train,
                                                                        dropped_classes=dropped_classes,
                                                                        transform=train_transforms if kwargs.get('data_type', 'tiff') == 'tiff' else train_transforms_png,
                                                                        channels=kwargs.get('channels', None))

    # full_dataset = TiffDataset(root=root_train,
    #                            dropped_classes=dropped_classes,
    #                            transform=train_transforms,
    #                            channels=kwargs.get('channels', None))

    train_size = int((1 - val_split) * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=16)

    if len(train_test_dir) > 0:
        if isinstance(train_test_dir[1], str):
            root_test = [os.path.join(root_dir, train_test_dir[1])]
        else:
            root_test = [os.path.join(root_dir, dir) for dir in train_test_dir[1]]

    test_dataset = dataset_loader(type=kwargs.get('data_type', 'tiff'))(root=root_test,
                                                                        dropped_classes=dropped_classes,
                                                                        transform=test_transforms if kwargs.get('data_type', 'tiff') == 'tiff' else test_transforms_png,
                                                                        channels=kwargs.get('channels', None))
    # test_dataset = TiffDataset(root=root_test,
    #                            dropped_classes=dropped_classes,
    #                            transform=test_transforms,
    #                            channels=kwargs.get('channels', None))

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=16)

    print('Train loader', len(train_loader.dataset), 'Val loader', len(val_loader.dataset), 'Test loader', len(test_loader.dataset))

    return train_loader, val_loader, test_loader, test_dataset.classes
