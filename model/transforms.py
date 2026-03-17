from torchvision import transforms as T
import torch

from utils import OverlappingCropMultiChannel, convert_to_list

def get_train_transforms(crop_size=500, out_size=256):
    return T.Compose([
        T.RandomVerticalFlip(),
        T.RandomHorizontalFlip(),
        T.RandomRotation(90),
        T.Lambda(lambda image: [T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1)(image_slice) for image_slice in convert_to_list(image)]),
        T.Lambda(lambda image_list: torch.stack([image_slice for image_slice in image_list]).squeeze(1)),
        T.CenterCrop(math.ceil(np.sqrt(0.5) * 2160)),
        T.RandomCrop(1500),
        OverlappingCropMultiChannel(crop_size, crop_size, pad=False),
        T.Lambda(lambda crops: torch.stack([crop for crop in crops])),
        T.Resize(out_size),
        T.Normalize(mean=0.5, std=0.5)
    ])

def get_test_transforms(crop_size=500, out_size=256):
    return T.Compose([
        T.CenterCrop(1500),
        OverlappingCropMultiChannel(crop_size, crop_size, pad=False),
        T.Lambda(lambda crops: torch.stack([crop for crop in crops])),
        T.Resize(out_size),
        T.Normalize(mean=0.5, std=0.5)
    ])
