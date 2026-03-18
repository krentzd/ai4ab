"""TIFF and PNG image dataset with class merging, dropping, channel selection, and subsampling.

Loaded arrays are normalised to (C, H, W) regardless of source format or channel ordering.
"""

import math
import os
import random
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

import numpy as np
import tifffile
import torch
from PIL import Image
from torchvision import datasets

IMG_EXTENSIONS = ('.tif', '.tiff', '.png')


def has_file_allowed_extension(
    filename: str,
    extensions: Union[str, Tuple[str, ...]],
) -> bool:
    """Return True if filename ends with one of the given extensions (case-insensitive)."""
    return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))


def merged_class_to_idx(
    classes: List[str],
    class_merge_dict: Dict[str, str],
) -> Tuple[Dict[str, int], List[str]]:
    """Build class_to_idx with merging: folders sharing a merged label share an index."""
    class_to_idx: Dict[str, int] = {}
    idx_val = 0
    classes_: List[str] = []

    for c in sorted(classes):
        if c in class_merge_dict.keys():
            resolved_labels = [
                class_merge_dict[k] if k in class_merge_dict else k
                for k in class_to_idx.keys()
            ]
            if class_merge_dict[c] not in resolved_labels:
                class_to_idx[c] = idx_val
                idx_val += 1
                classes_.append(class_merge_dict[c])
            else:
                idx_ = resolved_labels.index(class_merge_dict[c])
                idx_val_ = list(class_to_idx.values())[idx_]
                class_to_idx[c] = idx_val_
        else:
            class_to_idx[c] = idx_val
            idx_val += 1
            classes_.append(c)

    return class_to_idx, classes_


def make_dataset(
    directory: str,
    class_to_idx: Optional[Dict[str, int]] = None,
    extensions: Optional[Union[str, Tuple[str, ...]]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
    subsampling_factor: float = 1.,
) -> List[Tuple[str, int]]:
    """Walk class subdirectories and return (path, class_index) samples. Raises ValueError/FileNotFoundError on bad args or missing files."""
    directory = [os.path.expanduser(dir) for dir in directory]

    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError(
            "Both extensions and is_valid_file cannot be None or not None at the same time"
        )

    if extensions is not None:
        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, extensions)

    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    instances = []

    for dir in directory:
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(dir, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                fnames = random.sample(fnames, math.floor(subsampling_factor * len(fnames)))
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = path, class_index
                        instances.append(item)

    return instances


class ImageDataset(datasets.VisionDataset):
    """Multi-channel TIFF and PNG dataset with class merging, dropping, and subsampling.

    Expects root/class_name/image.{tif,tiff,png} structure. Multiple roots must share identical class subdirectories.
    """

    def __init__(
        self,
        root,
        root_all=None,
        bit_depth: int = 8,
        extensions=IMG_EXTENSIONS,
        transform=None,
        dropped_classes: List[str] = [],
        class_merge_dict: Optional[Dict[str, str]] = None,
        subsampling_factor: float = 1.,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        channels: Optional[List[int]] = None,
    ) -> None:
        super().__init__(root, transform=transform)

        if isinstance(root, str):
            root = [root]

        if len(root) > 1:
            assert all(
                i for i in [os.path.isdir(d) for d in root]
            ), 'Root is not a directory'

        self.dropped_classes = dropped_classes
        self.class_merge_dict = class_merge_dict
        self.subsampling_factor = subsampling_factor

        self.channels = channels
        if self.channels:
            self.channels = [int(ch) for ch in self.channels]

        if root_all is None:
            root_all = root

        classes, class_to_idx = self.find_classes(root_all)
        samples = self.make_dataset(root, class_to_idx, extensions, is_valid_file)

        self.extensions = extensions
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.bit_depth = bit_depth

    def make_dataset(
        self,
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        """Delegate to module-level make_dataset."""
        if class_to_idx is None:
            raise ValueError("The class_to_idx parameter cannot be None.")
        return make_dataset(
            directory,
            class_to_idx,
            extensions=extensions,
            is_valid_file=is_valid_file,
            subsampling_factor=self.subsampling_factor,
        )

    def find_classes(self, directory) -> Tuple[List[str], Dict[str, int]]:
        """Scan the first root directory to build the class list and index mapping."""

        classes = []
        for directory_ in directory:
            for entry in os.scandir(directory_):
                if entry.is_dir() and entry.name not in classes:
                    classes.append(entry.name)
        classes = sorted(classes)
        classes = [c for c in classes if c not in self.dropped_classes]
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        if self.class_merge_dict:
            class_to_idx, classes = merged_class_to_idx(classes, self.class_merge_dict)
        else:
            class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        return classes, class_to_idx

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Load, normalise, and transform the sample at index."""
        path, target = self.samples[index]

        if path.lower().endswith('.png'):
            sample = np.array(Image.open(path))
        else:
            sample = tifffile.imread(path)

        # Normalise to (C, H, W): add channel dim if 2D, transpose if channels-last.
        if sample.ndim == 2:
            sample = sample[np.newaxis, ...]
        elif sample.ndim == 3 and sample.shape[2] < sample.shape[0]:
            sample = sample.transpose(2, 0, 1)

        if self.channels:
            sample = sample[self.channels]

        sample = torch.FloatTensor(sample / (2 ** self.bit_depth - 1))

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)
