import os
import tifffile
import numpy as np
from torchvision import datasets
import torch
import random
import math
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

IMG_EXTENSIONS = ('.tif', '.tiff')

def has_file_allowed_extension(filename: str, extensions: Union[str, Tuple[str, ...]]) -> bool:

    return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))

def merged_class_to_idx(classes, class_merge_dict):
    """ Index classes based on class_merge_dict"""
    class_to_idx = dict()
    idx_val = 0
    classes_ = []
    for c in sorted(classes):
        if c in class_merge_dict.keys():
            if class_merge_dict[c] not in [class_merge_dict[c] if c in class_merge_dict.keys() else c for c in class_to_idx.keys()]:
                class_to_idx[c] = idx_val
                idx_val += 1
                classes_.append(class_merge_dict[c])
            else:
                idx_ = [class_merge_dict[c] if c in class_merge_dict.keys() else c for c in class_to_idx.keys()].index(class_merge_dict[c])
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
    subsampling_factor=1.,
) -> List[Tuple[str, int]]:

    directory = [os.path.expanduser(dir) for dir in directory]

    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

    if extensions is not None:

        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    instances = []
    available_classes = set()
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
                        #TODO: implement switch for testing mode!
                        item = path, class_index #(class_index, path) --> only for testing
                        instances.append(item)

                        if target_class not in available_classes:
                            available_classes.add(target_class)

    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        if extensions is not None:
            msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
        raise FileNotFoundError(msg)

    return instances

class TiffDataset(datasets.VisionDataset):

    def __init__(
        self,
        root,
        bit_depth=8,
        extensions=IMG_EXTENSIONS,
        transform=None,
        dropped_classes=[],
        class_merge_dict=None,
        subsampling_factor=1.,
        is_valid_file=None,
        channels=None
    ):

        super().__init__(root, transform=transform)

        # Check that all root directories have same number of sub-directories (classes)
        if isinstance(root, str):
            root = [root]

        if len(root) > 1:
            assert all(i for i in [os.path.isdir(d) for d in root]), 'Root is not a directory'
        if len(root) > 1:
            assert  all([l == len(next(os.walk(root[0]))[1]) for l in [len(next(os.walk(dir))[1]) for dir in root]]), 'Root directories of replicates contain different numbers of classes!'

        self.dropped_classes = dropped_classes
        self.class_merge_dict = class_merge_dict
        self.subsampling_factor = subsampling_factor

        self.channels = channels
        if self.channels:
            self.channels = [int(ch) for ch in self.channels]

        classes, class_to_idx = self.find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)

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

        if class_to_idx is None:
            raise ValueError("The class_to_idx parameter cannot be None.")
        return make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file, subsampling_factor=self.subsampling_factor)

    def find_classes(self, directory):
        # Since all directories have same classes, only find classes for first directory
        directory = directory[0]

        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        classes = [c for c in classes if c not in self.dropped_classes]
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        if self.class_merge_dict:
            # Merge classes by mapping from FOLDER --> MERGED_CLASS --> INDEX
            class_to_idx, classes = merged_class_to_idx(classes, self.class_merge_dict)
        else:
            class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        return classes, class_to_idx


    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        path, target = self.samples[index]

        if self.channels:
            sample = tifffile.imread(path)[self.channels]
        else:
            sample = tifffile.imread(path)

        sample = torch.FloatTensor(sample / (2**self.bit_depth - 1))

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)
