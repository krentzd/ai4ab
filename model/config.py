"""Central configuration for the ai4ab pipeline."""

import argparse
import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

import torch

NUM_WORKERS: int = 12
CKPT_GLOB: str = "*.tar"
CKPT_LOSS_FIELD: int = 3


@dataclass
class TrainingConfig:
    """All hyperparameters and paths for a training or testing run."""

    data_dir: Optional[str] = None
    save_dir: str = ""
    train_dir: List[str] = field(default_factory=list)
    test_dir: List[str] = field(default_factory=list)
    val_dir: List[str] = field(default_factory=list)
    train_val_test_dir: List = field(default_factory=list)
    dropped_classes: List[str] = field(default_factory=list)
    dropped_moa: List[str] = field(default_factory=list)
    dose: Optional[str] = None
    channels: Optional[List[int]] = None
    full_image_size: int = 1500
    crop_size: int = 500
    out_size: int = 256
    val_split: float = 0.2
    batch_size: int = 16
    epochs: int = 250
    lr: float = 0.001
    l2: float = 0.001
    subsampling_factor: float = 1.0
    use_e_coli_moa: bool = False
    use_class_weights: bool = False
    freeze_layers: bool = False
    pretrained: bool = False
    ckpt_path: Optional[str] = None
    im_bit_depth: int = 8
    ckpt_freq: int = 1
    plot_curves: bool = True
    num_classes: Optional[int] = None
    num_channels: Optional[int] = None
    n_crops_val: Optional[int] = None
    aug_wo_rand_rot: bool = False

    def finalise_model_params(self, num_classes: int) -> None:
        """Derive and store num_classes, num_channels, and n_crops_val from resolved data."""
        self.num_classes = num_classes
        self.num_channels = len(self.channels)
        self.n_crops_val = int((self.full_image_size / self.crop_size) ** 2)

    def resolve_dropped_classes(self) -> None:
        """Expand dropped_moa group names into dropped_classes entries."""
        from data.class_params import get_dropped_moa_classes
        if len(self.dropped_moa) > 0:
            self.dropped_classes += get_dropped_moa_classes(self.dropped_moa)

    def build_model(self, device: str, ckpt_epoch: int = None):
        """Instantiate AvgPoolCNN, optionally load a checkpoint, and move to device.

        Training: loads cfg.ckpt_path if set, then applies layer freezing.
        Testing:  loads the checkpoint selected by ckpt_epoch via load_ckpt.
        """
        from models import AvgPoolCNN
        from utils import load_ckpt

        model = AvgPoolCNN(
            num_classes=self.num_classes,
            num_channels=self.num_channels,
            pretrained=self.pretrained,
            n_crops=self.n_crops_val,
        )

        if ckpt_epoch is not None:
            ckpt = load_ckpt(ckpt_epoch, self.save_dir)
            model.load_state_dict(ckpt['model_state_dict'])
        elif self.ckpt_path:
            print(f'Loading model from checkpoint {self.ckpt_path}...')
            ckpt = torch.load(self.ckpt_path)
            model.load_state_dict(ckpt['model_state_dict'])
            if self.freeze_layers:
                for param in model.parameters():
                    param.requires_grad = False
                for param in model.fc_final.parameters():
                    param.requires_grad = True

        return model.to(device)

    @classmethod
    def from_args(cls) -> 'TrainingConfig':
        """Parse training CLI arguments, resolve train_val_test_dir, and return a TrainingConfig."""
        from utils.utils import parse_train_val_test_dir

        parser = argparse.ArgumentParser()
        parser.add_argument('--data_dir', required=False)
        parser.add_argument('--save_dir', required=True)
        parser.add_argument('--train_dir', nargs='+', default=[])
        parser.add_argument('--test_dir', nargs='+', default=[])
        parser.add_argument('--val_dir', nargs='+', default=[])
        parser.add_argument('--dropped_classes', nargs='+', default=[])
        parser.add_argument('--dropped_moa', nargs='+', default=[])
        parser.add_argument('--dose', default=None, type=str)
        parser.add_argument('--channels', nargs='+', default=None, type=int)
        parser.add_argument('--full_image_size', default=1500, type=int)
        parser.add_argument('--crop_size', default=500, type=int)
        parser.add_argument('--out_size', default=256, type=int)
        parser.add_argument('--val_split', default=0.2, type=float)
        parser.add_argument('--batch_size', default=16, type=int)
        parser.add_argument('--epochs', default=250, type=int)
        parser.add_argument('--lr', default=0.001, type=float)
        parser.add_argument('--l2', default=0.001, type=float)
        parser.add_argument('--subsampling_factor', default=1, type=float)
        parser.add_argument('--use_e_coli_moa', action='store_true', default=False)
        parser.add_argument('--use_class_weights', action='store_true', default=False)
        parser.add_argument('--freeze_layers', action='store_true', default=False)
        parser.add_argument('--pretrained', action='store_true', default=False)
        parser.add_argument('--ckpt_path', default=None)
        parser.add_argument('--im_bit_depth', default=8, type=int)
        parser.add_argument('--ckpt_freq', default=1, type=int)
        parser.add_argument('--plot_curves', action='store_true', default=True)
        parser.add_argument('--aug_wo_rand_rot', action='store_true', default=False)
        args = parser.parse_args()

        cfg = cls(**vars(args))
        cfg.resolve_dropped_classes()
        cfg.train_val_test_dir = parse_train_val_test_dir(
            cfg.data_dir, cfg.train_dir, cfg.test_dir, cfg.val_dir, cfg.dose,
        )
        return cfg

    @classmethod
    def from_test_args(cls) -> tuple:
        """Restore a saved config, apply CLI overrides, return (cfg, ckpt, test_dir_ext)."""
        from utils.utils import parse_train_val_test_dir

        parser = argparse.ArgumentParser()
        parser.add_argument('--save_dir', required=True)
        parser.add_argument('--ckpt', default=-1, type=int)
        parser.add_argument('--test_dir_ext', default=None)
        parser.add_argument('--data_dir', default=None)
        parser.add_argument('--test_dir', default=None)
        parser.add_argument('--dropped_classes', nargs='+', default=None)
        args = parser.parse_args()

        config_path = os.path.join(args.save_dir, 'commandline_args.txt')
        with open(config_path, 'r') as f:
            cfg = cls(**json.load(f))

        cfg.dropped_classes = args.dropped_classes if args.dropped_classes is not None else []
        if args.data_dir is not None:
            cfg.data_dir = args.data_dir
        if args.test_dir is not None:
            cfg.test_dir = args.test_dir

        if isinstance(cfg.test_dir, list):
            cfg.test_dir = cfg.test_dir[0]

        cfg.train_val_test_dir = parse_train_val_test_dir(
            cfg.data_dir, cfg.train_dir, cfg.test_dir, cfg.val_dir, cfg.dose,
        )
        return cfg, args.ckpt, args.test_dir_ext

    def save(self) -> None:
        """Serialise this config to <save_dir>/commandline_args.txt as JSON."""
        with open(os.path.join(self.save_dir, 'commandline_args.txt'), 'w') as f:
            json.dump(dataclasses.asdict(self), f, indent=2)
