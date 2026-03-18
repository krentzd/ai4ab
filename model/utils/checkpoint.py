"""Checkpoint loading."""

import glob
import os

import torch

from config import CKPT_GLOB, CKPT_LOSS_FIELD


def load_ckpt(ckpt: int, save_dir: str) -> dict:
    """Load a checkpoint. Pass ckpt=-1 to auto-select the lowest-loss checkpoint."""
    if ckpt > 0:
        ckpt_path = glob.glob(
            os.path.join(save_dir, 'ckpts', '*_' + str(ckpt) + '_*.tar')
        )[0]
    elif ckpt == -1:
        ckpt_paths = glob.glob(os.path.join(save_dir, 'ckpts', CKPT_GLOB))
        ckpt_path = sorted(
            ckpt_paths,
            key=lambda s: os.path.basename(s).split('_')[CKPT_LOSS_FIELD],
        )[0]

    return torch.load(ckpt_path)
