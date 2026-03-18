"""Split a (C, H, W) tensor into a list of (1, H, W) single-channel tensors."""

import torch


def convert_to_list(x: torch.Tensor) -> list:
    """Return x split along dim 0, each slice unsqueezed to (1, H, W)."""
    return [x_i.unsqueeze(0) for x_i in x]
