"""EfficientNet-B0 backbone with average-pooling tile aggregation."""

import os
from typing import Tuple, Union, List
import numpy as np

import torch
import torch.nn as nn
from torch.hub import download_url_to_file
from torchvision import models

from huggingface_hub import PyTorchModelHubMixin

_WEIGHTS_URL = 'https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth'
_WEIGHTS_FILENAME = 'efficientnet_b0_rwightman-3dd342df.pth'
_WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), 'weights')
_WEIGHTS_PATH = os.path.join(_WEIGHTS_DIR, _WEIGHTS_FILENAME)


class AvgPoolCNN(
    nn.Module,
    PyTorchModelHubMixin,
    repo_url="https://github.com/krentzd/ai4ab"
):
    """EfficientNet-B0 that processes n_crops tiles per sample and averages their features.

    Input: (bs, n_crops, C, H, W) → Output: (logits (bs, num_classes), feat_vec (bs, 1280))
    """

    def __init__(
        self,
        num_classes: int = 2,
        num_channels: int = 3,
        pretrained: bool = False,
        n_crops: int = 9,
        id2label: dict = None
    ) -> None:
        super().__init__()

        self.id2label = id2label

        self.backbone = models.efficientnet_b0()
        if pretrained:
            print('Using pretrained model')
            weights = self._load_pretrained_weights()
            self.backbone.load_state_dict(weights)
        self.backbone.features[0][0] = nn.Conv2d(num_channels, 32, 3, stride=2, padding=1)

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.avg_pool_2 = nn.AvgPool1d(kernel_size=n_crops)
        self.fc_final = nn.Linear(1280, num_classes)

    @staticmethod
    def _load_pretrained_weights() -> dict:
        """Download weights to models/weights/, falling back to the same path if unavailable."""
        os.makedirs(_WEIGHTS_DIR, exist_ok=True)

        if not os.path.exists(_WEIGHTS_PATH):
            try:
                print(f'Downloading weights to {_WEIGHTS_PATH}')
                download_url_to_file(_WEIGHTS_URL, _WEIGHTS_PATH)
            except Exception as e:
                raise RuntimeError(
                    f'Download failed ({e}) and no cached weights found at {_WEIGHTS_PATH}.'
                ) from e

        return torch.load(_WEIGHTS_PATH)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bs, ncrops, c, h, w = x.size()
        x = x.view(-1, c, h, w)
        out = self.backbone.features(x)
        out = self.avg_pool(out).view(bs, ncrops, -1)
        feat_vec = self.avg_pool_2(out.permute((0, 2, 1))).view(bs, -1)
        out = self.fc_final(feat_vec)

        return out, feat_vec

    def predict(self, x: torch.Tensor) -> Union[List[int], List[str]]:
        pred = self.forward(x)[0].argmax(dim=1)
        if self.id2label:
            return [self.id2label[str(p.item())] for p in pred]
        else:
            return [p.item() for p in pred]

    def feat_vecs(self, x: torch.Tensor) -> np.ndarray:
        return self.forward(x)[1].detach().numpy()
