from torch import nn
from torchvision import models
import torch.nn.functional as F
import torch

import torch.nn as nn

class AvgPoolCNN(nn.Module):
    def __init__(self, num_classes=2, num_channels=3, pretrained=False, dropout=0, n_crops=9):
        super().__init__()

        self.backbone = models.efficientnet_b0()
        if pretrained:
            print('Using pretrained model')
            weights = torch.load('../model/efficientnet_b0_rwightman-3dd342df.pth')
            self.backbone.load_state_dict(weights)
        self.backbone.features[0][0] = nn.Conv2d(num_channels, 32, 3, stride=2, padding=1)
        # GAP
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

        self.avg_pool_2 = nn.AvgPool1d(kernel_size=n_crops)

        self.fc_final = nn.Linear(1280, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Take input x composed of n tiles
        bs, ncrops, c, h, w = x.size()
        # Reshape data to predict on all crops in one pass
        x = x.view(-1, c, h, w)
        # Pass n tiles through network
        out = self.backbone.features(x)
        out = self.avg_pool(out).view(bs, ncrops, -1)

        feat_vec = self.avg_pool_2(out.permute((0, 2, 1))).view(bs, -1)

        out = self.fc_final(feat_vec)
        out = self.dropout(out)

        return out, feat_vec
