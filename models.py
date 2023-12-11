from torch import nn
from torchvision import models
import torch.nn.functional as F
import torch

import torch.nn as nn

# Based on https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py#L8
class LinearSelfAttention(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim):
        super().__init__()

        # Maybe this should be a linear layer?
        self.query = nn.Sequential(nn.Linear(in_dim, in_dim // 2),
                                   nn.ReLU(),
                                   nn.Linear(in_dim // 2, in_dim // 4),
                                   nn.ReLU(),
                                   nn.Linear(in_dim // 4, 1))
        self.key = nn.Sequential(nn.Linear(in_dim, in_dim // 2),
                                   nn.ReLU(),
                                   nn.Linear(in_dim // 2, in_dim // 4),
                                   nn.ReLU(),
                                   nn.Linear(in_dim // 4, 1))
        self.value = nn.Sequential(nn.Linear(in_dim, in_dim),
                                   nn.ReLU(),
                                   nn.Linear(in_dim, in_dim))
        # self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1)

    def forward(self, x):
        bs, ncrops, feature_dim  = x.size()

        proj_query = self.query(x)
        proj_key = self.key(x).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)

        attention = self.softmax(energy)

        proj_value = self.value(x)

        out = torch.bmm(proj_value.permute(0,2,1), attention)
        out = out.view(bs, ncrops, feature_dim)

        # out = self.gamma * out #+ x

        return out, attention

# Based on https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py#L8
class SelfAttention(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim):
        super().__init__()

        self.query_conv = nn.Conv1d(in_channels=in_dim, out_channels=1 , kernel_size= 1)
        self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=1, kernel_size= 1)
        self.value_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size= 1)
        # self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1)

    def forward(self, x):
        bs, ncrops, feature_dim  = x.size()

        proj_query = self.query_conv(x.permute(0,2,1)).permute(0, 2, 1)
        proj_key = self.key_conv(x.permute(0,2,1))
        energy = torch.bmm(proj_query, proj_key)

        attention = self.softmax(energy)

        proj_value = self.value_conv(x.permute(0,2,1))

        out = torch.bmm(proj_value, attention)
        out = out.view(bs, ncrops, feature_dim)

        # out = self.gamma * out #+ x

        return out, attention

class SelfAttentionWeakCNN(nn.Module):
    def __init__(self, num_classes=2, num_channels=4, pretrained=False, dropout=0, **kwargs):
        super().__init__()
        # self.first_layer = nn.Conv2d(num_input_channels, 3, 3, stride=2)

        self.backbone = models.efficientnet_b0()
        if pretrained and num_channels == 3:
            weights = torch.load('../model/efficientnet_b0_rwightman-3dd342df.pth')
            self.backbone.load_state_dict(weights)
        self.backbone.features[0][0] = nn.Conv2d(num_channels, 32, 3, stride=2, padding=1)
        # GAP
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        # Feature aggregation --> Set ncrops automatically!

        self.attention_layer = SelfAttention(in_dim=1280)
        # self.attention_layer = LinearSelfAttention(in_dim=1280)

        self.fc_final = nn.Linear(1280, num_classes) # For multi-class problem must increase num of neurons
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Take input x composed of n tiles
        bs, ncrops, c, h, w = x.size()

        # Reshape data to predict on all crops in one pass
        x = x.view(-1, c, h, w)
        # x = self.first_layer(x)
        # Pass n tiles through network
        out = self.backbone.features(x)
        # Reshape to bs, ncrops, feat_vec_size
        out = self.avg_pool(out).view(bs, ncrops, -1)
        # Aggregate embeddings across all crops
        # feat_vec = self.conv1d(out).view(bs, 1280)
        out, attention_map = self.attention_layer(out)

        feat_vec = out.mean(dim=1)

        out = self.fc_final(feat_vec)
        out = self.dropout(out)
        return out, feat_vec, attention_map


class WeakCNN(nn.Module):
    def __init__(self, num_classes=2, num_channels=4, pretrained=False, dropout=0):
        super().__init__()

        self.backbone = models.efficientnet_b0()
        if pretrained and num_channels == 3:
            print('Using pretrained model')
            weights = torch.load('../model/efficientnet_b0_rwightman-3dd342df.pth')
            self.backbone.load_state_dict(weights)
        self.backbone.features[0][0] = nn.Conv2d(num_channels, 32, 3, stride=2, padding=1)
        # GAP
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

        self.fc_final = nn.Linear(1280, num_classes) # For multi-class problem must increase num of neurons
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Take input x composed of n tiles
        bs, ncrops, c, h, w = x.size()
        # Reshape data to predict on all crops in one pass
        x = x.view(-1, c, h, w)
        # x = self.first_layer(x)
        # Pass n tiles through network
        out = self.backbone.features(x)
        # Reshape to bs, ncrops, feat_vec_size
        out = self.avg_pool(out).view(bs, ncrops, -1)
        # Aggregate embeddings across all crops
        # feat_vec = self.conv1d(out).view(bs, 1280)

        feat_vec = out.mean(dim=1)

        out = self.fc_final(feat_vec)
        out = self.dropout(out)
        return out, feat_vec
