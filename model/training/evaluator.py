"""Evaluation loop: runs inference and writes results to disk."""

import os

import numpy as np
import torch


def test(model, test_loader, save_dir, device):
    """Run inference over test_loader and save feat_vecs, labels, preds, outputs."""
    labels = []
    preds = []
    feat_vecs = []
    test_outputs = []

    model.eval()
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(device)
            label = label.to(device)

            test_output, feat_vec = model(data)
            test_pred = test_output.argmax(dim=1)

            test_outputs.append(test_output)
            labels.append(label)
            preds.append(test_pred)
            feat_vecs.append(feat_vec)

    test_outputs = torch.cat(test_outputs, dim=0)
    labels = torch.cat(labels, dim=0)
    preds = torch.cat(preds, dim=0)
    feat_vecs = torch.cat(feat_vecs, dim=0)

    np.savetxt(os.path.join(save_dir, 'feat_vecs.txt'), feat_vecs.cpu().numpy())
    np.savetxt(os.path.join(save_dir, 'labels.txt'), labels.cpu().numpy())
    np.savetxt(os.path.join(save_dir, 'preds.txt'), preds.cpu().numpy())
    np.savetxt(os.path.join(save_dir, 'test_outputs.txt'), test_outputs.cpu().numpy())
