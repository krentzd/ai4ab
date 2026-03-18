"""Visualisation utilities."""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision


def plot_sample_batch(loader, out_size: int, save_dir: str, save_name: str = 'sample_images.png') -> None:
    """Draw one batch from loader, build a grid, denormalise, and save as a PNG."""
    images, __ = next(iter(loader))
    grid = torchvision.utils.make_grid(images[:4].view(-1, 1, out_size, out_size))
    grid = grid * 0.5 + 0.5
    npimg = grid.numpy()
    fig = plt.figure(figsize=(30, 30))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(os.path.join(save_dir, save_name))
    plt.close()


def plot_training_curves(
    train_loss_history: list,
    val_loss_history: list,
    train_accuracy_history: list,
    val_accuracy_history: list,
    epochs: int,
    save_dir: str,
) -> None:
    """Save loss and accuracy curves to <save_dir>/loss_curves.png and accuracy_curves.png."""
    plt.figure('Loss')
    plt.plot(train_loss_history, label='train')
    plt.plot(val_loss_history, label='validation')
    plt.xlim([0, epochs])
    plt.legend()
    plt.title('Loss')
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'))
    plt.close()

    plt.figure('Accuracy')
    plt.plot(train_accuracy_history, label='train')
    plt.plot(val_accuracy_history, label='validation')
    plt.xlim([0, epochs])
    plt.legend()
    plt.title('Accuracy')
    plt.savefig(os.path.join(save_dir, 'accuracy_curves.png'))
    plt.close()
