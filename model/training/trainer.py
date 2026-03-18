"""Supervised training loop."""

import os

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.viz import plot_training_curves


def train(model, criterion, optimizer, epochs, train_loader, val_loader, save_dir, device,
          ckpt_freq=1, plot_curves=True):
    """Train model for epochs, saving checkpoints, TensorBoard logs, and optionally loss/accuracy curves."""
    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'tensorboard'))

    train_loss_history = []
    train_accuracy_history = []
    val_loss_history = []
    val_accuracy_history = []

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0

        model.train()
        for data, label in train_loader:
            data = data.to(device)
            label = label.to(device)

            output, __ = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = ((output.argmax(dim=1) == label).float().mean())
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)

        print('Epoch : {}, train accuracy : {}, train loss : {}'.format(
            epoch + 1, epoch_accuracy, epoch_loss
        ))
        train_loss_history.append(epoch_loss.cpu().detach().numpy())
        train_accuracy_history.append(epoch_accuracy.cpu().detach().numpy())

        model.eval()
        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in val_loader:
                data = data.to(device)
                label = label.to(device)

                val_output, __ = model(data)

                val_loss = criterion(val_output, label)
                acc = ((val_output.argmax(dim=1) == label).float().mean())
                epoch_val_accuracy += acc / len(val_loader)
                epoch_val_loss += val_loss / len(val_loader)

            print('Epoch : {}, val_accuracy : {}, val_loss : {}'.format(
                epoch + 1, epoch_val_accuracy, epoch_val_loss
            ))
            val_loss_history.append(epoch_val_loss.cpu().detach().numpy())
            val_accuracy_history.append(epoch_val_accuracy.cpu().detach().numpy())

        writer.add_scalars('Loss', {
            'train': epoch_loss,
            'val': epoch_val_loss,
        }, epoch + 1)
        writer.add_scalars('Accuracy', {
            'train': epoch_accuracy,
            'val': epoch_val_accuracy,
        }, epoch + 1)

        if epoch != 0 and (epoch + 1) % ckpt_freq == 0:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                },
                os.path.join(
                    save_dir,
                    f'ckpts/model_ckpt_{epoch + 1}_{epoch_val_loss.cpu().detach().numpy():.2f}.tar',
                ),
            )

            if plot_curves:
                plot_training_curves(
                    train_loss_history, val_loss_history,
                    train_accuracy_history, val_accuracy_history,
                    epochs, save_dir,
                )

    writer.close()
