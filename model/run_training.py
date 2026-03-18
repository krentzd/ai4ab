"""Training entry point."""

import torch
from torch import nn, optim

from config import TrainingConfig
from data import load_data
from training import train
from utils import make_model_directories, plot_sample_batch


if __name__ == '__main__':
    cfg = TrainingConfig.from_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print('CUDA available')

    make_model_directories(save_dir=cfg.save_dir)

    train_loader, val_loader, test_loader, classes, class_weights = load_data(cfg)

    cfg.finalise_model_params(len(classes))
    cfg.save()

    plot_sample_batch(train_loader, cfg.out_size, cfg.save_dir)

    model = cfg.build_model(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.l2)
    class_weights = torch.Tensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        epochs=cfg.epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        save_dir=cfg.save_dir,
        device=device,
        ckpt_freq=cfg.ckpt_freq,
        plot_curves=cfg.plot_curves,
    )
