"""Testing entry point."""

import torch

from config import TrainingConfig
from data import load_data
from training import test
from utils import make_model_directories


if __name__ == '__main__':
    cfg, ckpt_epoch, test_dir_ext = TrainingConfig.from_test_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print('CUDA available')

    save_dir = make_model_directories(
        save_dir=cfg.save_dir,
        test_dir=cfg.test_dir,
        test_dir_ext=test_dir_ext,
    )

    __, __, test_loader, classes, __ = load_data(cfg)

    model = cfg.build_model(device, ckpt_epoch=ckpt_epoch)

    test(
        model=model,
        test_loader=test_loader,
        save_dir=save_dir,
        device=device,
    )
