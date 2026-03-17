import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import argparse
import json
import os
import numpy as np
import torchvision
import random
import glob
import datetime

from load_data import load_data
from utils import plot_sample_batch, make_dir, make_model_directories, parse_train_val_test_dir
from models import AvgPoolCNN
from dataset_class_params import get_dropped_moa_classes

from training_and_testing import test


if __name__ == '__main__':
    # Parse input parameters
    parser = argparse.ArgumentParser()
    # General args
    parser.add_argument('--data_dir', required=False)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--test_dir', nargs='+', default=[])
    # Train args
    parser.add_argument('--train_dir', nargs='+', default=[])
    parser.add_argument('--val_dir', nargs='+', default=[])
    parser.add_argument('--dropped_classes', nargs='+', default=[])
    parser.add_argument('--dropped_moa', nargs='+', default=[])
    parser.add_argument('--dose', default=None, type=str)
    parser.add_argument('--num_channels', default=3, type=int)
    parser.add_argument('--channels', nargs='+', default=None, type=int)
    parser.add_argument('--crop_size', default=500, type=int)
    parser.add_argument('--out_size', default=256, type=int)
    parser.add_argument('--val_split', default=0.2, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=250, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--l2', default=0, type=float)
    parser.add_argument('--subsampling_factor', default=1, type=float)
    parser.add_argument('--use_e_coli_moa', action='store_true', default=False)
    parser.add_argument('--freeze_layers', action='store_true', default=False)
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--ckpt_path', default=None)

    args = parser.parse_args()

    #----------General setup---------#
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device =='cuda':
        print('CUDA available')

    # Prepare required directories
    make_model_directories(save_dir=args.save_dir)

    if len(args.dropped_moa) > 0 and args.use_e_coli_moa:
        args.dropped_classes += get_dropped_moa_classes(args.dropped_moa)

    with open(os.path.join(args.save_dir, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    train_val_test_dir = parse_train_val_test_dir(args.data_dir,
                                                  args.train_dir,
                                                  args.test_dir,
                                                  args.val_dir,
                                                  args.dose)

    train_loader, val_loader, test_loader, classes, class_weights = load_data(
        root_dir=args.data_dir,
        train_val_test_dir=train_val_test_dir,
        dropped_classes=args.dropped_classes,
        batch_size=args.batch_size,
        val_split=args.val_split,
        crop_size=args.crop_size,
        out_size=args.out_size,
        channels=args.channels,
        subsampling_factor=args.subsampling_factor,
        use_e_coli_moa=args.use_e_coli_moa)

    dataiter = iter(train_loader)
    images, __ = next(dataiter)

    plot_sample_batch(torchvision.utils.make_grid(images[:4].view(-1,1,args.out_size,args.out_size)), args.save_dir)

    #----------Model configuration----------#
    model = AvgPoolCNN(num_classes=len(classes),
                       num_channels=args.num_channels,
                       pretrained=args.pretrained,
                       n_crops=int((1500 / args.crop_size) ** 2)).to(device)

    if args.ckpt_path:
        print(f'Loading model from checkpoint {args.ckpt_path}...')
        ckpt = torch.load(args.ckpt_path)
        model.load_state_dict(ckpt['model_state_dict'])
        if args.freeze_layers:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc_final.parameters():
                param.requires_grad = True

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    class_weights = torch.Tensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train(model=model,
          criterion=criterion,
          optimizer=optimizer,
          epochs=args.epochs,
          train_loader=train_loader,
          val_loader=val_loader,
          save_dir=args.save_dir)
