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

from load_data import load_data
from utils import make_model_directories, load_ckpt
from models import AvgPoolCNN
from training_and_testing import test

if __name__ == '__main__':
    # Parse input parameters
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default=None)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--test_dir', default=None)
    parser.add_argument('--ckpt', default=-1, type=int)
    parser.add_argument('--test_dir_ext', default=None)

    args = parser.parse_args()

    #----------General setup---------#
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device =='cuda':
        print('CUDA available')

    with open(os.path.join(args.save_dir, 'commandline_args.txt'), 'r') as f:
        cmd_args = json.load(f)

    if args.test_dir:
        cmd_args['test_dir'] = args.test_dir

    if isinstance(cmd_args['test_dir'], list):
        cmd_args['test_dir'] = cmd_args['test_dir'][0]

    #----------Prepare dataloaders----------#
    __, __, test_loader, classes, __ = load_data(
        root_dir=cmd_args['data_dir'] if args.data_dir is None else args.data_dir,
        train_val_test_dir=[cmd_args['test_dir'], cmd_args['test_dir'], cmd_args['test_dir']],
        dropped_classes=[],
        batch_size=cmd_args['batch_size'],
        val_split=cmd_args['val_split'],
        crop_size=cmd_args['crop_size'],
        out_size=cmd_args['out_size'],
        channels=cmd_args['channels'],
        use_e_coli_moa=cmd_args['use_e_coli_moa'],
        bit_depth=cmd_args['im_bit_depth'])

    #----------Model configuration----------#
    model = AvgPoolCNN(num_classes=len(classes) - len(cmd_args['dropped_classes']),
                       num_channels=len(cmd_args['channels']),
                       pretrained=False,
                       n_crops=int((1500 / cmd_args['crop_size']) ** 2)).to(device)

    ckpt = load_ckpt(args.ckpt, args.save_dir)
    model.load_state_dict(ckpt['model_state_dict'])

    model = model.to(device)

    # Prepare required directories
    save_dir = make_model_directories(save_dir=args.save_dir,
                                      test_dir=cmd_args['test_dir'],
                                      test_dir_ext=args.test_dir_ext)

    test(model=model,
         test_loader=test_loader,
         classes=classes,
         save_dir=save_dir,
         device=device)
