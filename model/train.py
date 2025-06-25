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
from utils import plot_sample_batch
from models import AvgPoolCNN

def make_dir(dir):
    """Create directories including subdirectories"""
    dir_lst = dir.split('/')
    for idx in range(1, len(dir_lst) + 1):
        if not os.path.exists(os.path.join(*dir_lst[:idx])):
            os.mkdir(os.path.join(*dir_lst[:idx]))

def train(model,
          criterion,
          optimizer,
          epochs,
          train_loader,
          val_loader,
          save_dir,
          **kwargs):
    """
    Training loop
    """
    train_loss_history = []
    train_accuracy_history = []
    val_loss_history = []
    val_accuracy_history = []

    start_epoch = kwargs.get('start_epoch', 0)
    for epoch in range(start_epoch, epochs):
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
            epoch_accuracy += acc/len(train_loader)
            epoch_loss += loss/len(train_loader)

        print('Epoch : {}, train accuracy : {}, train loss : {}'.format(epoch+1, epoch_accuracy,epoch_loss))
        train_loss_history.append(epoch_loss.cpu().detach().numpy())
        train_accuracy_history.append(epoch_accuracy.cpu().detach().numpy())

        model.eval()
        with torch.no_grad():
            epoch_val_accuracy= 0
            epoch_val_loss = 0
            for data, label in val_loader:
                data = data.to(device)
                label = label.to(device)

                val_output, __ = model(data)

                val_loss = criterion(val_output,label)
                acc = ((val_output.argmax(dim=1) == label).float().mean())
                epoch_val_accuracy += acc/ len(val_loader)
                epoch_val_loss += val_loss/ len(val_loader)

            print('Epoch : {}, val_accuracy : {}, val_loss : {}'.format(epoch+1, epoch_val_accuracy,epoch_val_loss))
            val_loss_history.append(epoch_val_loss.cpu().detach().numpy())
            val_accuracy_history.append(epoch_val_accuracy.cpu().detach().numpy())

        if (epoch + 1) % 10 == 0 and epoch != 0:
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss
            }, os.path.join(save_dir, f'ckpts/model_ckpt_{epoch + 1}_{epoch_val_loss.cpu().detach().numpy():.2f}.tar'))

        # Save updated training curves after each epoch
        plt.figure('Loss')
        plt.plot(train_loss_history, label='train')
        plt.plot(val_loss_history, label='validation')
        plt.xlim([0, epochs])
        plt.legend()
        plt.title('Loss')
        plt.savefig(os.path.join(save_dir, f'loss_curves_from_{start_epoch}.png'))
        plt.close()

        plt.figure('Accuracy')
        plt.plot(train_accuracy_history, label='train')
        plt.plot(val_accuracy_history, label='validation')
        plt.xlim([0, epochs])
        plt.legend()
        plt.title('Accuracy')
        plt.savefig(os.path.join(save_dir, f'accuracy_curves_from_{start_epoch}.png'))
        plt.close()


if __name__ == '__main__':
    # Parse input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=False)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--dropped_classes', nargs='+', default=[])
    parser.add_argument('--dropped_moa', nargs='+', default=[])
    parser.add_argument('--train_dir', nargs='+', default=[])
    parser.add_argument('--val_dir', nargs='+', default=[])
    parser.add_argument('--test_dir', nargs='+', default=[])
    parser.add_argument('--dose', default=None, type=str)
    parser.add_argument('--num_channels', default=3, type=int)
    parser.add_argument('--channels', nargs='+', default=None, type=int)
    parser.add_argument('--crop_size', default=512, type=int)
    parser.add_argument('--val_split', default=0.2, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--bottleneck_size', default=None)
    parser.add_argument('--n_crops', default=9, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--l2', default=0, type=float)
    parser.add_argument('--subsampling_factor', default=1, type=float)
    parser.add_argument('--ckpt_path', default=None)
    parser.add_argument('--use_e_coli_moa', action='store_true', default=False)
    parser.add_argument('--max_pool_model', action='store_true', default=False)
    parser.add_argument('--freeze_layers', action='store_true', default=False)
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--data_type', default='tiff')

    args = parser.parse_args()

    torch.manual_seed(111)

    if not os.path.exists(args.save_dir):
        make_dir(args.save_dir)

    if not os.path.exists(os.path.join(args.save_dir, 'ckpts')):
        os.mkdir(os.path.join(args.save_dir, 'ckpts'))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device =='cuda':
        print('CUDA available')

    if args.channels:
        args.num_channels = len(args.channels)


    all_dir = os.listdir(args.data_dir)

    if args.dose:
        all_dir = [os.path.join(d, args.dose) for d in all_dir]

    if args.val_dir[0] == 'None':
        args.val_dir = None

    elif len(args.val_dir) == 0:
        args.val_dir = [random.choice([dir for dir in all_dir if dir not in args.test_dir])]

    if len(args.train_dir) == 0:
        if args.val_dir:
            args.train_dir = [dir for dir in all_dir if dir not in args.test_dir + args.val_dir]
        else:
            args.train_dir = [dir for dir in all_dir if dir not in args.test_dir]

    print('Train: ', args.train_dir)
    print('Val: ', args.val_dir)
    print('Test: ', args.test_dir)


    if args.ckpt_path:
        if eval(args.ckpt_path) == -1:
            try:
                ckpt_paths = glob.glob(os.path.join(args.save_dir, 'ckpts', '*.tar'))
                args.ckpt_path = sorted(ckpt_paths, key=lambda s: os.path.basename(s).split('_')[2])[-1]
            except IndexError:
                args.ckpt_path = None
        else:
            start_epoch = eval(os.path.basename(args.ckpt_path).split('_')[2])
    else:
        start_epoch = 0

    if len(args.dropped_moa) > 0 and args.use_e_coli_moa:
        doses = ['0.125xIC50', '0.25xIC50', '0.5xIC50', '1xIC50']
        for moa in args.dropped_moa:
            if moa == 'PBP1':
                class_names = ['Cefsulodin', 'PenicillinG', 'Sulbactam']
            elif moa == 'PBP2':
                class_names = ['Avibactam', 'Mecillinam', 'Meropenem', 'Relebactam', 'Clavulanate']
            elif moa == 'PBP3':
                class_names = ['Aztreonam', 'Ceftriaxone', 'Cefepime']
            elif moa == 'Gyrase':
                class_names = ['Ciprofloxacin', 'Levofloxacin', 'Norfloxacin']
            elif moa == 'Ribosome':
                class_names = ['Doxycycline', 'Kanamycin', 'Chloramphenicol', 'Clarithromycin']
            elif moa == 'Membrane':
                class_names = ['Colistin', 'PolymyxinB']

            class_names_ = [c for c in ['Ciprofloxacin', 'Cefsulodin', 'Relebactam', 'Ceftriaxone', 'Doxycycline'] if c not in class_names]

            class_names += class_names_

            classes_ = [f'{x}_{d}' for d in doses for x in class_names]
            args.dropped_classes += classes_

    with open(os.path.join(args.save_dir, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    train_loader, val_loader, __, classes, class_weights = load_data(root_dir=args.data_dir,
                                                      	             train_val_test_dir=[args.train_dir, args.val_dir, args.test_dir],
                				                                  	 dropped_classes=args.dropped_classes,
                				                                  	 batch_size=args.batch_size,
                				                                  	 val_split=args.val_split,
                				                                  	 crop_size=args.crop_size,
                				                                  	 channels=args.channels,
                				                                  	 data_type=args.data_type,
                				                                  	 subsampling_factor=args.subsampling_factor,
                                                                     use_e_coli_moa=args.use_e_coli_moa)

    print('Classes', classes)
    # Plot a sample batch and save
    dataiter = iter(train_loader)
    images, __ = next(dataiter)

    plot_sample_batch(torchvision.utils.make_grid(images[:4].view(-1,1,256,256)), args.save_dir)

    # Instantiate CNN, optimizer and loss function
    num_tiles = int((1500 / args.crop_size) ** 2)

    model = AvgPoolCNN(num_classes=len(classes),
                       num_channels=args.num_channels,
                       pretrained=args.pretrained).to(device)


    print(model)
    print('Trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

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
    print('Class weights', classes, class_weights)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train(model=model,
          criterion=criterion,
          optimizer=optimizer,
          epochs=args.epochs,
          train_loader=train_loader,
          val_loader=val_loader,
          save_dir=args.save_dir)
