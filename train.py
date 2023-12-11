import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import argparse
import json
import os
import numpy as np
import torchvision

from load_data import load_data
from utils import plot_sample_batch
from models import WeakCNN, SelfAttentionWeakCNN

def make_dir(dir):
    """Create directories including subdirectories"""
    dir_lst = dir.split('/')
    for idx in range(1, len(dir_lst) + 1):
        if not os.path.exists(os.path.join(*dir_lst[:idx])):
            os.mkdir(os.path.join(*dir_lst[:idx]))

def save_attention_map_overlay(attention_matrix, data, epoch, save_dir):
    # Dims: batch_size, n_crops, height, width
    make_dir(os.path.join(save_dir, 'attention_maps'))

    # __, __, im_dim = attention_matrix.size()

    ones = torch.ones(1, 9)

    attention_map = torch.matmul(ones, attention_matrix[0].cpu()).squeeze().numpy()

    # a_row_list = []
    # for h in range(im_dim):
    #     row_elements = [attention_map[i + h * im_dim] for i in range(im_dim)]
    #     a_row_list.append(np.hstack(row_elements))
    #
    # attention_image = np.vstack(a_row_list)

    a_row_0 = np.hstack((attention_map[0], attention_map[1], attention_map[2]))
    a_row_1 = np.hstack((attention_map[3], attention_map[4], attention_map[5]))
    a_row_2 = np.hstack((attention_map[6], attention_map[7], attention_map[8]))

    attention_image = np.vstack((a_row_0, a_row_1, a_row_2))

    # attention_map = attention_map[0].cpu().numpy()

    plt.figure('Attention map')
    plt.imshow(attention_image)
    plt.savefig(os.path.join(save_dir, 'attention_maps', f'attention_map_epoch_{epoch}.png'))
    plt.colorbar()
    plt.close()

    image = data[0].cpu().numpy()

    # row_list = []
    # for h in range(im_dim):
    #     row_elements = [image[i + h * im_dim][3] for i in range(im_dim)]
    #     row_list.append(np.hstack(row_elements))
    #
    # concat_image = np.vstack(row_list)

    row_0 = np.hstack((image[0][-1], image[1][-1], image[2][-1]))
    row_1 = np.hstack((image[3][-1], image[4][-1], image[5][-1]))
    row_2 = np.hstack((image[6][-1], image[7][-1], image[8][-1]))

    concat_image = np.vstack((row_0, row_1, row_2))

    plt.figure('Figure')
    plt.imshow(concat_image, cmap='gray')
    plt.savefig(os.path.join(save_dir, 'attention_maps', f'image_epoch_{epoch}.png'))
    plt.close()

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

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0

        model.train()
        for data, label in train_loader:
            data = data.to(device)
            label = label.to(device)

            if kwargs.get('self_attention', False):
                output, __, __ = model(data)

            else:
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

                if kwargs.get('self_attention', False):
                    val_output, __, attention_map = model(data)
                else:
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

        if kwargs.get('self_attention', False):
            save_attention_map_overlay(attention_map, data, epoch, save_dir)

        # Save updated training curves after each epoch
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


if __name__ == '__main__':
    # Parse input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=False)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--dropped_classes', nargs='+', default=[])
    parser.add_argument('--train_dir', nargs='+', default=[])
    parser.add_argument('--test_dir', nargs='+', default=[])
    parser.add_argument('--dose', default=3, type=int)
    parser.add_argument('--num_channels', default=4, type=int)
    parser.add_argument('--channels', nargs='+', default=None)
    parser.add_argument('--crop_size', default=512, type=int)
    parser.add_argument('--val_split', default=0.2, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--l2', default=0, type=float)
    parser.add_argument('--ckpt_path', default=None)
    parser.add_argument('--self_attention', action='store_true', default=False)
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

    if len(args.train_dir) == 0:
        # all_dir = [f'R1/dose_{args.dose}', f'R2/dose_{args.dose}', f'R3/dose_{args.dose}', f'R4/dose_{args.dose}']#, f'R5/dose_{args.dose}', f'R6/dose_{args.dose}' , f'R6_1_to_10/dose_{args.dose}', f'R6A_1_to_10/dose_{args.dose}']
        # all_dir = [f'R1', f'R2', f'R3', f'R4']#, f'R5/dose_{args.dose}', f'R6/dose_{args.dose}' , f'R6_1_to_10/dose_{args.dose}', f'R6A_1_to_10/dose_{args.dose}']
        all_dir = [f'R6/dose_{args.dose}', f'R6A/dose_{args.dose}', f'R6B/dose_{args.dose}']

        args.train_dir = [dir for dir in all_dir if dir not in args.test_dir]

    with open(os.path.join(args.save_dir, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    train_loader, val_loader, __, classes = load_data(root_dir=args.data_dir,
                                                                train_test_dir=[args.train_dir, args.test_dir],
                                                                batch_size=args.batch_size,
                                                                val_split=args.val_split,
                                                                crop_size=args.crop_size,
                                                                channels=args.channels,
                                                                data_type=args.data_type)

    print('Classes', classes)
    # Plot a sample batch and save
    dataiter = iter(train_loader)
    images, __ = next(dataiter)

    plot_sample_batch(torchvision.utils.make_grid(images[:4].view(-1,1,256,256)), args.save_dir)

    # Instantiate CNN, optimizer and loss function
    num_tiles = int((1500 / args.crop_size) ** 2)

    if not args.self_attention:
        model = WeakCNN(num_classes=len(classes),
                        num_channels=args.num_channels,
                        pretrained=args.pretrained).to(device)

    elif args.self_attention:
        model = SelfAttentionWeakCNN(num_classes=len(classes),
                        num_channels=args.num_channels,
                        pretrained=False).to(device)

    print(model)
    print('Trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    if args.ckpt_path:
        ckpt = torch.load(args.ckpt_path)
        model.load_state_dict(ckpt['model_state_dict'])
        if args.freeze_layers:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc_final.parameters():
                param.requires_grad = True

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    criterion = nn.CrossEntropyLoss()

    train(model=model,
          criterion=criterion,
          optimizer=optimizer,
          epochs=args.epochs,
          train_loader=train_loader,
          val_loader=val_loader,
          save_dir=args.save_dir,
          self_attention=args.self_attention)
