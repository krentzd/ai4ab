import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
import os
import glob
import scikitplot as skplt
import json
import numpy as np
import torchvision
from torch import nn
from tqdm import tqdm

from load_data import load_data
from utils import plot_predictions, plot_representations, get_tsne, get_umap, get_pca, plot_image_representations
from models import AvgPoolCNN

def make_dir(dir):
    """Create directories including subdirectories"""
    dir_lst = dir.split('/')
    for idx in range(1, len(dir_lst) + 1):
        if not os.path.exists(os.path.join(*dir_lst[:idx])):
            os.mkdir(os.path.join(*dir_lst[:idx]))

def make_prediction_matrix(labels, preds, classes_true, classes_pred, save_name, mode='normalize'):
    from matplotlib.ticker import MultipleLocator

    def return_counts_array(labels, preds):
        counts_array = np.zeros((int(max(labels)) + 1, int(max(preds)) + 1))
        for l in np.unique(labels):
            x = np.array(preds)[np.array(labels) == l]
            p = np.unique(x, return_counts=True)
            if p[0].size > 0:
                for p_idx, p_val in zip(p[0], p[1]):
                    counts_array[l, p_idx] = p_val
        return counts_array

    counts_array = return_counts_array(labels, preds)
    if mode == 'normalize':
        counts_array_temp = np.zeros_like(counts_array)
        for i, row in enumerate(counts_array):
            counts_array_temp[i] = row / row.sum()
        counts_array = counts_array_temp
    elif mode == 'max':
        counts_array_temp = np.zeros_like(counts_array)
        for i, row in enumerate(counts_array):
            counts_array_temp[i, np.argmax(row)] = 1
        counts_array = counts_array_temp

    fig, ax = plt.subplots(figsize=(20,20))
    ax.matshow(counts_array, cmap=plt.cm.Blues)
    plt.gca().xaxis.tick_bottom()

    ax.set_xticklabels([''] + classes_true, rotation=90)
    ax.set_yticklabels([''] + classes_pred)

    for i in range(counts_array.shape[1]):
        for j in range(counts_array.shape[0]):
            c = counts_array[j,i]
            if mode == 'normalize':
                ax.text(i, j, str(c), va='center', ha='center', c='black' if c < 0.5 else 'white')
            else:
                ax.text(i, j, str(c), va='center', ha='center', c='black' if c < 0.5 * counts_array.max() else 'white')
    if mode == 'normalize':
        plt.title('Normalized Prediction Counts')
    if mode == 'max':
        plt.title('Max Prediction Counts')
    else:
        plt.title('Prediction Counts')

    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_major_locator(MultipleLocator(1))

    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    print('Saving to', save_name)
    plt.savefig(save_name)

def test_on_weak(model, test_loader, classes, save_dir, **kwargs):
    """
    Compute accuracy on test dataset, plot AUC, show predicted images
    """
    images = []
    labels = []
    preds = []
    class_probs = []
    test_accuracy = 0
    feat_vecs = []
    test_outputs = []

    fnames = []

    model.eval()
    with torch.no_grad():
        for i, (data, label) in enumerate(tqdm(test_loader)):
            data = data.to(device)
            label = label.to(device)

            test_output, feat_vec = model(data)

            test_pred = (test_output.argmax(dim=1))

            acc = (test_pred == label).float().mean()

            test_accuracy += acc/len(test_loader)

            images.append(data[:,4]) #Save center crop
            test_outputs.append(test_output)
            labels.append(label)
            preds.append(test_pred)
            class_probs.append(F.softmax(test_output, dim=1))
            feat_vecs.append(feat_vec)


    images = torch.cat(images, dim=0)
    test_outputs = torch.cat(test_outputs, dim=0)
    labels = torch.cat(labels, dim=0)
    preds = torch.cat(preds, dim=0)
    class_probs = torch.cat(class_probs, dim=0)
    feat_vecs = torch.cat(feat_vecs, dim=0)

    np.savetxt(os.path.join(save_dir, 'feat_vecs.txt'), feat_vecs.cpu().numpy())
    np.savetxt(os.path.join(save_dir, 'labels.txt'), labels.cpu().numpy())
    np.savetxt(os.path.join(save_dir, 'preds.txt'), preds.cpu().numpy())
    np.savetxt(os.path.join(save_dir, 'test_outputs.txt'), test_outputs.cpu().numpy())

if __name__ == '__main__':
    # Parse input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=None)
    parser.add_argument('--test_dir', default=None)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--use_imagenet', action='store_true', default=False)
    parser.add_argument('--ckpt', default=1000, type=int)
    parser.add_argument('--dropped_label', default=0, type=int)
    parser.add_argument('--test_mode', default='normal')
    parser.add_argument('--model', default='5layer')
    parser.add_argument('--dropped_classes', nargs='+', default=[])
    parser.add_argument('--save_dir_addition', default='')

    args = parser.parse_args()

    with open(os.path.join(args.save_dir, 'commandline_args.txt'), 'r') as f:
        cmd_args = json.load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device =='cuda':
        print('CUDA available')

    torch.manual_seed(111)
    print(cmd_args)
    print(args.data_dir)

    if args.test_dir:
        cmd_args['test_dir'] = args.test_dir

    if len(args.dropped_classes) > 0:
        cmd_args['dropped_classes'] = args.dropped_classes

    if isinstance(cmd_args['test_dir'], list):
        cmd_args['test_dir'] = cmd_args['test_dir'][0]

    print('Test: ', cmd_args['test_dir'])

    if cmd_args.get('use_e_coli_moa', False):
        e_coli_classes = ['Avibactam_0.125xIC50', 'Avibactam_0.25xIC50', 'Avibactam_0.5xIC50', 'Avibactam_1xIC50', 'Aztreonam_0.125xIC50', 'Aztreonam_0.25xIC50', 'Aztreonam_0.5xIC50', 'Aztreonam_1xIC50', 'Cefepime_0.125xIC50', 'Cefepime_0.25xIC50', 'Cefepime_0.5xIC50', 'Cefepime_1xIC50', 'Cefsulodin_0.125xIC50', 'Cefsulodin_0.25xIC50', 'Cefsulodin_0.5xIC50', 'Cefsulodin_1xIC50', 'Ceftriaxone_0.125xIC50', 'Ceftriaxone_0.25xIC50', 'Ceftriaxone_0.5xIC50', 'Ceftriaxone_1xIC50', 'Chloramphenicol_0.125xIC50', 'Chloramphenicol_0.25xIC50', 'Chloramphenicol_0.5xIC50', 'Chloramphenicol_1xIC50', 'Ciprofloxacin_0.125xIC50', 'Ciprofloxacin_0.25xIC50', 'Ciprofloxacin_0.5xIC50', 'Ciprofloxacin_1xIC50', 'Clarithromycin_0.125xIC50', 'Clarithromycin_0.25xIC50', 'Clarithromycin_0.5xIC50', 'Clarithromycin_1xIC50', 'Clavulanate_0.125xIC50', 'Clavulanate_0.25xIC50', 'Clavulanate_0.5xIC50', 'Clavulanate_1xIC50', 'Colistin_0.125xIC50', 'Colistin_0.25xIC50', 'Colistin_0.5xIC50', 'Colistin_1xIC50', 'DMSO', 'Doxycycline_0.125xIC50', 'Doxycycline_0.25xIC50', 'Doxycycline_0.5xIC50', 'Doxycycline_1xIC50', 'Kanamycin_0.125xIC50', 'Kanamycin_0.25xIC50', 'Kanamycin_0.5xIC50', 'Kanamycin_1xIC50', 'Levofloxacin_0.125xIC50', 'Levofloxacin_0.25xIC50', 'Levofloxacin_0.5xIC50', 'Levofloxacin_1xIC50', 'Mecillinam_0.125xIC50', 'Mecillinam_0.25xIC50', 'Mecillinam_0.5xIC50', 'Mecillinam_1xIC50', 'Meropenem_0.125xIC50', 'Meropenem_0.25xIC50', 'Meropenem_0.5xIC50', 'Meropenem_1xIC50', 'Norfloxacin_0.125xIC50', 'Norfloxacin_0.25xIC50', 'Norfloxacin_0.5xIC50', 'Norfloxacin_1xIC50', 'PenicillinG_0.125xIC50', 'PenicillinG_0.25xIC50', 'PenicillinG_0.5xIC50', 'PenicillinG_1xIC50', 'PolymyxinB_0.125xIC50', 'PolymyxinB_0.25xIC50', 'PolymyxinB_0.5xIC50', 'PolymyxinB_1xIC50', 'Relebactam_0.125xIC50', 'Relebactam_0.25xIC50', 'Relebactam_0.5xIC50', 'Relebactam_1xIC50', 'Rifampicin_0.125xIC50', 'Rifampicin_0.25xIC50', 'Rifampicin_0.5xIC50', 'Rifampicin_1xIC50', 'Sulbactam_0.125xIC50', 'Sulbactam_0.25xIC50', 'Sulbactam_0.5xIC50', 'Sulbactam_1xIC50', 'Trimethoprim_0.125xIC50', 'Trimethoprim_0.25xIC50', 'Trimethoprim_0.5xIC50', 'Trimethoprim_1xIC50']

        dropped_cls = [c for c in e_coli_classes if c.split('_')[-1] in ['0.125xIC50', '0.25xIC50', '0.5xIC50']]

    __, __, test_loader, classes, __ = load_data(root_dir=cmd_args['data_dir'] if args.data_dir is None else args.data_dir,
                                                 train_val_test_dir=[cmd_args['test_dir'], cmd_args['test_dir'], cmd_args['test_dir']],
                                      	         dropped_classes=[],
                                                 batch_size=cmd_args['batch_size'],
                                                 val_split=cmd_args['val_split'],
                                                 crop_size=cmd_args['crop_size'],
                                                 channels=cmd_args.get('channels', [0, 1, 2]),
                                                 data_type=cmd_args.get('data_type', 'tiff'),
                                                 use_e_coli_moa=cmd_args.get('use_e_coli_moa', False))

    print(classes)
    if args.ckpt > 0:
        ckpt_path = glob.glob(os.path.join(args.save_dir, 'ckpts', '*_' + str(args.ckpt) + '_*.tar'))[0]

    elif args.ckpt == -1:
        ckpt_paths = glob.glob(os.path.join(args.save_dir, 'ckpts', '*.tar'))
        ckpt_path = sorted(ckpt_paths, key=lambda s: os.path.basename(s).split('_')[3])[0]

    print(ckpt_path)
    ckpt = torch.load(ckpt_path)

    num_tiles = int((1500 / cmd_args['crop_size']) ** 2)

    model = AvgPoolCNN(num_classes=len(classes) - len(cmd_args['dropped_classes']),
                       num_channels=cmd_args['num_channels'],
                       pretrained=False).to(device)# if not args.use_imagenet else True).to(device)

    if not args.use_imagenet:
        model.load_state_dict(ckpt['model_state_dict'])

    model = model.to(device)

    save_dir = os.path.join(cmd_args['save_dir'], f"{cmd_args['test_dir']}{args.save_dir_addition}")

    make_dir(save_dir)

    test_on_weak(model=model,
                 test_loader=test_loader,
                 classes=classes,
                 training_classes=classes,
                 save_dir=save_dir)
