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
from models import WeakCNN, SelfAttentionWeakCNN

def make_prediction_matrix(labels, preds, classes_true, classes_pred, save_name, mode='max'):
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

    fig, ax = plt.subplots(figsize=(15,15))
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
    plt.savefig(save_name)

def make_dir(dir):
    """Create directories including subdirectories"""
    dir_lst = dir.split('/')
    for idx in range(1, len(dir_lst) + 1):
        if not os.path.exists(os.path.join(*dir_lst[:idx])):
            os.mkdir(os.path.join(*dir_lst[:idx]))

def test(model, test_loader, classes, save_dir):
    """
    Compute accuracy on test dataset, plot AUC, show predicted images
    """
    images = []
    labels = []
    preds = []
    class_probs = []
    test_accuracy = 0
    feat_vecs = []

    model.eval()
    with torch.no_grad():
        for data, label in test_loader:

            data = data.to(device)
            label = label.to(device)

            test_output = model(data)
            test_pred = (test_output.argmax(dim=1))
            acc = (test_pred == label).float().mean()

            test_accuracy += acc/len(test_loader)

            images.append(data)
            labels.append(label)
            preds.append(test_pred)
            class_probs.append(F.softmax(test_output))
            # feat_vecs.append(feat_vec)

    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    preds = torch.cat(preds, dim=0)
    class_probs = torch.cat(class_probs, dim=0)
    # feat_vecs = torch.cat(feat_vecs, dim=0)

    plot_predictions(images, labels, preds, 15, test_accuracy, save_dir, classes=classes)

    skplt.metrics.plot_roc(labels.cpu().numpy(), class_probs.cpu().numpy(), plot_micro=False, plot_macro=False)
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'))
    plt.close()

    # skplt.metrics.plot_precision_recall_curve(labels.cpu().numpy(), class_probs.cpu().numpy())
    # plt.savefig(os.path.join(save_dir, 'pr_curve.png'))
    # plt.close()

    skplt.metrics.plot_confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy(), normalize=True)
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()
    ax = skplt.metrics.plot_confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy(), normalize=True, figsize=(15,15))
    ax.set_xticklabels(classes, fontsize="large", rotation=90)
    ax.set_yticklabels(classes, fontsize="large", rotation=0)
    plt.savefig(os.path.join(save_dir, 'confusion_matrix_on_full.svg'), dpi=150)
    plt.close()
    # DONE: Plot t-SNE of final layer
    # tsne_data = get_tsne(feat_vecs.cpu().numpy())
    # plot_representations(tsne_data, labels.cpu().numpy(), classes=classes, save_dir=save_dir)

    # TODO: Plot filters

def test_on_weak_with_dropped(model, test_loader, classes, save_dir, dropped_classes=[]):
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
    fnames =[]
    # label_dict = {0:0, 1:1, 2:2, 3:3, 4:4, 5:6, 6:8, 7:9, 8:11}

    # TODO: Predict antibiotic class from all crops
    model.eval()
    with torch.no_grad():
        for i, (data, label_and_name) in enumerate(tqdm(test_loader)):
            data = data.to(device)
            label = label_and_name[0].to(device)

            test_output, feat_vec = model(data)

            test_pred = (test_output.argmax(dim=1))

            acc = (test_pred == label).float().mean()

            test_accuracy += acc/len(test_loader)

            # images.append(data[:,0])
            test_outputs.append(test_output)
            labels.append(label)
            preds.append(test_pred)
            class_probs.append(F.softmax(test_output, dim=1))
            feat_vecs.append(feat_vec)
            fnames.append(label_and_name[1])

    # images = torch.cat(images, dim=0)
    test_outputs = torch.cat(test_outputs, dim=0)
    labels = torch.cat(labels, dim=0)
    preds = torch.cat(preds, dim=0)
    class_probs = torch.cat(class_probs, dim=0)
    feat_vecs = torch.cat(feat_vecs, dim=0)

    with open(os.path.join(save_dir, 'fnames.txt'), 'w') as f:
        for lines in fnames:
            for line in lines:
                f.write(f"{line}\n")

    np.savetxt(os.path.join(save_dir, 'feat_vecs.txt'), feat_vecs.cpu().numpy())
    np.savetxt(os.path.join(save_dir, 'labels.txt'), labels.cpu().numpy())
    np.savetxt(os.path.join(save_dir, 'preds.txt'), preds.cpu().numpy())
    np.savetxt(os.path.join(save_dir, 'test_outputs.txt'), test_outputs.cpu().numpy())
    np.savetxt(os.path.join(save_dir, 'class_probs.txt'), test_outputs.cpu().numpy())


    # plot_predictions(images, labels, preds, 16, test_accuracy, save_dir, classes=classes, save_name='predicted_images_on_full.png')

    tsne_data = get_tsne(feat_vecs.cpu().numpy())
    umap_data = get_umap(feat_vecs.cpu().numpy())
    pca_data = get_pca(feat_vecs.cpu().numpy())

    plot_representations(tsne_data, labels.cpu().numpy(), classes=classes, save_dir=save_dir, save_name='tsne_plot_on_full.png')
    plot_representations(umap_data, labels.cpu().numpy(), classes=classes, save_dir=save_dir, save_name='umap_plot_on_full.svg')
    plot_representations(pca_data, labels.cpu().numpy(), classes=classes, save_dir=save_dir, save_name='pca_plot_on_full.png')

    # plot_image_representations(tsne_data, images, labels.cpu().numpy(), classes=classes, save_dir=save_dir)
    # plot_image_representations(umap_data, images, labels.cpu().numpy(), classes=classes, save_dir=save_dir, save_name='umap_image_plot.svg')

    # skplt.metrics.plot_roc(labels.cpu().numpy(), class_probs.cpu().numpy(), plot_micro=True, plot_macro=True)
    # plt.savefig(os.path.join(save_dir, 'roc_curve_on_full.png'))
    # plt.close()

    # skplt.metrics.plot_precision_recall_curve(labels.cpu().numpy(), class_probs.cpu().numpy())
    # plt.savefig(os.path.join(save_dir, 'pr_curve_on_full.png'))
    # plt.close()

    # ax = skplt.metrics.plot_confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy(), normalize=True, figsize=(15,15))
    # ax.set_xticklabels(classes, fontsize="large", rotation=90)
    # ax.set_yticklabels(classes, fontsize="large", rotation=0)
    # plt.savefig(os.path.join(save_dir, 'confusion_matrix_on_full.png'), dpi=150)
    # plt.close()
    #
    # make_prediction_matrix(labels.cpu().numpy(), preds.cpu().numpy(), classes, classes, save_name= os.path.join(save_dir, 'confusion_matrix_max.svg'), mode='max')
    #
    #
    # # Plot reduced MoA CM
    # moa_class_as_num = {0: 3,
    #                     1: 3,
    #                     2: 1,
    #                     3: 3,
    #                     4: 5,
    #                     5: 4,
    #                     6: 6,
    #                     7: 4,
    #                     8: 3,
    #                     9: 0,
    #                     10: 4,
    #                     11: 2,
    #                     12: 5,
    #                     13: 2,
    #                     14: 2,
    #                     15: 5,
    #                     16: 1,
    #                     17: 6,
    #                     18: 3,
    #                     19: 7,
    #                     20: 0}
    #
    # moa_labels = [moa_class_as_num[i] for i in labels.cpu().numpy()]
    # moa_preds = [moa_class_as_num[i] for i in preds.cpu().numpy()]
    #
    # moa_classes = ['NA', 'pbp1', 'pbp2', 'pbp3', 'ribosome', 'topoisomerase', 'membrane-integrity', 'dna-synthesis']
    #
    # ax = skplt.metrics.plot_confusion_matrix(moa_labels, moa_preds, normalize=True, figsize=(15,15))
    # ax.set_xticklabels(moa_classes, fontsize="large", rotation=90)
    # ax.set_yticklabels(moa_classes, fontsize="large", rotation=0)
    # plt.savefig(os.path.join(save_dir, 'confusion_matrix_on_moa_w_pbp.svg'), dpi=150)
    # plt.close()
    #
    # make_prediction_matrix(moa_labels, moa_preds, moa_classes, moa_classes, save_name= os.path.join(save_dir, 'confusion_matrix_moa_w_pbp_max.svg'), mode='max')
    #
    # dict_sorted = dict(sorted(moa_class_as_num.items(), key=lambda item: item[1]))
    # sorted_label_dict = dict(zip(dict_sorted.keys(), moa_class_as_num.keys()))
    # sorted_label_dict_inv = dict(zip(moa_class_as_num.keys(), dict_sorted.keys()))
    #
    # labels_sorted = [sorted_label_dict[i] for i in labels.cpu().numpy()]
    # preds_sorted = [sorted_label_dict[i] for i in preds.cpu().numpy()]
    #
    # class_dict_sorted = dict(zip(classes, [i for i in range(len(classes))]))
    # class_dict_sorted_inv = dict(zip([i for i in range(len(classes))], classes))
    #
    # classes_sorted = [class_dict_sorted_inv[sorted_label_dict_inv[class_dict_sorted[c]]] for c in classes]
    #
    # ax = skplt.metrics.plot_confusion_matrix(labels_sorted, preds_sorted, normalize=True, figsize=(15,15))
    # ax.set_xticklabels(classes_sorted, fontsize="large", rotation=90)
    # ax.set_yticklabels(classes_sorted, fontsize="large", rotation=0)
    # plt.savefig(os.path.join(save_dir, 'confusion_matrix_on_full_sorted.png'), dpi=150)
    # plt.close()
    #
    # make_prediction_matrix(labels_sorted, preds_sorted, classes_sorted, classes_sorted, save_name= os.path.join(save_dir, 'confusion_matrix_max_sorted.svg'), mode='max')
    #
    # moa_class_as_num = {0: 1,
    #                     1: 1,
    #                     2: 1,
    #                     3: 1,
    #                     4: 3,
    #                     5: 2,
    #                     6: 4,
    #                     7: 2,
    #                     8: 1,
    #                     9: 0,
    #                     10: 2,
    #                     11: 1,
    #                     12: 3,
    #                     13: 1,
    #                     14: 1,
    #                     15: 3,
    #                     16: 1,
    #                     17: 4,
    #                     18: 1,
    #                     19: 5,
    #                     20: 0}
    #
    # moa_labels = [moa_class_as_num[i] for i in labels.cpu().numpy()]
    # moa_preds = [moa_class_as_num[i] for i in preds.cpu().numpy()]
    #
    # moa_classes = ['NA', 'cell-wall', 'ribosome', 'topoisomerase', 'membrane-integrity', 'dna-synthesis']
    #
    # ax = skplt.metrics.plot_confusion_matrix(moa_labels, moa_preds, normalize=True, figsize=(15,15))
    # ax.set_xticklabels(moa_classes, fontsize="large", rotation=90)
    # ax.set_yticklabels(moa_classes, fontsize="large", rotation=0)
    # plt.savefig(os.path.join(save_dir, 'confusion_matrix_on_moa_w_cw.svg'), dpi=150)
    # plt.close()
    #
    # make_prediction_matrix(moa_labels, moa_preds, moa_classes, moa_classes, save_name= os.path.join(save_dir, 'confusion_matrix_on_moa_w_cw_max.svg'), mode='max')
    #

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

    # label_dict = {0:0, 1:1, 2:2, 3:3, 4:4, 5:6, 6:8, 7:9, 8:11}

    # TODO: Predict antibiotic class from all crops
    model.eval()
    with torch.no_grad():
        for i, (data, label) in enumerate(tqdm(test_loader)):
            data = data.to(device)
            label = label.to(device)

            if kwargs.get('self_attention', False):
                test_output, feat_vec, __ = model(data)

            else:
                test_output, feat_vec = model(data)

            test_pred = (test_output.argmax(dim=1))

            acc = (test_pred == label).float().mean()

            test_accuracy += acc/len(test_loader)

            # images.append(data[:,0])
            test_outputs.append(test_output)
            labels.append(label)
            preds.append(test_pred)
            class_probs.append(F.softmax(test_output, dim=1))
            feat_vecs.append(feat_vec)


    # images = torch.cat(images, dim=0)
    test_outputs = torch.cat(test_outputs, dim=0)
    labels = torch.cat(labels, dim=0)
    preds = torch.cat(preds, dim=0)
    class_probs = torch.cat(class_probs, dim=0)
    feat_vecs = torch.cat(feat_vecs, dim=0)

    np.savetxt(os.path.join(save_dir, 'feat_vecs.txt'), feat_vecs.cpu().numpy())
    np.savetxt(os.path.join(save_dir, 'labels.txt'), labels.cpu().numpy())
    np.savetxt(os.path.join(save_dir, 'preds.txt'), preds.cpu().numpy())
    np.savetxt(os.path.join(save_dir, 'test_outputs.txt'), test_outputs.cpu().numpy())


    # plot_predictions(images, labels, preds, 16, test_accuracy, save_dir, classes=classes, save_name='predicted_images_on_full.png')

    # tsne_data = get_tsne(feat_vecs.cpu().numpy())
    # umap_data = get_umap(feat_vecs.cpu().numpy())
    # pca_data = get_pca(feat_vecs.cpu().numpy())

    # plot_representations(tsne_data, labels.cpu().numpy(), classes=classes, save_dir=save_dir, save_name='tsne_plot_on_full.png')
    # plot_representations(umap_data, labels.cpu().numpy(), classes=classes, save_dir=save_dir, save_name='umap_plot_on_full.svg')
    # plot_representations(pca_data, labels.cpu().numpy(), classes=classes, save_dir=save_dir, save_name='pca_plot_on_full.png')

    # plot_image_representations(tsne_data, images, labels.cpu().numpy(), classes=classes, save_dir=save_dir)
    # plot_image_representations(umap_data, images, labels.cpu().numpy(), classes=classes, save_dir=save_dir, save_name='umap_image_plot.svg')

    # skplt.metrics.plot_roc(labels.cpu().numpy(), class_probs.cpu().numpy(), plot_micro=True, plot_macro=True)
    # plt.savefig(os.path.join(save_dir, 'roc_curve_on_full.png'))
    # plt.close()

    # skplt.metrics.plot_precision_recall_curve(labels.cpu().numpy(), class_probs.cpu().numpy())
    # plt.savefig(os.path.join(save_dir, 'pr_curve_on_full.png'))
    # plt.close()

    ax = skplt.metrics.plot_confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy(), normalize=True, figsize=(15,15))
    ax.set_xticklabels(classes, fontsize="large", rotation=90)
    ax.set_yticklabels(classes, fontsize="large", rotation=0)
    plt.savefig(os.path.join(save_dir, 'confusion_matrix_on_full.png'), dpi=150)
    plt.close()

    make_prediction_matrix(labels.cpu().numpy(), preds.cpu().numpy(), classes, classes, save_name= os.path.join(save_dir, 'confusion_matrix_max.svg'), mode='max')


    # Plot reduced MoA CM
    moa_class_as_num = {0: 3,
                        1: 3,
                        2: 1,
                        3: 3,
                        4: 6,
                        5: 4,
                        6: 5,
                        7: 7,
                        8: 5,
                        9: 3,
                        10: 0,
                        11: 5,
                        12: 2,
                        13: 6,
                        14: 2,
                        15: 2,
                        16: 6,
                        17: 1,
                        18: 7,
                        19: 4,
                        20: 4,
                        21: 3,
                        22: 8,
                        23: 0}

    moa_labels = [moa_class_as_num[i] for i in labels.cpu().numpy()]
    moa_preds = [moa_class_as_num[i] for i in preds.cpu().numpy()]

    moa_classes = ['NA', 'pbp1', 'pbp2', 'pbp3', 'beta-lactamase', 'ribosome', 'topoisomerase', 'membrane-integrity', 'dna-synthesis']

    ax = skplt.metrics.plot_confusion_matrix(moa_labels, moa_preds, normalize=True, figsize=(15,15))
    ax.set_xticklabels(moa_classes, fontsize="large", rotation=90)
    ax.set_yticklabels(moa_classes, fontsize="large", rotation=0)
    plt.savefig(os.path.join(save_dir, 'confusion_matrix_on_moa_w_pbp.svg'), dpi=150)
    plt.close()

    make_prediction_matrix(moa_labels, moa_preds, moa_classes, moa_classes, save_name= os.path.join(save_dir, 'confusion_matrix_moa_w_pbp_max.svg'), mode='max')

    dict_sorted = dict(sorted(moa_class_as_num.items(), key=lambda item: item[1]))
    sorted_label_dict = dict(zip(dict_sorted.keys(), moa_class_as_num.keys()))
    sorted_label_dict_inv = dict(zip(moa_class_as_num.keys(), dict_sorted.keys()))

    labels_sorted = [sorted_label_dict[i] for i in labels.cpu().numpy()]
    preds_sorted = [sorted_label_dict[i] for i in preds.cpu().numpy()]

    class_dict_sorted = dict(zip(classes, [i for i in range(len(classes))]))
    class_dict_sorted_inv = dict(zip([i for i in range(len(classes))], classes))

    classes_sorted = [class_dict_sorted_inv[sorted_label_dict_inv[class_dict_sorted[c]]] for c in classes]

    ax = skplt.metrics.plot_confusion_matrix(labels_sorted, preds_sorted, normalize=True, figsize=(15,15))
    ax.set_xticklabels(classes_sorted, fontsize="large", rotation=90)
    ax.set_yticklabels(classes_sorted, fontsize="large", rotation=0)
    plt.savefig(os.path.join(save_dir, 'confusion_matrix_on_full_sorted.png'), dpi=150)
    plt.close()

    make_prediction_matrix(labels_sorted, preds_sorted, classes_sorted, classes_sorted, save_name= os.path.join(save_dir, 'confusion_matrix_max_sorted.svg'), mode='max')

    moa_class_as_num = {0: 1,
                        1: 1,
                        2: 1,
                        3: 1,
                        4: 4,
                        5: 2,
                        6: 3,
                        7: 5,
                        8: 3,
                        9: 1,
                        10: 0,
                        11: 3,
                        12: 1,
                        13: 4,
                        14: 1,
                        15: 1,
                        16: 4,
                        17: 1,
                        18: 5,
                        19: 2,
                        20: 2,
                        21: 1,
                        22: 6,
                        23: 0}

    moa_labels = [moa_class_as_num[i] for i in labels.cpu().numpy()]
    moa_preds = [moa_class_as_num[i] for i in preds.cpu().numpy()]

    moa_classes = ['NA', 'cell-wall', 'beta-lactamase', 'ribosome', 'topoisomerase', 'membrane-integrity', 'dna-synthesis']

    ax = skplt.metrics.plot_confusion_matrix(moa_labels, moa_preds, normalize=True, figsize=(15,15))
    ax.set_xticklabels(moa_classes, fontsize="large", rotation=90)
    ax.set_yticklabels(moa_classes, fontsize="large", rotation=0)
    plt.savefig(os.path.join(save_dir, 'confusion_matrix_on_moa_w_cw.svg'), dpi=150)
    plt.close()

    make_prediction_matrix(moa_labels, moa_preds, moa_classes, moa_classes, save_name= os.path.join(save_dir, 'confusion_matrix_on_moa_w_cw_max.svg'), mode='max')

if __name__ == '__main__':
    # Parse input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=None)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--ckpt', default=1000, type=int)
    parser.add_argument('--dropped_label', default=0, type=int)
    parser.add_argument('--test_mode', default='normal')
    parser.add_argument('--model', default='5layer')
    parser.add_argument('--dropped_classes', nargs='+', default=[])

    args = parser.parse_args()

    moa_dict = {'AZT': 'pbp3',
                'CEFP': 'pbp3',
                'CEFS': 'pbp1',
                'CEFX': 'pbp3',
                'CIP': 'topoisomerase',
                'CLA': 'beta-lactamase',
                'CM': 'ribosome',
                'COL': 'membrane-integrity',
                'CTM': 'ribosome',
                'CXM': 'pbp3',
                'DMSO': 'NA',
                'DOX': 'ribosome',
                'IMI': 'pbp2',
                'LEV': 'topoisomerase',
                'MEC': 'pbp2',
                'MER': 'pbp2',
                'NOR': 'topoisomerase',
                'PenG': 'pbp1',
                'PolB': 'membrane-integrity',
                'REL': 'beta-lactamase',
                'SUL': 'beta-lactamase',
                'TEM': 'pbp3',
                'TMP': 'dna-synthesis',
                'WT': 'NA'}

    with open(os.path.join(args.save_dir, 'commandline_args.txt'), 'r') as f:
        cmd_args = json.load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device =='cuda':
        print('CUDA available')

    torch.manual_seed(111)

    __, __, test_loader, classes = load_data(root_dir=cmd_args['data_dir'] if args.data_dir is None else args.data_dir,
                                                      train_test_dir=[cmd_args['test_dir'], 'R4'],
                                                      batch_size=cmd_args['batch_size'],
                                                      val_split=cmd_args['val_split'],
                                                      crop_size=cmd_args['crop_size'],
                                                      channels=cmd_args.get('channels', None),
                                                      data_type=cmd_args.get('data_type', 'tiff'))
    print(classes)
    if args.ckpt > 0:
        ckpt_path = glob.glob(os.path.join(args.save_dir, 'ckpts', '*_' + str(args.ckpt) + '_*.tar'))[0]

    elif args.ckpt == -1:
        ckpt_paths = glob.glob(os.path.join(args.save_dir, 'ckpts', '*.tar'))
        ckpt_path = sorted(ckpt_paths, key=lambda s: os.path.basename(s).split('_')[3])[0]

    print(ckpt_path)
    ckpt = torch.load(ckpt_path)

    num_tiles = int((1500 / cmd_args['crop_size']) ** 2)

    if not cmd_args['self_attention']:
        model = WeakCNN(num_classes=len(classes),
                        num_channels=cmd_args['num_channels'],
                        pretrained=False).to(device)

    elif cmd_args['self_attention']:
        model = SelfAttentionWeakCNN(num_classes=len(classes),
                        num_channels=cmd_args['num_channels'],
                        pretrained=False).to(device)

    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)

    if len(cmd_args['dropped_classes']) == 0:
        test_on_weak(model=model,
                     test_loader=test_loader,
                     classes=classes,
                     save_dir=args.save_dir,
                     self_attention=cmd_args['self_attention'])
    elif len(cmd_args['dropped_classes']) > 0:
        test_on_weak_with_dropped(model=model,
                                 test_loader=test_loader,
                                 classes=classes,
                                 save_dir=args.save_dir,
                                 dropped_classes=cmd_args['dropped_classes'])
