import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
import json
import math

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

class DataLoader:
    def __init__(
        self,
        experiment='BF',
        params_dir='E_coli_params',
        moa_dict_in=None,
        on_moa=False
    ):

        self.experiment = experiment
        self.params_dir = params_dir
        self.moa_dict_in = moa_dict_in
        self.on_moa = on_moa
    
    def _load_files(
        self,
        channels,
        replicate
    ):
        path_pattern = f'../../../DATA/E_coli_benchmarking/AvgPoolCNN_cross_val_{self.experiment}/test_on_rep_{replicate}/Plate_{replicate}/'

        path = glob(path_pattern)[0]
        labels = np.loadtxt(os.path.join(path, 'labels.txt'))
        preds = np.loadtxt(os.path.join(path, 'preds.txt'))
        test_outputs = np.loadtxt(os.path.join(path, 'test_outputs.txt'))

        with open(os.path.join(path, 'classes.txt'), 'r') as file:
            classes = json.load(file)
        
        return labels, preds, test_outputs, classes

    def load_files(
        self,
        channels_list,
        replicate_list
    ):
        self.channels_list = channels_list
        labels = []
        preds = []
        test_outputs = []
        plate_id = []
        channel_id = []

        for ch_id, ch in enumerate(channels_list):
            for p_id, rep in enumerate(replicate_list):
                labels_, preds_, test_outputs_, classes = self._load_files(ch, rep)

                labels.append(labels_)
                preds.append(preds_)
                test_outputs.append(test_outputs_)
                plate_id.append(np.ones_like(labels_) * p_id)
                channel_id.append(np.ones_like(labels_) * ch_id)

        self.labels = np.hstack(labels)
        self.preds = np.hstack(preds)
        self.test_outputs = np.vstack(test_outputs)
        self.plate_id = np.hstack(plate_id)
        self.channel_id = np.hstack(channel_id)

        self.classes_in = classes
        
        self._get_labels(self.params_dir)

    def _load_labels_from_specs(
        self,
        params_dir
    ):
        d = []
        for l in ['moa_dict', 'moa_dict_inv', 'dose_dict', 'classes', 'moa_classes', 'labels_srtd_by_moa', 'moa_labels_srtd']:
            with open(os.path.join(params_dir, f'{l}.json'), 'r') as f:
                d.append(json.load(f))
        return tuple(d)

    def _get_labels(
        self,
        params_dir
    ):        
        self.moa_dict, self.moa_dict_inv, self.dose_dict, self.classes, self.moa_classes, self.labels_srtd_by_moa, self.moa_labels_srtd = self._load_labels_from_specs(params_dir)

        if self.moa_dict_in:
            self.classes = self.classes_in
            self.moa_dict = self.moa_dict_in

        if self.on_moa:
            self.classes = self.classes_in
            self.moa_dict = dict(zip(self.classes, [c.split('_') for c in self.classes]))
            self.moa_dict_w_dose = {k: (v, self.dose_dict[k.split('_')[1]] if k.split('_')[0] not in ['DMSO', 'Control'] else 0) for k, v in self.moa_dict.items()}
        else:
            self.moa_dict_w_dose = {k: (v, self.dose_dict[k.split('_')[1]] if k.split('_')[0] not in ['DMSO'] else 0) for k, v in self.moa_dict.items()}
            self.moa_to_num = dict(zip(self.moa_classes, [i for i in range(len(self.moa_classes))]))

        
        self.label_to_name = dict(zip([i for i in range(len(self.classes))], self.classes))
        self.mic_id = [self.moa_dict_w_dose[self.label_to_name[l]][1] for l in self.labels]

        
        if self.on_moa:
            self.moa_labels = self.labels
            self.moa_preds = self.preds
        else:
            self.moa_labels = [self.moa_to_num[self.moa_dict_w_dose[self.label_to_name[l]][0]] for l in self.labels]
            self.moa_preds = [self.moa_to_num[self.moa_dict_w_dose[self.label_to_name[l]][0]] for l in self.preds]

        self.labels_as_name = [self.label_to_name[l].split('_')[0] for l in self.labels]
        self.moa_labels_as_name = [[self.moa_dict_w_dose[self.label_to_name[l]][0]][0] for l in self.labels]


class Evaluator:
    def __init__(
        self,
        loader,
    ):
        self.loader = loader

        self.labels = self.loader.labels
        self.preds = self.loader.preds
        self.plate_id = self.loader.plate_id
        self.channel_id = self.loader.channel_id
        self.test_outputs = self.loader.test_outputs
        self.classes = self.loader.classes
        self.ch_name_list = self.loader.channels_list

        self.moa_classes = self.loader.moa_classes
        self.moa_dict = self.loader.moa_dict
        self.moa_dict_inv = self.loader.moa_dict_inv
        self.dose_dict = self.loader.dose_dict

        self.labels_srtd_by_moa = self.loader.labels_srtd_by_moa
        self.moa_labels_strd = self.loader.moa_labels_srtd

        self.moa_dict_w_dose = self.loader.moa_dict_w_dose
        self.label_to_name = self.loader.label_to_name
        self.mic_id = self.loader.mic_id
        self.moa_labels = self.loader.moa_labels
        self.moa_preds = self.loader.moa_preds
        self.labels_as_name = self.loader.labels_as_name
        self.moa_labels_as_name = self.loader.moa_labels_as_name

    def index(
        self,
        input_maps,
        input_choices
    ):
        "Returns boolean list to index array"
        idx_list_ = []
        for maps, choices in zip(input_maps, input_choices):
            if isinstance(choices[0], str) and choices[0] in self.ch_name_list:
                choices = [self.ch_name_list.index(c) for c in choices]
            idx_list_.append(np.logical_or.reduce([np.array(maps) == c for c in choices]))

        return np.logical_and.reduce(idx_list_)

    def p_conditional(
        self,
        dose,
        channel,
        plate
    ):
        "Computes P(cmpd|dose) and returns array reduced to number of classes + DMSO"
        from scipy import special
        idx_list = self.index([self.plate_id, self.channel_id, self.mic_id], [[plate], [channel], [0, dose]])
        p_cmpd_and_dose = special.softmax(self.test_outputs[idx_list])

        idx_list_2 = self.index([[self.moa_dict_w_dose[c][1] for c in self.classes]], [[0, dose]])
        p_dose = (p_cmpd_and_dose[:,idx_list_2]).sum()
        p_cond = p_cmpd_and_dose[:,idx_list_2] / p_dose

        return p_cond, np.array(self.classes)[idx_list_2]

    def compute_conditional_max_accuracy(
        self,
        dose,
        channel,
        plate
    ):
        from sklearn import metrics
        from collections import Counter
        idx_list = self.index([self.plate_id, self.channel_id, self.mic_id], [[plate], [channel], [0, dose]])
        cond_classes = self.p_conditional(dose, channel, plate)[1]
        cond_labels_dict = dict(zip([self.classes.index(c_n) for c_n in cond_classes], [i for i in range(len(cond_classes))]))
        cond_labels = [cond_labels_dict[l] for l in self.labels[idx_list]]
        cond_preds = [p.argmax() for p in self.p_conditional(dose, channel, plate)[0]]

        cond_preds_max = []
        cond_labels_max = []
        for l_ in set(cond_labels):
            l_idx = self.index([cond_labels],[[l_]])
            p_ctr = Counter(np.array(cond_preds)[l_idx])
            cond_labels_max.append(l_)
            cond_preds_max.append(p_ctr.most_common(1)[0][0])

        return metrics.accuracy_score(cond_labels_max, cond_preds_max)

    def compute_conditional_accuracy(
        self,
        dose,
        channel,
        plate
    ):
        from sklearn import metrics
        from collections import Counter
        idx_list = self.index([self.plate_id, self.channel_id, self.mic_id], [[plate], [channel], [0, dose]])
        cond_classes = self.p_conditional(dose, channel, plate)[1]
        cond_labels_dict = dict(zip([self.classes.index(c_n) for c_n in cond_classes], [i for i in range(len(cond_classes))]))
        cond_labels = [cond_labels_dict[l] for l in self.labels[idx_list]]
        cond_preds = [p.argmax() for p in self.p_conditional(dose, channel, plate)[0]]

        return metrics.accuracy_score(cond_labels, cond_preds)
    
    def compute_conditional_moa_max_accuracy(
        self,
        dose,
        channel,
        plate
    ):
        from sklearn import metrics
        from collections import Counter
        idx_list = self.index([self.plate_id, self.channel_id, self.mic_id], [[plate], [channel], [0, dose]])
        cond_classes = self.p_conditional(dose, channel, plate)[1]
        cond_labels_dict = dict(zip([self.classes.index(c_n) for c_n in cond_classes], [i for i in range(len(cond_classes))]))
        cond_labels = [cond_labels_dict[l] for l in self.labels[idx_list]]
        cond_preds = [p.argmax() for p in self.p_conditional(dose, channel, plate)[0]]

        moa_cond_dict = {k: v for k, v in self.moa_dict.items()}

        moa_cond_labels = [moa_cond_dict[cond_classes[l]] for l in cond_labels]
        moa_cond_preds = [moa_cond_dict[cond_classes[l]] for l in cond_preds]
        moa_cond_preds_max = []
        moa_cond_labels_max = []
        for l_ in set(cond_labels):
            l_idx = self.index([cond_labels],[[l_]])
            p_ctr = Counter(np.array(cond_preds)[l_idx])
            moa_cond_labels_max.append(moa_cond_dict[cond_classes[l_]])
            moa_cond_preds_max.append(moa_cond_dict[cond_classes[p_ctr.most_common(1)[0][0]]])
        
        return metrics.accuracy_score(moa_cond_labels_max, moa_cond_preds_max)

    def compute_conditional_moa_accuracy(
        self,
        dose,
        channel,
        plate
    ):
        from sklearn import metrics
        idx_list = self.index([self.plate_id, self.channel_id, self.mic_id], [[plate], [channel], [0, dose]])
        cond_classes = self.p_conditional(dose, channel, plate)[1]
        cond_labels_dict = dict(zip([self.classes.index(c_n) for c_n in cond_classes], [i for i in range(len(cond_classes))]))
        cond_labels = [cond_labels_dict[l] for l in self.labels[idx_list]]
        cond_preds = [p.argmax() for p in self.p_conditional(dose, channel, plate)[0]]

        moa_cond_dict = {k: v for k, v in self.moa_dict.items()}

        moa_cond_labels = [moa_cond_dict[cond_classes[l]] for l in cond_labels]
        moa_cond_preds = [moa_cond_dict[cond_classes[l]] for l in cond_preds]

        return metrics.accuracy_score(moa_cond_labels, moa_cond_preds)


def get_acc_dicts():

    moa_acc_dict = dict()
    moa_max_acc_dict = dict()
    for param in ['BF', 'no_data_aug', 'no_resize', 'on_moa', '16bit']:
    
        loader = DataLoader(experiment='BF' if param == 'BF' else f'BF_{param}',
                            moa_dict_in=load_json('E_coli_params/moa_dict_16bit.json') if param == '16bit' else None,
                            on_moa=True if param == 'on_moa' else False)
        
        loader.load_files(channels_list=['BF'], replicate_list=[1,2,3,4])
        evaluator = Evaluator(loader=loader)
        for pl in [0,1,2,3]:
            if param == 'on_moa':
                moa_acc = evaluator.compute_conditional_accuracy(dose=4, channel=0, plate=pl)
                moa_max_acc = evaluator.compute_conditional_max_accuracy(dose=4, channel=0, plate=pl)
                moa_acc_dict[(param, pl)] = moa_acc
                moa_max_acc_dict[(param, pl)] = moa_max_acc
            
            else:
                moa_acc = evaluator.compute_conditional_moa_accuracy(dose=4, channel=0, plate=pl)
                moa_max_acc = evaluator.compute_conditional_moa_max_accuracy(dose=4, channel=0, plate=pl)
                moa_acc_dict[(param, pl)] = moa_acc
                moa_max_acc_dict[(param, pl)] = moa_max_acc

    return moa_acc_dict, moa_max_acc_dict

def plot_accuracies(acc_dict, title='MoA classification\naccuracy by FOV (1xIC50)', save_name='moa_acc_by_fov_benchmarking.svg'):

    from matplotlib.lines import Line2D
    
    cnd_list = ['BF', 'no_data_aug', 'no_resize', '16bit', 'on_moa']
    
    fig = plt.figure(figsize =(2.5,2.5))
    ax = fig.add_axes([0, 0, 1, 1])
    
    data = []
    for cnd in cnd_list:
        data.append([acc_dict[(cnd,pl)] for pl in [0,1,2,3]])
    
    ax.set_xticklabels([cnd.replace('_', '\n') for cnd in cnd_list], fontsize=9)
    bp = ax.boxplot(data, widths=[0.5] * len(cnd_list), positions=[i + 1 for i in range(len(cnd_list))], showfliers=False, meanline=True, showmeans=True)
    
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']
    
    for k in range(len(cnd_list)):
        bp['means'][k].set_color(colors[k])
        bp['means'][k].set_linewidth(1)
        bp['means'][k].set_linestyle('-')
    
    
    l_width = 0
    for k in range(len(cnd_list)):
        bp['boxes'][k].set_linewidth(l_width)
        bp['medians'][k].set_linewidth(l_width)
    
    for k in range(len(cnd_list) * 2):
        bp['whiskers'][k].set_linewidth(l_width)
        bp['caps'][k].set_linewidth(l_width)
    
    
    for i, (vals, c) in enumerate(zip(data, [i + 1 for i in range(len(cnd_list))])):
        b = c + 0.1
        a = c - 0.1
        for j, m in enumerate(['o', 'v', 'p', 's']):
            ax.scatter([(b - a) * np.random.random_sample(1) + a], vals[j], color=colors[i], marker=m, s=75, alpha=0.5)
    
    
    plt.xlim([0.7, 5.3])
    plt.ylim([0,1.1])
    plt.hlines(1/9, 0.7, 7.3, linestyle='dashed', color='black', alpha=0.5) 
    ax.set_ylabel('Hold-out test accuracy', fontsize=9)
    
    ax.set_title(title, fontsize=10)
        
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Replicate 1',
                              markerfacecolor='black', markersize=10),
                       Line2D([0], [0], marker='v', color='w', label='Replicate 2',
                              markerfacecolor='black', markersize=10),
                       Line2D([0], [0], marker='p', color='w', label='Replicate 3',
                              markerfacecolor='black', markersize=10),
                       Line2D([0], [0], marker='s', color='w', label='Replicate 4',
                              markerfacecolor='black', markersize=10),
                       ]
    
    plt.legend(handles=legend_elements, frameon=False)
    plt.savefig(save_name)
    plt.show()