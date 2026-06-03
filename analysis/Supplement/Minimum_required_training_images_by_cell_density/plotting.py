import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
import json
import math
from scipy.optimize import curve_fit

class SubsamplingDataLoader:
    def __init__(
        self,
        params_dir='E_coli_params'
    ):
        self.params_dir = params_dir

    def _load_files(
        self,
        sub,
        replicate,
        density
    ):
        path_pattern = f'../../../DATA/E_coli/AvgPoolCNN_training_data_subsampling_BF_{density}_density/subsample_{sub}/test_on_rep_{replicate}/Plate_{replicate}/'
        path = glob(path_pattern)[0]

        labels = np.loadtxt(os.path.join(path, 'labels.txt'))
        preds = np.loadtxt(os.path.join(path, 'preds.txt'))
        test_outputs = np.loadtxt(os.path.join(path, 'test_outputs.txt'))
        return labels, preds, test_outputs

    def load_files(
        self,
        density,
    ):

        labels = []
        preds = []
        test_outputs = []
        plate_id = []
        subsample_id = []

        for sub_id, sub in enumerate(range(1,16)):
            for p_id, rep in enumerate([1,2,3,4]):
                labels_, preds_, test_outputs_ = self._load_files(sub, rep, density)

                labels.append(labels_)
                preds.append(preds_)
                test_outputs.append(test_outputs_)
                plate_id.append(np.ones_like(labels_) * p_id)
                subsample_id.append(np.ones_like(labels_) * sub_id)

        self.labels = np.hstack(labels)
        self.preds = np.hstack(preds)
        self.test_outputs = np.vstack(test_outputs)
        self.plate_id = np.hstack(plate_id)
        self.subsample_id = np.hstack(subsample_id)

        self._get_labels(self.params_dir)

    def _load_labels_from_specs(
        self,
        params_dir
    ):
        d = []
        for l in ['moa_dict', 'dose_dict', 'classes', 'moa_classes', 'labels_srtd_by_moa', 'moa_labels_srtd']:
            with open(os.path.join(params_dir, f'{l}.json'), 'r') as f:
                d.append(json.load(f))
        return tuple(d)

    def _get_labels(
        self,
        params_dir
    ):
        self.moa_dict, self.dose_dict, self.classes, self.moa_classes, self.labels_srtd_by_moa, self.moa_labels_srtd = self._load_labels_from_specs(params_dir)

        self.moa_dict_w_dose = {k: (v, self.dose_dict[k.split('_')[1]] if k not in ['DMSO'] else 0) for k, v in self.moa_dict.items()}
        self.moa_to_num = dict(zip(self.moa_classes, [i for i in range(len(self.moa_classes))]))

        self.label_to_name = dict(zip([i for i in range(len(self.classes))], self.classes))
        self.mic_id = [self.moa_dict_w_dose[self.label_to_name[l]][1] for l in self.labels]

        self.moa_labels = [self.moa_to_num[self.moa_dict_w_dose[self.label_to_name[l]][0]] for l in self.labels]
        self.moa_preds = [self.moa_to_num[self.moa_dict_w_dose[self.label_to_name[l]][0]] for l in self.preds]

        self.labels_as_name = [self.label_to_name[l].split('_')[0] for l in self.labels]
        self.moa_labels_as_name = [[self.moa_dict_w_dose[self.label_to_name[l]][0]][0] for l in self.labels]


class Evaluator:
    def __init__(
        self,
        loader
    ):
        self.labels = loader.labels
        self.moa_dict = loader.moa_dict
        self.plate_id = loader.plate_id
        self.subsample_id = loader.subsample_id
        self.mic_id = loader.mic_id
        self.test_outputs = loader.test_outputs
        self.classes = loader.classes
        self.moa_dict_w_dose = loader.moa_dict_w_dose
        self.sub_vals = [0.02, 0.04, 0.06, 0.08] + [i/10 for i in range(1,11)]

    def index(
        self,
        input_maps,
        input_choices
    ):
        "Returns boolean list to index array"
        idx_list_ = []
        for maps, choices in zip(input_maps, input_choices):
            idx_list_.append(np.logical_or.reduce([np.array(maps) == c for c in choices]))

        return np.logical_and.reduce(idx_list_)

    def p_conditional(
        self,
        dose,
        sub,
        plate
    ):
        "Computes P(cmpd|dose) and returns array reduced to number of classes + DMSO"
        from scipy import special
        idx_list = self.index([self.plate_id, self.subsample_id, self.mic_id], [[plate], [sub], [0, dose]])
        p_cmpd_and_dose = special.softmax(self.test_outputs[idx_list])

        idx_list_2 = self.index([[self.moa_dict_w_dose[c][1] for c in self.classes]], [[0, dose]])
        p_dose = (p_cmpd_and_dose[:,idx_list_2]).sum()
        p_cond = p_cmpd_and_dose[:,idx_list_2] / p_dose

        return p_cond, np.array(self.classes)[idx_list_2]

    def _compute_conditional_moa_max_accuracy(
        self,
        dose,
        sub,
        plate
    ):
        from sklearn import metrics
        from collections import Counter
        idx_list = self.index([self.plate_id, self.subsample_id, self.mic_id], [[plate], [sub], [0, dose]])
        cond_classes = self.p_conditional(dose, sub, plate)[1]
        cond_labels_dict = dict(zip([self.classes.index(c_n) for c_n in cond_classes], [i for i in range(len(cond_classes))]))
        cond_labels = [cond_labels_dict[l] for l in self.labels[idx_list]]
        cond_preds = [p.argmax() for p in self.p_conditional(dose, sub, plate)[0]]

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

    def _compute_mean_acc(
        self,
        sub_vals,
        dose=4
    ):
        acc_list = []
        for pl in [0, 1, 2, 3]:
            acc_list.append([self._compute_conditional_moa_max_accuracy(dose, sub, pl) for sub in [i for i in range(0,len(sub_vals))]])
        mean_acc = np.array(acc_list).mean(axis=0)
        std_acc = np.array(acc_list).std(axis=0)

        return mean_acc, std_acc

def get_accuracy_dicts():

    mean_acc_dict = dict()
    std_acc_dict = dict()
    
    for density in ['low', 'medium', 'high']:
        loader = SubsamplingDataLoader()
        loader.load_files(density=density)
        evaluator = Evaluator(loader)
        
        mean_acc, std_acc = evaluator._compute_mean_acc([i for i in range(1,16)])
    
        mean_acc_dict[density] = mean_acc
        std_acc_dict[density] = std_acc
    
    return mean_acc_dict, std_acc_dict

def func(x, M, a):
    return M * (1 - np.exp(-a * x))

def func_inv(y, M, a):
    return - np.log(1 - y/M) / a

def plot_curve_fit(mean_acc_dict, std_acc_dict):

    sub_vals = [i / 15 for i in range(1,16)]
    p_plateau = 0.95
    
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(9,2))
    xvals = np.linspace(0,1)
    
    plateau_vals = []
    
    dens_dict = {'low': 'Low', 'medium': 'Medium', 'high': 'High'}
    
    for ax_idx, dens, clr in zip([0, 1 ,2], ['low', 'medium', 'high'], ['tab:blue', 'tab:orange', 'tab:green']):

        mean_acc = mean_acc_dict[dens]
        std_acc = std_acc_dict[dens]
        
        popt, pcov = curve_fit(func, sub_vals, mean_acc)
    
        plateau_vals.append(popt[0])
        
        ax[ax_idx].plot(sub_vals, mean_acc, '--', linewidth=2, label='Mean accuracy at 1xIC50')
        ax[ax_idx].plot(xvals, func(xvals, *popt), label='Exponential curve fit', linewidth=2, alpha=0.6)
        ax[ax_idx].fill_between(sub_vals, mean_acc - std_acc, mean_acc + std_acc, alpha=0.15)
        
        ax[ax_idx].set_xlabel('Images per condition', fontsize=8)
        ax[ax_idx].set_ylabel('MoA accuracy', fontsize=8)
            
        xticks = [0, 20 / 120, 40 / 120, 60 / 120, 80 / 120, 100 / 120, 1]
        ax[ax_idx].set_xticks(xticks, [f'{math.floor((i * 45)*0.8):1d}' for i in xticks], fontsize=8)
        ax[ax_idx].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], [f'{i:.1f}' for i in [0, 0.2, 0.4, 0.6, 0.8, 1.0]], fontsize=8)
        
        ax[ax_idx].set_ylim(-0.05, 1.05)
        ax[ax_idx].set_xlim(-0.05,1.05)
    
        ax[ax_idx].legend(frameon=False,loc='lower right', fontsize=8)
    
        ax[ax_idx].set_title(f'{dens_dict[dens]} density', fontsize=10)
    fig.tight_layout()
    plt.savefig('low_medium_high_fits.svg')
    
    plt.show()

def plot_min_num_images(mean_acc_dict, std_acc_dict, p_plateau=0.95):

    # Plot for increasing cell density side-by-side
    sub_vals = sub_vals = [i / 15 for i in range(1,16)]
    xvals = np.linspace(0,1)
    
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(9,2))
    
    for ax_idx, dens, clr in zip([0, 1 ,2], ['low', 'medium', 'high'], ['tab:green', 'tab:blue', 'tab:red']):
        x_vals = []
    
        mean_acc = mean_acc_dict[dens]
    
        popt, pcov = curve_fit(func, sub_vals, mean_acc)
        x_vals.append(func_inv(popt[0] * p_plateau, *popt))
        ax[ax_idx].vlines(func_inv(popt[0] * p_plateau, *popt), -0.05, p_plateau, color='black', linestyle='dashed')
        ax[ax_idx].plot(xvals, func(xvals, *popt) / popt[0], linewidth=2, label=dens, color=clr)
    
        ax[ax_idx].hlines(p_plateau, -0.05, func_inv(popt[0] * p_plateau, *popt), color='black', linestyle='dashed', label=f'{p_plateau * 100:.0f}% of plateau')
    
        ax[ax_idx].set_yticks([0, 0.2, 0.4, 0.6, 0.8, p_plateau], [0, 0.2, 0.4, 0.6, 0.8, p_plateau], fontsize=8)
    
        xticks = [0, *x_vals, 40 / 120, 60 / 120, 80 / 120, 100 / 120, 1]
        ax[ax_idx].set_xticks(xticks, [f'{math.floor((i * 45) * 0.8):1d}' for i in xticks], fontsize=8)
    
        ax[ax_idx].set_xlabel('Images per condition', fontsize=8)
        ax[ax_idx].set_ylabel('% of acc. plateau', fontsize=8)
        ax[ax_idx].set_ylim(-0.05, 1.05)
        ax[ax_idx].set_xlim(-0.05,1.05)
        ax[ax_idx].legend(frameon=False, loc='lower right', fontsize=8)
    
    fig.tight_layout()
    fig.suptitle(f'Number of images to reach {p_plateau * 100:.0f}% of plateau', fontsize=10)
    fig.subplots_adjust(top=0.85)
    plt.savefig('low_medium_high_num_images.svg')
    plt.show()