from glob import glob
import numpy as np 
import os 
import json

from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter


def index(
    input_maps,
    input_choices
):
    "Returns boolean list to index array"
    idx_list_ = []
    for maps, choices in zip(input_maps, input_choices):
        idx_list_.append(np.logical_or.reduce([np.array(maps) == c for c in choices]))

    return np.logical_and.reduce(idx_list_)

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

moa_dict = load_json('HZI_params/moa_dict.json')
moa_sorting_dict = load_json('HZI_params/moa_sorting_dict.json')
cmpd_sorting_dict = load_json('HZI_params/cmpd_sorting_dict.json')
classes_all = load_json('HZI_params/classes_all.json')
classes_all_ = load_json('HZI_params/classes_all_.json')

class Plotter:    
    def __init__(self, dose='1xIC50'):
        self.dose = dose
        
        if dose == '1xIC50':
            self.num_data_reps = 10
        elif dose == '2xIC50':
            self.num_data_reps = 9
        
        self.get_predictions()

        
    def get_predictions(self):
        
        acc_list = []
        acc_moa_list = []
        acc_max_list = []
        acc_moa_max_list = []
        
        preds_dict = dict()
        for data_rep in range(1, self.num_data_reps + 1):
            path_pattern = f'../../../DATA/E_coli_HZI/AvgPoolCNN_cross_val_BF/test_on_rep_{data_rep}/R{data_rep}'
            path = glob(path_pattern)[0]
            
            labels = np.loadtxt(os.path.join(path, 'labels.txt'))
            preds = np.loadtxt(os.path.join(path, 'preds.txt'))
            test_outputs = np.loadtxt(os.path.join(path, 'test_outputs.txt'))
            
            with open(os.path.join(path, 'classes.txt'), 'r') as f:
                classes = json.load(f)
        
            label_to_name = dict(zip([i for i in range(len(classes))], [c for c in classes]))
            labels_as_name = np.asarray([label_to_name[l] for l in labels])
        
            pred_to_name = dict(zip([i for i in range(len(classes_all))], [c for c in classes_all]))
        
            # Get stacked test_outputs
            idx_ = index([[x.split('_')[-1] for x in classes_all_]], [[self.dose, 'DMSO']])
            idx = np.repeat([idx_], test_outputs.shape[0], axis=0)
            test_outputs_ = test_outputs[idx].reshape(test_outputs.shape[0],-1)
            
            l_idx = index([[x.split('_')[-1] for x in labels_as_name]], [[self.dose, '1', '2']])
            
            labels_as_name_ = labels_as_name[l_idx]
            test_outputs__ = test_outputs_[l_idx]
            
            classes__ = np.array(classes_all_)[idx_]
            
            out_to_class = dict(zip([i for i in range(len(classes__))], [c for c in classes__]))
            
            preds_ = [out_to_class[x] for x in test_outputs__.argmax(axis=1)]
        
            labels, preds, classes = list(labels_as_name_), preds_, classes__
        
            labels_moa = [moa_dict[x.split('_')[0]] for x in labels]
            preds_moa = [moa_dict[x.split('_')[0]] for x in preds]
        
            labels_max = []
            preds_max = []
            for l_ in set(labels):
                l_idx = index([labels],[[l_]])
                p_ctr = Counter(np.array(preds)[l_idx])
                labels_max.append(l_)
                preds_max.append(p_ctr.most_common(1)[0][0])
                preds_dict[data_rep, l_] = p_ctr.most_common(1)[0][0]
        
            labels_moa_max = []
            preds_moa_max = []
            for l_ in set(labels):
                l_idx = index([labels],[[l_]])
                p_ctr = Counter(np.array(preds_moa)[l_idx])
                labels_moa_max.append(moa_dict[l_.split('_')[0]])
                preds_moa_max.append(p_ctr.most_common(1)[0][0])
        
            
            acc = metrics.accuracy_score(labels, preds)
            acc_moa = metrics.accuracy_score(labels_moa, preds_moa)
            acc_max = metrics.accuracy_score(labels_max, preds_max)
            acc_moa_max = metrics.accuracy_score(labels_moa_max, preds_moa_max)
            
            acc_list.append(acc)
            acc_moa_list.append(acc_moa)
            acc_max_list.append(acc_max)
            acc_moa_max_list.append(acc_moa_max)

        self.acc_moa_max_list = acc_moa_max_list

        np.savetxt(f'acc_moa_max_list_{self.dose}.txt', self.acc_moa_max_list, delimiter=",", fmt="%s") 
        
        preds_moa_dict = dict()
        counts_moa_dict = dict()
        for val, key in preds_dict.items():
            if moa_dict[key.split('_')[0]] not in counts_moa_dict.keys():
                counts_moa_dict[moa_dict[key.split('_')[0]]] = 0
            counts_moa_dict[moa_dict[key.split('_')[0]]] += 1
            
            if moa_dict[key.split('_')[0]] == moa_dict[val[1].split('_')[0]]:
                if moa_dict[key.split('_')[0]] not in preds_moa_dict.keys():
                    preds_moa_dict[moa_dict[key.split('_')[0]]] = 0
                preds_moa_dict[moa_dict[key.split('_')[0]]] += 1
        
        labels_max = [x[1] for x in preds_dict.keys()]
        preds_max = [x for x in preds_dict.values()]
        
        self.labels_max_srtd = [cmpd_sorting_dict[x.split('_')[0]] for x in labels_max]
        self.preds_max_srtd = [cmpd_sorting_dict[x.split('_')[0]] for x in preds_max]

        np.savetxt(f'labels_max_srtd_{self.dose}.txt', [x.split('_')[1] for x in self.labels_max_srtd], delimiter=",", fmt="%s") 
        np.savetxt(f'preds_max_srtd_{self.dose}.txt', [x.split('_')[1] for x in self.preds_max_srtd], delimiter=",", fmt="%s") 
        
        self.classes_srtd = [x.split('_')[-1] for x in np.unique(self.labels_max_srtd)]
        
        labels_moa_max = [moa_dict[x.split('_')[0]] for x in labels_max]
        preds_moa_max = [moa_dict[x.split('_')[0]] for x in preds_max]
        
        self.labels_moa_max_srtd = [moa_sorting_dict[x] for x in labels_moa_max]
        self.preds_moa_max_srtd = [moa_sorting_dict[x] for x in preds_moa_max]

        np.savetxt(f'labels_moa_max_srtd_{self.dose}.txt', [x.split('_')[1] for x in self.labels_moa_max_srtd], delimiter=",", fmt="%s") 
        np.savetxt(f'preds_moa_max_srtd_{self.dose}.txt', [x.split('_')[1] for x in self.preds_moa_max_srtd], delimiter=",", fmt="%s") 
        
        self.classes_moa_srtd = [x.split('_')[-1] for x in np.unique(self.labels_moa_max_srtd)]
    

    def plot_compound_confusion_matrix(self):
        
        acc = metrics.accuracy_score(self.labels_max_srtd, self.preds_max_srtd)
        cf_matrix = metrics.confusion_matrix(self.labels_max_srtd, self.preds_max_srtd)
        
        fig, ax = plt.subplots(figsize=(7, 7))
        sns.heatmap(cf_matrix, annot=True, cmap='Blues', cbar=False, xticklabels=self.classes_srtd, yticklabels=self.classes_srtd, square=True, linewidths=0.5, linecolor='white')
        plt.title(f'Well-level predictions on replicates 1-{self.num_data_reps}\nConcentration: {self.dose}\nAccuracy: {acc * 100:.2f}%')
        plt.xlabel('Predicted compound')
        plt.ylabel('True compound')
        fig.tight_layout()
        plt.show()

    def plot_moa_confusion_matrix(self):
            
        acc = metrics.accuracy_score(self.labels_moa_max_srtd, self.preds_moa_max_srtd)
        cf_matrix = metrics.confusion_matrix(self.labels_moa_max_srtd, self.preds_moa_max_srtd)
        
        fig, ax = plt.subplots(figsize=(7, 7))
        sns.heatmap(cf_matrix, annot=True, cmap='Blues', cbar=False, xticklabels=self.classes_moa_srtd, yticklabels=self.classes_moa_srtd, square=True, linewidths=0.5, linecolor='white')
        plt.title(f'Well-level predictions on replicates 1-{self.num_data_reps}\nConcentration: {self.dose}\nAccuracy: {acc * 100:.2f}%')
        plt.xlabel('Predicted MoA')
        plt.ylabel('True MoA')
        fig.tight_layout()
        plt.show()

    def plot_moa_accuracy(self):
            
        plt.figure(figsize=(7, 2))
        plt.bar([i for i in range(1,self.num_data_reps + 1)], self.acc_moa_max_list)
        plt.hlines(np.mean(self.acc_moa_max_list), 0, self.num_data_reps + 1, linestyle='dashed', color='k', label='Mean accuracy')
        plt.xticks([i for i in range(1,self.num_data_reps + 1)], [f'R{i}' for i in range(1,self.num_data_reps + 1)])
        plt.xlabel('Hold-out test replicate')
        plt.ylabel('Hold-out test accuracy')
        plt.xlim([0.5,self.num_data_reps + 0.5])
        plt.legend(loc='lower right')
        plt.title(f'Mean well-level MoA accuracy ({self.dose}): {np.mean(self.acc_moa_max_list) * 100:.2f}%')
        plt.show()
