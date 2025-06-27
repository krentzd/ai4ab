import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
import json
import math

def plot_num_images(
    plotter1,
    plotter2,
    num_classes=23,
    num_doses=4,
    title='Number of images to reach 95% of plateau',
    save_name='diff_pre_train_no_pre_train.svg'
):
    from matplotlib.patches import FancyArrowPatch

    num_images_pre = []
    num_images_no_pre = []
    for d in [1,2,3,4]:
        num_images_pre.append(plotter1.min_num_images(dose=d) * num_classes * num_doses)
        num_images_no_pre.append(plotter2.min_num_images(dose=d) * num_classes * num_doses)

    diff = [x-y for x,y in zip(num_images_no_pre, num_images_pre)]
    percent_diff = [(x-y)/((x+y)/2) for x, y in zip(num_images_no_pre, num_images_pre)]

    fig = plt.figure(figsize=(6,4))
    plt.bar([x-0.15 for x in range(4)], num_images_no_pre, width=0.3, label='No pre-training', color='tab:blue')
    plt.bar([x+0.15 for x in range(4)], num_images_pre, width=0.3, label='Pre-training', color='tab:red')

    for x, (p, y1, y2) in enumerate(zip(percent_diff, num_images_pre, num_images_no_pre)):
        plt.text(x, y2+40, f'{p*100:.1f}%', fontsize=10)
        plt.hlines(y2, x+0.1, x+0.2, color='k')
        plt.arrow(x+0.15, y2, 0, y1-y2+60, head_width=0.1, head_length=50, color='k', overhang=0.2)

    plt.legend(frameon=False)
    plt.ylim([0,2100])
    xtick_classes = ['0.125xIC50', '0.25xIC50', '0.5xIC50', '1xIC50']
    plt.xticks([i for i in range(4)], xtick_classes, rotation=0)
    plt.ylabel('Number of training images')
    plt.title(title)
    plt.savefig(save_name)
    plt.show()

class DataLoader:
    def __init__(
        self,
        experiment='cross_val',
        params_dir='E_coli_params'
    ):

        self.experiment = experiment
        self.params_dir = params_dir

    def _load_files(
        self,
        channels,
        replicate
    ):
        path_pattern = f'../DATA/E_coli/AvgPoolCNN_{self.experiment}_{channels}/test_on_rep_{replicate}/Plate_{replicate}/'
        path = glob(path_pattern)[0]

        feat_vecs = np.loadtxt(os.path.join(path, 'feat_vecs.txt'))
        labels = np.loadtxt(os.path.join(path, 'labels.txt'))
        preds = np.loadtxt(os.path.join(path, 'preds.txt'))
        test_outputs = np.loadtxt(os.path.join(path, 'test_outputs.txt'))
        return feat_vecs, labels, preds, test_outputs

    def load_files(
        self,
        channels_list,
        replicate_list
    ):
        self.channels_list = channels_list
        feat_vecs = []
        labels = []
        preds = []
        test_outputs = []
        plate_id = []
        channel_id = []

        for ch_id, ch in enumerate(channels_list):
            for p_id, rep in enumerate(replicate_list):
                feat_vecs_, labels_, preds_, test_outputs_ = self._load_files(ch, rep)

                feat_vecs.append(feat_vecs_)
                labels.append(labels_)
                preds.append(preds_)
                test_outputs.append(test_outputs_)
                plate_id.append(np.ones_like(labels_) * p_id)
                channel_id.append(np.ones_like(labels_) * ch_id)

        self.feat_vecs = np.vstack(test_outputs)
        self.labels = np.hstack(labels)
        self.preds = np.hstack(preds)
        self.test_outputs = np.vstack(test_outputs)
        self.plate_id = np.hstack(plate_id)
        self.channel_id = np.hstack(channel_id)

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

        self.moa_dict_w_dose = {k: (v, self.dose_dict[k.split('_')[1]] if k not in ['DMSO'] else 0) for k, v in self.moa_dict.items()}
        self.moa_to_num = dict(zip(self.moa_classes, [i for i in range(len(self.moa_classes))]))

        self.label_to_name = dict(zip([i for i in range(len(self.classes))], self.classes))
        self.mic_id = [self.moa_dict_w_dose[self.label_to_name[l]][1] for l in self.labels]

        self.moa_labels = [self.moa_to_num[self.moa_dict_w_dose[self.label_to_name[l]][0]] for l in self.labels]
        self.moa_preds = [self.moa_to_num[self.moa_dict_w_dose[self.label_to_name[l]][0]] for l in self.preds]

        self.labels_as_name = [self.label_to_name[l].split('_')[0] for l in self.labels]
        self.moa_labels_as_name = [[self.moa_dict_w_dose[self.label_to_name[l]][0]][0] for l in self.labels]

class ResultsPlotter:
    def __init__(
        self,
        loader,
    ):
        self.loader = loader

        self.feat_vecs = self.loader.feat_vecs
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
        self.moa_to_num = self.loader.moa_to_num
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

    def make_confusion_matrix(
        self,
        labels,
        preds,
        classes_true,
        classes_pred,
        save_name,
        mode='normalize',
        title='Confusion matrix',
        label_name='compound',
        title_fontsize=20,
        tick_fontsize=12,
        label_fontsize=14
    ):
        from matplotlib.ticker import MultipleLocator

        # Sort label indices by sorted labels
        if label_name == 'compound':
            label_dict = dict(zip([i for i in range(len(classes_true))], [self.labels_srtd_by_moa.index(c) for c in classes_true]))
            labels = [label_dict[l] for l in labels]
            preds = [label_dict[l] for l in preds]
            classes_true = self.labels_srtd_by_moa
            classes_pred = self.labels_srtd_by_moa

        elif label_name == 'MoA':
            label_dict = dict(zip([i for i in classes_true], [self.moa_labels_strd.index(c) for c in classes_true]))

            labels = [label_dict[l] for l in labels]
            preds = [label_dict[l] for l in preds]
            classes_true = self.moa_labels_strd
            classes_pred = self.moa_labels_strd

        if isinstance(labels[0], str):
            labels_dict = dict(zip(classes_true, [i for i in range(len(classes_true))]))
            labels = [labels_dict[l] for l in labels]
            preds = [labels_dict[l] for l in preds]

        def return_counts_array(labels, preds):
            counts_array = np.zeros((len(classes_pred), len(classes_pred)))
            for l in np.unique(labels):
                x = np.array(preds)[np.array(labels) == l]
                p = np.unique(x, return_counts=True)
                if p[0].size > 0:
                    for p_idx, p_val in zip(p[0], p[1]):
                        counts_array[l, p_idx] = p_val
            return counts_array

        counts_array = return_counts_array(labels, preds)
        counts_array_norm = np.zeros_like(counts_array)
        for i, row in enumerate(counts_array):
            counts_array_norm[i] = row / row.sum()

        fig, ax = plt.subplots(figsize=(7,7))
        ax.matshow(counts_array_norm, cmap=plt.cm.Blues)
        plt.gca().xaxis.tick_bottom()

        ax.set_xticklabels([''] + classes_true, rotation=90, fontsize=tick_fontsize)
        ax.set_yticklabels([''] + classes_pred, fontsize=tick_fontsize)

        for i in range(counts_array.shape[1]):
            for j in range(counts_array.shape[0]):
                c = counts_array[j,i]
                c_n = counts_array_norm[j,i]
                ax.text(i, j, f'{int(c)}', va='center', ha='center', c='black' if c_n < 0.5 else 'white')

        plt.title(title, fontsize=title_fontsize)

        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_major_locator(MultipleLocator(1))

        plt.xlabel(f'Predicted {label_name}', fontsize=label_fontsize)
        plt.ylabel(f'True {label_name}', fontsize=label_fontsize)
        print('Saving to', save_name)
        plt.savefig(save_name)


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

    def plot_cond_confusion_matrix(
        self,
        dose,
        channel,
        plate,
        save_name='cond_cmpd_conf_matrix.svg',
        save=False,
        title='Confusion matrix',
        **kwargs
    ):
        from sklearn import metrics
        idx_list = self.index([self.plate_id, self.channel_id, self.mic_id], [[plate], [channel], [0, dose]])
        cond_classes = self.p_conditional(dose, channel, plate)[1]
        cond_labels_dict = dict(zip([self.classes.index(c_n) for c_n in cond_classes], [i for i in range(len(cond_classes))]))
        cond_labels = [cond_labels_dict[l] for l in self.labels[idx_list]]
        cond_preds = [p.argmax() for p in self.p_conditional(dose, channel, plate)[0]]

        cond_classes_ = [c.split('_')[0] for c in cond_classes]

        self.make_confusion_matrix(cond_labels, cond_preds, cond_classes_, cond_classes_, save_name=save_name, title=title, **kwargs)

    def plot_cond_moa_confusion_matrix(
        self,
        dose,
        channel,
        plate,
        save_name='cond_moa_conf_matrix.svg',
        save=False,
        title='Confusion matrix',
        **kwargs
    ):
        from sklearn import metrics
        idx_list = self.index([self.plate_id, self.channel_id, self.mic_id], [[plate], [channel], [0, dose]])
        cond_classes = self.p_conditional(dose, channel, plate)[1]
        cond_labels_dict = dict(zip([self.classes.index(c_n) for c_n in cond_classes], [i for i in range(len(cond_classes))]))
        cond_labels = [cond_labels_dict[l] for l in self.labels[idx_list]]
        cond_preds = [p.argmax() for p in self.p_conditional(dose, channel, plate)[0]]

        # moa_cond_dict = {k: moa_reduced_dict[v] if v in moa_reduced_dict.keys() else v for k, v in moa_dict.items()}
        moa_cond_dict = {k: v for k, v in self.moa_dict.items()}

        moa_cond_labels = [moa_cond_dict[cond_classes[l]] for l in cond_labels]
        moa_cond_preds = [moa_cond_dict[cond_classes[l]] for l in cond_preds]

        self.make_confusion_matrix(moa_cond_labels, moa_cond_preds, sorted(set(moa_cond_dict.values())), sorted(set(moa_cond_dict.values())), save_name=save_name, title=title, **kwargs)

    def _get_umap(
        self,
        data,
        n_components=2,
        n_neighbors=500,
        min_dist=1.,
        metric='cosine'
    ):
        import umap
        umap_ = umap.UMAP(
            n_components=n_components,
            random_state=1,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric
        )
        umap_data = umap_.fit_transform(data)
        return umap_data

    def _get_median_vecs(
        self,
        dose=4,
        concatenate_vecs=True
    ):
        feat_vecs_srtd_ = []
        feat_vecs_med_srtd_ = []
        labels_as_name_srtd_ = []
        labels_as_name_srtd_no_med_ = []
        for l in list(set(self.labels_as_name)):
            if l in ['DMSO', 'Water']:
                idx =  self.index([self.mic_id, self.labels_as_name], [[0], [l]])
            else:
                idx =  self.index([self.mic_id, self.labels_as_name], [[dose], [l]])
            feat_vecs_srtd_.append(self.feat_vecs[idx])
            feat_vecs_med_srtd_.append(np.median(self.feat_vecs[idx], axis=0))
            labels_as_name_srtd_.append(l)
            labels_as_name_srtd_no_med_.append([l] * self.feat_vecs[idx].shape[0])

        feat_vecs_srtd = np.vstack(feat_vecs_srtd_)
        feat_vecs_med_srtd = np.vstack(feat_vecs_med_srtd_)
        labels_as_name_srtd = np.hstack(labels_as_name_srtd_)
        labels_as_name_srtd_no_med = np.hstack(labels_as_name_srtd_no_med_)

        feat_vecs_srtd_concat = np.vstack([feat_vecs_srtd, feat_vecs_med_srtd])

        if concatenate_vecs:
            return feat_vecs_srtd_concat, labels_as_name_srtd, labels_as_name_srtd_no_med

        else:
            feat_vecs_med = []
            labels_med = []
            for l in self.labels_srtd_by_moa:
                l_idx = list(labels_as_name_srtd).index(l)
                feat_vecs_med.append(feat_vecs_med_srtd[l_idx])
                labels_med.append(l)

            feat_vecs_med = np.vstack(feat_vecs_med)
            return feat_vecs_med, labels_med

    def _get_median_vecs_by_dose(
        self
    ):
        feat_vecs_srtd_ = []
        feat_vecs_med_srtd_ = []
        labels_as_name_srtd_ = []
        labels_as_name_srtd_no_med_ = []
        dose_labels_no_med_ = []
        dose_labels_ = []

        for d in [1,2,3,4]:
            for l in list(set(self.labels_as_name)):
                if l in ['DMSO', 'Water']:
                    idx =  self.index([self.mic_id, self.labels_as_name], [[0], [l]])
                else:
                    idx =  self.index([self.mic_id, self.labels_as_name], [[d], [l]])
                feat_vecs_srtd_.append(self.feat_vecs[idx])
                feat_vecs_med_srtd_.append(np.median(self.feat_vecs[idx], axis=0))
                labels_as_name_srtd_.append(l)
                dose_labels_.append(d)
                labels_as_name_srtd_no_med_.append([l] * self.feat_vecs[idx].shape[0])
                dose_labels_no_med_.append([d] * self.feat_vecs[idx].shape[0])

        feat_vecs_srtd = np.vstack(feat_vecs_srtd_)
        feat_vecs_med_srtd = np.vstack(feat_vecs_med_srtd_)
        labels_as_name_srtd = np.hstack(labels_as_name_srtd_)
        labels_as_name_srtd_no_med = np.hstack(labels_as_name_srtd_no_med_)
        dose_labels_no_med = np.hstack(dose_labels_no_med_)
        dose_labels = np.hstack(dose_labels_)
        feat_vecs_srtd_concat = np.vstack([feat_vecs_srtd, feat_vecs_med_srtd])

        return feat_vecs_srtd_concat, labels_as_name_srtd, labels_as_name_srtd_no_med, dose_labels, dose_labels_no_med

    def plot_umap(
        self,
        dose=4,
        n_components=2,
        n_neighbors=500,
        min_dist=1,
        metric='cosine',
        use_moa_labels=False,
        save_name='umap_plot.svg',
        title='UMAP'
    ):

        import matplotlib.patches as mpatches
        import matplotlib as mpl

        from matplotlib.colors import ListedColormap

        cmap = ListedColormap(colors=mpl.colormaps['tab20b'].colors + mpl.colormaps['tab20c'].colors)

        feat_vecs_srtd_concat, labels_as_name_srtd, labels_as_name_srtd_no_med = self._get_median_vecs(dose=dose)

        X = self._get_umap(
            feat_vecs_srtd_concat,
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric
        )

        X_no_med, X_med = X[:-labels_as_name_srtd.shape[0]], X[-labels_as_name_srtd.shape[0]:]

        if use_moa_labels:
            colour_list = ['gainsboro', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:olive']
            handles = []
            for l, c in zip(self.moa_classes, colour_list):
                handles.append(mpatches.Patch(color=c, label=l))

            plt.figure(figsize=(7,7))
            for l in labels_as_name_srtd:
                idx = self.index([labels_as_name_srtd_no_med], [[l]])
                c = colour_list[self.moa_to_num[self.moa_dict_w_dose[f'{l}_1xIC50' if l not in ['DMSO', 'Water'] else l][0]]]
                plt.scatter(X_no_med[idx][:,0], X_no_med[idx][:,1], s=100, alpha=0.2, c=c, edgecolor='grey')

            for l, x in zip(labels_as_name_srtd, X_med):
                c = colour_list[self.moa_to_num[self.moa_dict_w_dose[f'{l}_1xIC50' if l not in ['DMSO', 'Water'] else l][0]]]
                plt.scatter(x[0], x[1], s=150, edgecolor='black', alpha=0.75, label=self.moa_dict_w_dose[f'{l}_1xIC50' if l not in ['DMSO', 'Water'] else l][0], c=c)

        else:
            handles = []
            for l, c in enumerate(labels_as_name_srtd):
                handles.append(mpatches.Patch(color=cmap(int(l)), label=c))

            plt.figure(figsize=(7,7))
            for l, c in enumerate(labels_as_name_srtd):
                idx = self.index([labels_as_name_srtd_no_med], [[c]])
                plt.scatter(X_no_med[idx][:,0], X_no_med[idx][:,1], s=100, alpha=0.2, color=cmap(int(l)) , edgecolor='grey')

            for i, (l, x) in enumerate(zip(labels_as_name_srtd, X_med)):
                plt.scatter(x[0], x[1], s=150, edgecolor='black',alpha=0.75, label=l, color=cmap(int(i)))

        plt.legend(handles=handles)
        plt.title(title)
        plt.savefig(save_name)
        plt.show()

    def plot_umap_by_dose(
        self,
        n_components=2,
        n_neighbors=500*4,
        min_dist=1,
        metric='cosine',
        use_moa_labels=False,
        save_name='umap_plot.svg',
    ):
        import matplotlib as mpl
        import matplotlib.patches as mpatches
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(colors=mpl.colormaps['tab20b'].colors + mpl.colormaps['tab20c'].colors)

        feat_vecs_srtd_concat, labels_as_name_srtd, labels_as_name_srtd_no_med, dose_labels, dose_labels_no_med = self._get_median_vecs_by_dose()

        X = self._get_umap(
            feat_vecs_srtd_concat,
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric
        )

        X_no_med, X_med = X[:-labels_as_name_srtd.shape[0]], X[-labels_as_name_srtd.shape[0]:]
        labels_as_name_srtd_ = sorted([l for l in labels_as_name_srtd[:23]])

        if use_moa_labels:
            # moa_list = ['Control', 'Cell wall (PBP 1)', 'Cell wall (PBP 2)', 'Cell wall (PBP 3)', 'Gyrase', 'Ribosome', 'Membrane integrity', 'RNA polymerase', 'DNA synthesis']
            colour_list = ['black', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:olive']

            colour_dict = dict(zip(self.moa_classes, colour_list))

            for d in [1,2,3,4]:
                plt.figure(figsize=(5,5))
                idx = self.index([dose_labels_no_med], [[d]])
                plt.scatter(X_no_med[~idx][:,0], X_no_med[~idx][:,1], s=50, alpha=0.025, c='grey', edgecolor='grey')

                for l in labels_as_name_srtd[:23]:
                    idx = self.index([labels_as_name_srtd_no_med, dose_labels_no_med], [[l], [d]])
                    c = colour_dict[self.moa_dict_w_dose[f'{l}_1xIC50' if l not in ['DMSO', 'Water'] else l][0]]
                    plt.scatter(X_no_med[idx][:,0], X_no_med[idx][:,1], s=50, alpha=0.15, c=c, edgecolor='grey')

                for l in labels_as_name_srtd[:23]:
                    if l in ['DMSO', 'Water']:
                        d_ = 0
                    else:
                        d_ = d
                    idx = self.index([labels_as_name_srtd, dose_labels], [[l], [d]])
                    c = colour_dict[self.moa_dict_w_dose[f'{l}_1xIC50' if l not in ['DMSO', 'Water'] else l][0]]
                    plt.scatter(X_med[idx][:,0], X_med[idx][:,1], s=100 if not l=='DMSO' else 250, edgecolor='black',alpha=0.75, label=self.moa_dict_w_dose[f'{l}_1xIC50' if l not in ['DMSO', 'Water'] else l][0], c=c)

                plt.axis('off')
                plt.tight_layout()
                plt.savefig(save_name, dpi=300)
                plt.show()

        else:
            for d in [1,2,3,4]:
                plt.figure(figsize=(5,5))
                # idx = self.index([labels_as_name_srtd_no_med, dose_labels_no_med], [[l], [d]])
                idx = self.index([dose_labels_no_med], [[d]])

                plt.scatter(X_no_med[~idx][:,0], X_no_med[~idx][:,1], s=50, alpha=0.025, c='grey')

                for c, l in enumerate(labels_as_name_srtd_):
                    idx = self.index([labels_as_name_srtd_no_med, dose_labels_no_med], [[l], [d]])
                    plt.scatter(X_no_med[idx][:,0], X_no_med[idx][:,1], s=50, alpha=0.15, color=cmap(int(c)), edgecolor='grey')

                for c, l in enumerate(labels_as_name_srtd_):
                    if l in ['DMSO', 'Water']:
                        d_ = 0
                    else:
                        d_ = d
                    idx = self.index([labels_as_name_srtd, dose_labels], [[l], [d]])
                    plt.scatter(X_med[idx][:,0], X_med[idx][:,1], s=100 if not l=='DMSO' else 250, edgecolor='black',alpha=0.75, label=l, color=cmap(int(c)))

                handles = []
                for l_, c_ in enumerate(labels_as_name_srtd_):
                    handles.append(mpatches.Patch(color=cmap(int(l_)), label=c_))

                plt.axis('off')
                plt.tight_layout()
                plt.savefig(save_name, dpi=300)
                plt.show()

    def _cosine_similarity(
        self,
        A,
        B
    ):
        return np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))

    def _make_cosine_similarity_matrix(
        self,
        feat_vecs,
        labels
    ):
        sim_matrix = np.empty((len(labels), len(labels)))
        for x, fvec1 in enumerate(feat_vecs):
            for y, fvec2 in enumerate(feat_vecs):
                sim_matrix[x,y] = self._cosine_similarity(fvec1, fvec2)

        return sim_matrix

    def plot_cosine_similarity_matrix(
        self,
        save_name='cosine_similarity.svg',
        title='Cosine similarity of feature vectors'
    ):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler(with_std=False)

        feat_vecs_med, labels_med = self._get_median_vecs(concatenate_vecs=False)
        feat_vecs_med = scaler.fit_transform(feat_vecs_med)

        sim_matrix = self._make_cosine_similarity_matrix(feat_vecs_med, labels_med)

        plt.figure(figsize=(7,7))
        plt.matshow(sim_matrix, cmap='coolwarm', fignum=0)
        plt.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
        plt.xticks([i for i in range(len(labels_med))], labels_med)
        plt.xticks(rotation=90, fontsize=16)
        plt.yticks([i for i in range(len(labels_med))], labels_med, fontsize=16)
        plt.title(title, fontsize=20)
        plt.savefig(save_name)
        plt.show()


    def _compute_conditional_max_accuracy(
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

    def _compute_conditional_moa_max_accuracy(
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

        # moa_cond_dict = {k: moa_reduced_dict[v] if v in moa_reduced_dict.keys() else v for k, v in moa_dict.items()}
        moa_cond_dict = {k: v for k, v in self.moa_dict.items()}

        moa_cond_labels = [moa_cond_dict[cond_classes[l]] for l in cond_labels]
        moa_cond_preds = [moa_cond_dict[cond_classes[l]] for l in cond_preds]
        moa_cond_preds_max = []
        moa_cond_labels_max = []
        for l_ in set(moa_cond_labels):
            l_idx = self.index([moa_cond_labels],[[l_]])
            p_ctr = Counter(np.array(moa_cond_preds)[l_idx])
            moa_cond_labels_max.append(l_)
            moa_cond_preds_max.append(p_ctr.most_common(1)[0][0])

        return metrics.accuracy_score(moa_cond_labels_max, moa_cond_preds_max)

    def plot_channel_accuracies(
        self,
        use_moa_labels=False,
        save_name='accuracy_plot.svg'
    ):

        from matplotlib.lines import Line2D

        ch_name_list_non_confo = ['BF', 'FM4', 'Hoechst', 'FM4_BF', 'Hoechst_BF', 'Hoechst_FM4', 'Hoechst_FM4_BF']
        fig = plt.figure(figsize =(3.5,2.5))
        ax = fig.add_axes([0, 0, 1, 1])

        data = []
        for ch in ch_name_list_non_confo:
            if use_moa_labels:
                data.append([self._compute_conditional_moa_max_accuracy(4,ch,pl) for pl in [0,1,2,3]])
            else:
                data.append([self._compute_conditional_max_accuracy(4,ch,pl) for pl in [0,1,2,3]])

        print([ch.replace('_', ' + ') for ch in ch_name_list_non_confo])
        ax.set_xticklabels([ch.replace('_', '\n') for ch in ch_name_list_non_confo], fontsize=9)
        bp = ax.boxplot(data, widths=[0.5] * len(ch_name_list_non_confo), positions=[i + 1 for i in range(len(ch_name_list_non_confo))], showfliers=False, meanline=True, showmeans=True)

        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']

        for k in range(len(ch_name_list_non_confo)):
            bp['means'][k].set_color(colors[k])
            bp['means'][k].set_linewidth(1)
            bp['means'][k].set_linestyle('-')


        l_width = 0
        for k in range(len(ch_name_list_non_confo)):
            bp['boxes'][k].set_linewidth(l_width)
            bp['medians'][k].set_linewidth(l_width)

        for k in range(len(ch_name_list_non_confo) * 2):
            bp['whiskers'][k].set_linewidth(l_width)
            bp['caps'][k].set_linewidth(l_width)


        for i, (vals, c) in enumerate(zip(data, [i + 1 for i in range(len(ch_name_list_non_confo))])):
            b = c + 0.1
            a = c - 0.1
            for j, m in enumerate(['o', 'v', 'p', 's']):
                ax.scatter([(b - a) * np.random.random_sample(1) + a], vals[j], color=colors[i], marker=m, s=75, alpha=0.5)


        plt.xlim([0.7, 7.3])
        plt.ylim([0,1.1])
        plt.hlines(1/9 if use_moa_labels else 1/23, 0.7, 7.3, linestyle='dashed', color='black', alpha=0.5)
        ax.set_ylabel('Hold-out test accuracy', fontsize=9)
        if use_moa_labels:
            ax.set_title('MoA classification accuracy by well (1xIC50)', fontsize=10)
        else:
            ax.set_title('Compound classification accuracy by well (1xIC50)', fontsize=10)

        legend_elements = [Line2D([0], [0], marker='o', color='w', label='Replicate 1',
                                  markerfacecolor='black', markersize=10),
                           Line2D([0], [0], marker='v', color='w', label='Replicate 2',
                                  markerfacecolor='black', markersize=10),
                           Line2D([0], [0], marker='p', color='w', label='Replicate 3',
                                  markerfacecolor='black', markersize=10),
                           Line2D([0], [0], marker='s', color='w', label='Replicate 4',
                                  markerfacecolor='black', markersize=10),
                           ]

        # Create the figure
        plt.legend(handles=legend_elements, frameon=False)
        plt.savefig(save_name)
        plt.show()

    def _detection_threshold(
        self,
        plate=2,
        sigma=3
    ):
        from scipy import spatial

        idx_ctrl =  self.index([self.plate_id, self.labels_as_name], [[plate], ['DMSO']])

        median_ctrl_vec = np.median(self.feat_vecs[idx_ctrl], axis=0)
        ctrl_distances = []
        for vec in self.feat_vecs[idx_ctrl]:
            ctrl_distances.append(spatial.distance.cosine(vec, median_ctrl_vec))

        return np.mean(ctrl_distances) + sigma * np.std(ctrl_distances)

    def plot_drug_exposure_detection(
        self,
        plate=2,
        sigma=3,
        save_name='drug_detections_e_coli.svg'
    ):
        from scipy import spatial

        detection_threshold = self._detection_threshold(plate=plate, sigma=sigma)

        colour_list = ['gainsboro', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:olive']
        plt.figure(figsize=(7,5))

        colour_dict = dict(zip(self.moa_classes, colour_list))
        cosine_dist_cmpds_old = np.asarray([0,0,0,0])
        for moa in [m for m in self.moa_classes if m not in ['Control']]:
            cosine_dist_cmpds = []
            for cmpd in self.moa_dict_inv[moa]:
                cosine_dist = []
                for d in [1,2,3,4]:
                    feat_vecs_idx =  self.index([self.plate_id, self.mic_id, self.labels_as_name], [[plate], [d], [cmpd]])
                    feat_vecs_idx_ctrl =  self.index([self.plate_id, self.labels_as_name], [[plate], ['DMSO']])
                    cosine_dist_ = spatial.distance.cosine(np.median(self.feat_vecs[feat_vecs_idx], axis=0), np.median(self.feat_vecs[feat_vecs_idx_ctrl], axis=0))
                    if cosine_dist_ > detection_threshold:
                        cosine_dist.append(1)
                    else:
                        cosine_dist.append(0)

                cosine_dist_cmpds.append(cosine_dist)
            cosine_dist_cmpds = np.array(cosine_dist_cmpds)
            barplot = plt.bar([1,2,3,4], cosine_dist_cmpds.sum(axis=0), label=moa, bottom=cosine_dist_cmpds_old)
            for i in range(4):
                barplot[i].set_color(colour_dict[moa])
            cosine_dist_cmpds_old += cosine_dist_cmpds.sum(axis=0)

        plt.xticks([1,2,3,4], ['0.125', '0.25', '0.5', '1'], fontsize=16)
        plt.ylabel('Number of drugs detected', fontsize=16)
        plt.xlabel('Antibiotic concentration (x IC50)', fontsize=16)
        plt.title(r'Drug exposure detection ($\mu_{DMSO}\pm3\sigma_{DMSO}$)', fontsize=18)
        plt.hlines(22,0.6,4.4, linestyle='dashed')
        plt.yticks([i for i in range(5,21,5)], [i for i in range(5,21,5)], fontsize=16)
        plt.legend()
        plt.savefig('drug_detections_e_coli.svg', dpi=300)

    def plot_roc_curve(
        self,
        save_name='drug_exposure_detection_roc_curve_all_conc_ref.svg'
    ):
        from sklearn import metrics
        from sklearn.preprocessing import StandardScaler
        from scipy import spatial

        scaler = StandardScaler(with_std=False)
        feat_vecs = scaler.fit_transform(self.feat_vecs)

        mic_dict = {1:'0.125xIC50', 2:'0.25xIC50', 3:'0.5xIC50', 4:'1xIC50'}

        max_cosine_dist = 0
        cosine_dist_dict = {}
        y_true_dict = {}
        for d in [1,2,3,4]:
            detected_drugs = []
            for p_idx in [0,1,2,3]:
                cosine_dist = []
                y_true = []
                feat_vecs_idx_ctrl = self.index([self.plate_id, self.labels_as_name], [[p_idx], ['DMSO']])
                for moa in [m for m in self.moa_classes if m not in ['Control']]:
                    cosine_dist_cmpds = []
                    for cmpd in self.moa_dict_inv[moa]:
                            feat_vecs_idx = self.index([self.plate_id, self.mic_id, self.labels_as_name], [[p_idx], [d], [cmpd]])
                            for i in range(np.sum(feat_vecs_idx)):
                                cosine_dist_ = spatial.distance.cosine(feat_vecs[feat_vecs_idx][i], np.median(feat_vecs[feat_vecs_idx_ctrl], axis=0))
                                cosine_dist.append(cosine_dist_)
                                y_true.append(1)

                for i in range(np.sum(feat_vecs_idx_ctrl)):
                    cosine_dist_ = spatial.distance.cosine(feat_vecs[feat_vecs_idx_ctrl][i], np.median(feat_vecs[feat_vecs_idx_ctrl], axis=0))
                    cosine_dist.append(cosine_dist_)
                    y_true.append(0)

                cosine_dist_dict[d, p_idx] = cosine_dist
                y_true_dict[d,p_idx] = y_true

        plt.figure(figsize=(7,5))
        for d in [4,3,2,1]:
            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)

            for p_idx in [0,1,2,3]:

                y_score = np.array(cosine_dist_dict[d,p_idx])

                y_true = y_true_dict[d,p_idx]

                fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
                tprs.append(np.interp(mean_fpr, fpr, tpr))
                roc_auc = metrics.auc(fpr, tpr)
                aucs.append(roc_auc)

            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = metrics.auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            plt.plot(mean_fpr, mean_tpr,
                     label=f'{mic_dict[d]} (AUC={mean_auc:0.2f}$\pm${std_auc:0.2f})',
                     lw=2, alpha=1)

            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            plt.fill_between(mean_fpr, tprs_lower, tprs_upper, alpha=0.2) #label=r'$\pm$ 1 std. dev.')

        plt.plot([0, 1], [0, 1], linestyle='--', lw=1.5, color='tab:red',
                 label='Random classifier', alpha=.8)

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.title('Drug exposure detection ROC curves', fontsize=18)
        plt.legend(loc="lower right", frameon=False, fontsize=12)
        plt.savefig(save_name)
        plt.show()

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
        fpath_ending
    ):
        path_pattern = f'../DATA/E_coli/AvgPoolCNN_training_data_subsampling_BF{fpath_ending}/subsample_{sub}/test_on_rep_{replicate}/Plate_{replicate}/'
        path = glob(path_pattern)[0]

        feat_vecs = np.loadtxt(os.path.join(path, 'feat_vecs.txt'))
        labels = np.loadtxt(os.path.join(path, 'labels.txt'))
        preds = np.loadtxt(os.path.join(path, 'preds.txt'))
        test_outputs = np.loadtxt(os.path.join(path, 'test_outputs.txt'))
        return feat_vecs, labels, preds, test_outputs

    def load_files(
        self,
        pretrained=True
    ):
        if pretrained:
            fpath_ending = '_w_pretraining'
        else:
            fpath_ending = ''

        feat_vecs = []
        labels = []
        preds = []
        test_outputs = []
        plate_id = []
        subsample_id = []

        for sub_id, sub in enumerate([0.02, 0.04, 0.06, 0.08] + [i/10 for i in range(1,11)]):
            for p_id, rep in enumerate([1,2,3,4]):
                feat_vecs_, labels_, preds_, test_outputs_ = self._load_files(sub, rep, fpath_ending)

                feat_vecs.append(feat_vecs_)
                labels.append(labels_)
                preds.append(preds_)
                test_outputs.append(test_outputs_)
                plate_id.append(np.ones_like(labels_) * p_id)
                subsample_id.append(np.ones_like(labels_) * sub_id)

        self.feat_vecs = np.vstack(test_outputs)
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


class SubsamplingPlotter:
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
        for l_ in set(moa_cond_labels):
            l_idx = self.index([moa_cond_labels],[[l_]])
            p_ctr = Counter(np.array(moa_cond_preds)[l_idx])
            moa_cond_labels_max.append(l_)
            moa_cond_preds_max.append(p_ctr.most_common(1)[0][0])

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

    def _func(self, x, M, a):
        return M * (1 - np.exp(-a * x))

    def _func_inv(self, y, M, a):
        return - np.log(1 - y/M) / a

    def _curve_fit(
        self,
        sub_vals,
        mean_acc
    ):
        from scipy.optimize import curve_fit
        from sklearn.metrics import r2_score

        popt, pcov = curve_fit(self._func, sub_vals, mean_acc)

        return popt, pcov

    def plot_curve_fit(
        self,
        title='Exponential curve fit',
        save_name='exponential_curve_fit.svg'
    ):
        sub_vals = self.sub_vals
        mean_acc, std_acc = self._compute_mean_acc(sub_vals)
        popt, __ = self._curve_fit(sub_vals, mean_acc)

        plt.figure(figsize=(6,4))
        xvals = np.linspace(0,1)
        plt.plot(sub_vals, mean_acc, 'o--', linewidth=2, label='Mean accuracy at 1xIC50')
        plt.plot(xvals, self._func(xvals, *popt), label='Exponential curve fit', linewidth=2, alpha=0.8)
        plt.fill_between(sub_vals, mean_acc - std_acc, mean_acc + std_acc, alpha=0.15)

        plt.xlabel('Images per condition')
        plt.ylabel('MoA classification accuracy')

        xticks = [0, 20 / 120, 40 / 120, 60 / 120, 80 / 120, 100 / 120, 1]
        plt.xticks(xticks, [f'{round((i * 150)*0.8):1d}' for i in xticks], fontsize=12)
        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], [f'{i:.2f}' for i in [0, 0.2, 0.4, 0.6, 0.8, 1.0]], fontsize=12)

        plt.ylim(-0.05, 1.05)
        plt.xlim(-0.05,1.05)

        plt.legend(frameon=False)
        plt.title(title)
        plt.savefig(save_name)
        plt.show()

    def min_num_images(
        self,
        dose,
        p_plateau=0.95
    ):
        acc_list = []
        for pl in [0, 1, 2, 3]:
            acc_list.append([self._compute_conditional_moa_max_accuracy(dose, sub, pl) for sub in [i for i in range(0,len(self.sub_vals))]])
        mean_acc = np.array(acc_list).mean(axis=0)
        popt, __ = self._curve_fit(self.sub_vals, mean_acc)
        num_imgs = math.floor(self._func_inv(popt[0] * p_plateau, *popt) * 150 * 0.8)

        return num_imgs

    def plot_plateau(
        self,
        p_plateau=0.95,
        save_name='estimated_number_of_images_all_concentrations.svg'
    ):
        sub_vals = self.sub_vals
        xvals = np.linspace(0,1)
        mic_dict = {1:'0.125xIC50', 2:'0.25xIC50', 3:'0.5xIC50', 4:'1xIC50'}

        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(6,4))

        for ax_, m, clr in zip([(0,0), (0,1), (1,0), (1,1)], [4,3,2,1], ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']):
            ax_i, ax_j = ax_
            x_vals = []

            acc_list = []
            for pl in [0, 1, 2, 3]:
                acc_list.append([self._compute_conditional_moa_max_accuracy(m, sub, pl) for sub in [i for i in range(0,len(sub_vals))]])
            mean_acc = np.array(acc_list).mean(axis=0)

            popt, pcov = self._curve_fit(sub_vals, mean_acc)
            x_vals.append(self._func_inv(popt[0] * p_plateau, *popt))
            ax[ax_i, ax_j].vlines(self._func_inv(popt[0] * p_plateau, *popt), -0.05, p_plateau, color='black', linestyle='dashed')
            ax[ax_i, ax_j].plot(xvals, self._func(xvals, *popt) / popt[0], linewidth=2, label=mic_dict[m], color=clr)

            ax[ax_i, ax_j].hlines(p_plateau, -0.05, self._func_inv(popt[0] * p_plateau, *popt), color='black', linestyle='dashed', label=f'{p_plateau * 100:.0f}% of plateau')

            ax[ax_i, ax_j].set_yticks([0, 0.2, 0.4, 0.6, 0.8, p_plateau])

            xticks = [0, *x_vals, 40 / 120, 60 / 120, 80 / 120, 100 / 120, 1]
            ax[ax_i, ax_j].set_xticks(xticks, [f'{math.floor((i * 150) * 0.8):1d}' for i in xticks])

            ax[ax_i, ax_j].set_xlabel('Images per condition')
            ax[ax_i, ax_j].set_ylabel('% of acc. plateau')
            ax[ax_i, ax_j].set_ylim(-0.05, 1.05)
            ax[ax_i, ax_j].set_xlim(-0.05,1.05)
            ax[ax_i, ax_j].legend(frameon=False, loc='lower right')

        fig.tight_layout()
        fig.suptitle(f'Number of images to reach {p_plateau * 100:.0f}% of plateau')
        fig.subplots_adjust(top=0.9)
        plt.savefig(save_name)
        plt.show()

class LOCODataLoader:
    def __init__(
        self,
        params_dir='E_coli_params'
    ):
        self.params_dir = params_dir
        self.cmpd_names = ['Cefsulodin','PenicillinG', 'Avibactam', 'Mecillinam', 'Meropenem', 'Aztreonam', 'Ceftriaxone', 'Cefepime', 'Clavulanate', 'Relebactam', 'Sulbactam', 'Ciprofloxacin', 'Levofloxacin','Norfloxacin', 'Doxycycline', 'Kanamycin', 'Chloramphenicol', 'Clarithromycin', 'Colistin', 'PolymyxinB', 'Rifampicin', 'Trimethoprim']
    def _load_files(
        self,
        compound,
        replicate
    ):
        path_pattern = f'../DATA/E_coli/AvgPoolCNN_leave_one_cmpd_out_BF/test_on_rep_{replicate}/dropped_{compound}/Plate_{replicate}/'
        path = glob(path_pattern)[0]

        feat_vecs = np.loadtxt(os.path.join(path, 'feat_vecs.txt'))
        labels = np.loadtxt(os.path.join(path, 'labels.txt'))
        preds = np.loadtxt(os.path.join(path, 'preds.txt'))
        test_outputs = np.loadtxt(os.path.join(path, 'test_outputs.txt'))
        return feat_vecs, labels, preds, test_outputs

    def load_files(
        self
    ):
        feat_vecs = []
        labels = []
        preds = []
        test_outputs = []
        plate_id = []
        compound_id = []

        for cmpd_id, cmpd in enumerate(self.cmpd_names):
            for p_id, rep in enumerate([1,2,3,4]):
                feat_vecs_, labels_, preds_, test_outputs_ = self._load_files(cmpd, rep)

                feat_vecs.append(feat_vecs_)
                labels.append(labels_)
                preds.append(preds_)
                test_outputs.append(test_outputs_)
                plate_id.append(np.ones_like(labels_) * p_id)
                compound_id.append(np.ones_like(labels_) * cmpd_id)

        self.feat_vecs = np.vstack(feat_vecs)
        self.labels = np.hstack(labels)
        self.preds = np.hstack(preds)
        self.test_outputs = np.vstack(test_outputs)
        self.plate_id = np.hstack(plate_id)
        self.compound_id = np.hstack(compound_id)

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

        self.moa_dict_w_dose = {k: (v, self.dose_dict[k.split('_')[1]] if k not in ['DMSO'] else 0) for k, v in self.moa_dict.items()}
        self.moa_to_num = dict(zip(self.moa_classes, [i for i in range(len(self.moa_classes))]))

        self.label_to_name = dict(zip([i for i in range(len(self.classes))], self.classes))
        self.mic_id = [self.moa_dict_w_dose[self.label_to_name[l]][1] for l in self.labels]

        self.moa_labels = [self.moa_to_num[self.moa_dict_w_dose[self.label_to_name[l]][0]] for l in self.labels]
        self.moa_preds = [self.moa_to_num[self.moa_dict_w_dose[self.label_to_name[l]][0]] for l in self.preds]

        self.labels_as_name = [self.label_to_name[l].split('_')[0] for l in self.labels]
        self.moa_labels_as_name = [[self.moa_dict_w_dose[self.label_to_name[l]][0]][0] for l in self.labels]

class LOCOPlotter:
    def __init__(
        self,
        loader
    ):
        self.feat_vecs = loader.feat_vecs
        self.labels = loader.labels
        self.moa_dict = loader.moa_dict
        self.plate_id = loader.plate_id
        self.compound_id = loader.compound_id
        self.mic_id = loader.mic_id
        self.test_outputs = loader.test_outputs
        self.classes = loader.classes
        self.moa_dict_w_dose = loader.moa_dict_w_dose
        self.cmpd_names = loader.cmpd_names
        self.label_to_name = loader.label_to_name
        self.moa_to_num = loader.moa_to_num
        self.moa_classes = loader.moa_classes
        self.moa_dict_inv = loader.moa_dict_inv
        self.colour_list = ['gainsboro', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:olive']

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

    def _knn_clf(
        self,
        k=150,
        metric='cosine'
    ):
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.preprocessing import StandardScaler
        from collections import Counter
        from sklearn import metrics

        cmpd_names_ = [c for c in self.cmpd_names[:-2] if c not in ['Colistin', 'PolymyxinB']]
        acc_dict = dict()
        moa_pred_dict = dict()
        for p_id in [0,1,2,3]:
            for conc in ['0.125xIC50', '0.25xIC50', '0.5xIC50', '1xIC50']:
                for cmpd_id, cmpd_name in zip([self.cmpd_names.index(c) for c in cmpd_names_], cmpd_names_):
                    idx_list_drop = self.index([self.compound_id, self.labels, self.plate_id], [[self.cmpd_names.index(cmpd_name)], [self.classes.index(f'{cmpd_name}_{conc}')], [p_id]])
                    idx_list_train = self.index([self.compound_id, self.labels, self.plate_id], [[cmpd_id], [self.classes.index(f'{c}_{conc}') for c in self.cmpd_names if c not in [cmpd_name]], [p_id]])

                    feat_vecs_drop = self.feat_vecs[idx_list_drop]

                    feat_vecs_train = self.feat_vecs[idx_list_train]

                    scaler = StandardScaler(with_std=False)
                    feat_vecs_train = scaler.fit_transform(feat_vecs_train)
                    feat_vecs_drop = scaler.transform(feat_vecs_drop)

                    labels_drop = [self.label_to_name[l] for l in self.labels[idx_list_drop]]
                    moa_labels_drop = [self.moa_dict_w_dose[l][0] for l in labels_drop]

                    labels_train = [self.label_to_name[l] for l in self.labels[idx_list_train]]
                    moa_labels_train = [self.moa_dict_w_dose[l][0] for l in labels_train]

                    clf = KNeighborsClassifier(n_neighbors=k, metric=metric, weights='distance').fit(feat_vecs_train, moa_labels_train)

                    moa_labels_drop_hat = clf.predict(feat_vecs_drop)

                    cntr_labels = Counter(moa_labels_drop)
                    mc_labels = cntr_labels.most_common(1)[0][0]

                    cntr_preds = Counter(moa_labels_drop_hat)
                    mc_preds = cntr_preds.most_common(1)[0][0]

                    moa_hat_as_num = [self.moa_to_num[m] for m in moa_labels_drop_hat]
                    moa_gt_as_num = [self.moa_to_num[m] for m in moa_labels_drop]

                    qk = [self.moa_to_num[mc_preds]] * len(moa_labels_drop_hat)

                    acc_dict[p_id, conc, cmpd_name] = metrics.accuracy_score([mc_labels], [mc_preds])
                    moa_pred_dict[p_id, cmpd_name] = mc_preds

        return acc_dict, moa_pred_dict

    def plot_accuracy(
        self,
        save_name='loco_accuracy.svg'
    ):
        acc_dict, __ = self._knn_clf()
        pbp1 = ['Cefsulodin', 'PenicillinG', 'Sulbactam']
        pbp2 = ['Avibactam', 'Mecillinam', 'Meropenem', 'Relebactam', 'Clavulanate']
        pbp3 = ['Aztreonam', 'Ceftriaxone', 'Cefepime']
        gyr = ['Ciprofloxacin', 'Levofloxacin', 'Norfloxacin']
        rib = ['Doxycycline', 'Kanamycin', 'Chloramphenicol', 'Clarithromycin']
        mem = ['Colistin', 'PolymyxinB']

        data_knn_150 = []
        for conc in ['0.125xIC50', '0.25xIC50', '0.5xIC50', '1xIC50']:
            data_ = []
            for cmpds in [pbp1, pbp2, pbp3, gyr, rib]:
                data_.append([np.mean([acc_dict[p_id, conc, cmpd] for cmpd in cmpds]) for p_id in [0,1,2,3]])

            data_knn_150.append(data_)
        data_knn_150 = np.array(data_knn_150)

        plt.figure(figsize=(7,5))
        for m, i in zip(['Cell wall (PBP 1)', 'Cell wall (PBP 2)', 'Cell wall (PBP 3)', 'Gyrase', 'Ribosome'], range(5)):
            mean_vals = np.array([np.mean(data_knn_150[x,i,:]) for x in range(4)])
            std_vals = np.array([np.std(data_knn_150[x,i,:]) for x in range(4)])
            plt.plot(mean_vals, 'o--', linewidth=2, label=m)
            plt.fill_between([i for i in range(4)], mean_vals-std_vals, mean_vals+std_vals, alpha=0.1, edgecolor='grey')

        plt.xticks([0,1,2,3], ['0.125', '0.25', '0.5', '1'], fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel('Antibiotic concentration (x IC50)', fontsize=16)
        plt.ylabel('LOCO accuracy', fontsize=16)
        plt.title('k-NN LOCO accuracy', fontsize=18)
        plt.legend(frameon=True, fontsize=12)
        plt.savefig(save_name)
        plt.show()

    def plot_moa_predictions(
        self,
        save_name='loco_knn_150_preds.svg'
    ):
        acc_dict, __ = self._knn_clf()

        pbp1 = ['Cefsulodin', 'PenicillinG', 'Sulbactam']
        pbp2 = ['Avibactam', 'Mecillinam', 'Meropenem', 'Relebactam', 'Clavulanate']
        pbp3 = ['Aztreonam', 'Ceftriaxone', 'Cefepime']
        gyr = ['Ciprofloxacin', 'Levofloxacin', 'Norfloxacin']
        rib = ['Doxycycline', 'Kanamycin', 'Chloramphenicol', 'Clarithromycin']

        data = []
        for conc in ['0.125xIC50', '0.25xIC50', '0.5xIC50', '1xIC50']:
            data_ = []
            for cmpds in [pbp1, pbp2, pbp3, gyr, rib]:
                data_.append([np.sum([acc_dict[p_id, conc, cmpd] for cmpd in cmpds]) for p_id in [0,1,2,3]])
            data.append(data_)
        data = np.array(data)

        plt.figure(figsize=(7,5))
        plt.bar([x-0.3 for x in range(5)], data[3,:,0], width=0.2, label='Plate 1', color='tab:blue')
        plt.bar([x-0.1 for x in range(5)], data[3,:,1], width=0.2, label='Plate 2', color='tab:orange')
        plt.bar([x+0.1 for x in range(5)], data[3,:,2], width=0.2, label='Plate 3', color='tab:green')
        plt.bar([x+0.3 for x in range(5)], data[3,:,3], width=0.2, label='Plate 4', color='tab:red')
        plt.legend()
        xtick_classes = ['PBP 1', 'PBP 2', 'PBP 3', 'Gyrase', 'Ribosome']
        plt.xticks([i for i in range(5)], xtick_classes, rotation=90, fontsize=16)
        plt.ylabel('Correctly assigned drugs', fontsize=16)
        for y_, x_ in zip([3, 5, 3, 3, 4], [0, 1, 2, 3, 4]):
            plt.hlines(y_, x_-0.4, x_+0.4, colors='black', linestyles='solid', label='Perfect' if y_ == 3 and x_ == 0 else None)
        plt.yticks(fontsize=16)
        plt.legend(frameon=True, fontsize=12)
        plt.title('k-NN predictions (1xIC50)', fontsize=18)
        plt.savefig(save_name)
        plt.show()

    def plot_compound_predictions(
        self,
        compounds=None,
        plate=0,
        concentration='1xIC50',
        k=150,
        metric='cosine'
    ):
        from sklearn.preprocessing import StandardScaler
        from collections import Counter
        from sklearn import metrics
        from sklearn.neighbors import KNeighborsClassifier

        if not compounds:
            cmpd_names_ = [c for c in self.cmpd_names[:-2]]
        else:
            cmpd_names_ = compounds
        # if c not in ['Clavulanate', 'Sulbactam', 'Relebactam']]

        moa_list = self.moa_classes #['Control', 'Cell wall (PBP 1)', 'Cell wall (PBP 2)', 'Cell wall (PBP 3)', 'Gyrase', 'Ribosome', 'Membrane integrity', 'RNA polymerase', 'DNA synthesis']

        colour_dict = dict(zip(moa_list, self.colour_list))

        p_id = plate
        conc = concentration
        for cmpd_id, cmpd_name in zip([self.cmpd_names.index(c) for c in cmpd_names_], cmpd_names_):
            classes_ = [c for c in self.classes if c not in [f'{cmpd_name}_{d}' for d in ['0.125xIC50', '0.25xIC50', '0.5xIC50', '1xIC50']]]

            label_to_name_ = dict(zip([i for i in range(len(classes_))], classes_))

            idx_list_drop = self.index([self.compound_id, self.labels, self.plate_id], [[cmpd_id], [self.classes.index(f'{cmpd_name}_{conc}')], [p_id]])
            idx_list_train = self.index([self.compound_id, self.labels, self.plate_id], [[cmpd_id], [self.classes.index(f'{c}_{conc}') for c in self.cmpd_names if c not in [cmpd_name]], [p_id]])

            feat_vecs_drop = self.feat_vecs[idx_list_drop]
            feat_vecs_train = self.feat_vecs[idx_list_train]

            scaler = StandardScaler(with_std=False)
            feat_vecs_train = scaler.fit_transform(feat_vecs_train)
            feat_vecs_drop = scaler.transform(feat_vecs_drop)

            labels_drop = [self.label_to_name[l] for l in self.labels[idx_list_drop]]
            moa_labels_drop = [self.moa_dict_w_dose[l][0] for l in labels_drop]

            labels_train = [self.label_to_name[l] for l in self.labels[idx_list_train]]
            moa_labels_train = [self.moa_dict_w_dose[l][0] for l in labels_train]

            clf = KNeighborsClassifier(n_neighbors=k, metric=metric, weights='distance').fit(feat_vecs_train, labels_train)

            moa_labels_drop_hat = clf.predict(feat_vecs_drop)

            cntr_labels = Counter(moa_labels_drop)
            mc_labels = cntr_labels.most_common(1)[0][0]

            cntr_preds = Counter(moa_labels_drop_hat)
            mc_preds = cntr_preds.most_common(1)[0][0]

            print(mc_labels, mc_preds, metrics.accuracy_score([mc_labels], [mc_preds]))

            plt.figure(figsize=(2,1.5))
            bar_labels_ = [x[0] for x in cntr_preds.most_common(2)]
            bar_labels = [x.split('_')[0] for x in bar_labels_] + [''] * (2 - len(bar_labels_)) + ['Other']
            bar_vals_ = [x[1] for x in cntr_preds.most_common(2)]
            bar_vals = [x for x in bar_vals_] + [0] * (2 - len(bar_vals_))
            other_val = 50 - np.sum(bar_vals)
            bar_vals = bar_vals + [other_val]
            print(bar_vals)
            plt.title(f'{cmpd_name}')
            bar = plt.barh([0,1,2], bar_vals)

            for b_id, c_n in enumerate(bar_labels_):
                m_n = self.moa_dict_w_dose[c_n][0]
                col_n = colour_dict[m_n]
                bar[b_id].set_color(col_n)

            bar[-1].set_color('tab:grey')
            plt.yticks([0,1,2], bar_labels)
            plt.xlabel('Number of FOV')
            # plt.ylabel('Nearest neighbours')
            plt.xticks([0,10,20,30,40,50])
            plt.xlim([0,50])
            plt.gca().invert_yaxis()
            # plt.savefig(f'knn_preds_plate_3_150nn_{cmpd_name}.svg')
            plt.show()
            # acc_score = metrics.accuracy_score(moa_labels_drop, moa_labels_drop_hat)
            # acc_score = metric.accuracy_score([mc_labels], [mc_preds])

    def plot_prediction_matrix(
        self
    ):
        from matplotlib.colors import ListedColormap
        # import matplotlib.colors as colors

        __, moa_pred_dict = self._knn_clf()

        cmpd_names_ = ['Cefsulodin', 'PenicillinG', 'Sulbactam', 'Avibactam', 'Mecillinam', 'Meropenem', 'Clavulanate', 'Relebactam', 'Aztreonam', 'Ceftriaxone', 'Cefepime', 'Ciprofloxacin', 'Levofloxacin', 'Norfloxacin', 'Doxycycline', 'Kanamycin', 'Chloramphenicol', 'Clarithromycin']#, 'Colistin', 'PolymyxinB']
        moa_to_num_dict = dict(zip(self.moa_classes, [i for i in range(len(self.moa_classes))]))
        colour_dict = dict(zip(self.moa_classes, self.colour_list))

        moa_to_num_dict = dict(zip(self.moa_classes, [i for i in range(len(self.moa_classes))]))

        for m_n in ['Cell wall (PBP 1)', 'Cell wall (PBP 2)', 'Cell wall (PBP 3)', 'Gyrase', 'Ribosome']:
            moa_pred_matrix = np.empty((4, len(self.moa_dict_inv[m_n])))

            for i in range(4):
                for j, cn in enumerate(self.moa_dict_inv[m_n]):
                    moa_pred_matrix[i,j] = moa_to_num_dict[moa_pred_dict[i, cn]]

            fig, ax = plt.subplots(figsize=(len(self.moa_dict_inv[m_n]) / 2, 2))

            cmap_names = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:olive']

            cmap = ListedColormap([cmap_names[int(i)-1] for i in list(np.unique(moa_pred_matrix))])

            ax.pcolormesh(moa_pred_matrix, cmap=cmap, edgecolors='k', linewidth=1)
            ax.set_aspect(1)
            ax.invert_yaxis()
            # plt.grid()
            plt.xticks([i+0.5 for i in range(len(self.moa_dict_inv[m_n]))], self.moa_dict_inv[m_n], rotation=90)
            plt.yticks([i+0.5 for i in range(4)], [f'Plate {i+1}' for i in range(4)])
            plt.tick_params(axis='both', which='major', labelsize=10, labelbottom=True, bottom=True, top=False, labeltop=False)
            ax.set_title(f'{m_n}', fontsize=11)
            # plt.savefig(f'loco_preds_across_plates_{m_n}.svg')
            plt.show()

class LOMODataLoader:
    def __init__(
        self,
        params_dir='E_coli_params'
    ):
        self.params_dir = params_dir
        self.moa_names = ['Gyrase', 'Membrane', 'PBP1', 'PBP2', 'PBP3', 'Ribosome']
    def _load_files(
        self,
        moa,
        replicate
    ):
        path_pattern = f'../DATA/E_coli/AvgPoolCNN_leave_one_moa_out_BF/test_on_rep_{replicate}/dropped_{moa}/Plate_{replicate}/'
        path = glob(path_pattern)[0]

        feat_vecs = np.loadtxt(os.path.join(path, 'feat_vecs.txt'))
        labels = np.loadtxt(os.path.join(path, 'labels.txt'))
        preds = np.loadtxt(os.path.join(path, 'preds.txt'))

        return feat_vecs, labels, preds

    def load_files(
        self
    ):
        feat_vecs = []
        labels = []
        preds = []
        plate_id = []
        moa_id = []

        for moa_id_, moa in enumerate(self.moa_names):
            for p_id, rep in enumerate([1,2,3,4]):
                feat_vecs_, labels_, preds_ = self._load_files(moa, rep)

                feat_vecs.append(feat_vecs_)
                labels.append(labels_)
                preds.append(preds_)
                plate_id.append(np.ones_like(labels_) * p_id)
                moa_id.append(np.ones_like(labels_) * moa_id_)

        self.feat_vecs = np.vstack(feat_vecs)
        self.labels = np.hstack(labels)
        self.preds = np.hstack(preds)
        self.plate_id = np.hstack(plate_id)
        self.moa_id = np.hstack(moa_id)

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

        self.moa_dict_w_dose = {k: (v, self.dose_dict[k.split('_')[1]] if k not in ['DMSO'] else 0) for k, v in self.moa_dict.items()}
        self.moa_to_num = dict(zip(self.moa_classes, [i for i in range(len(self.moa_classes))]))

        self.label_to_name = dict(zip([i for i in range(len(self.classes))], self.classes))
        self.mic_id = [self.moa_dict_w_dose[self.label_to_name[l]][1] for l in self.labels]

        self.moa_labels = [self.moa_to_num[self.moa_dict_w_dose[self.label_to_name[l]][0]] for l in self.labels]
        self.moa_preds = [self.moa_to_num[self.moa_dict_w_dose[self.label_to_name[l]][0]] for l in self.preds]

        self.labels_as_name = [self.label_to_name[l].split('_')[0] for l in self.labels]
        self.moa_labels_as_name = [[self.moa_dict_w_dose[self.label_to_name[l]][0]][0] for l in self.labels]

class LOMOPlotter:
    def __init__(
        self,
        loader
    ):
        self.feat_vecs = loader.feat_vecs
        self.labels = loader.labels
        self.moa_dict = loader.moa_dict
        self.plate_id = loader.plate_id
        self.moa_id = loader.moa_id
        self.mic_id = loader.mic_id
        self.classes = loader.classes
        self.moa_dict_w_dose = loader.moa_dict_w_dose
        self.label_to_name = loader.label_to_name
        self.moa_to_num = loader.moa_to_num
        self.moa_classes = loader.moa_classes
        self.moa_dict_inv = loader.moa_dict_inv
        self.moa_labels = loader.moa_labels
        self.labels_as_name = loader.labels_as_name

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

    def plot_roc_curves(
        self
    ):
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn import metrics
        from sklearn.neighbors import LocalOutlierFactor

        cmpd_name_list = ['Ciprofloxacin', 'Norfloxacin', 'Levofloxacin', 'Colistin', 'PolymyxinB', 'PenicillinG', 'Cefsulodin', 'Sulbactam', 'Mecillinam', 'Avibactam', 'Meropenem', 'Clavulanate', 'Relebactam', 'Cefepime', 'Ceftriaxone', 'Aztreonam', 'Doxycycline', 'Kanamycin', 'Clarithromycin', 'Chloramphenicol']

        moa_name_list = ['Cell wall (PBP 1)', 'Cell wall (PBP 2)', 'Cell wall (PBP 3)', 'Gyrase', 'Ribosome', 'Membrane integrity']

        addtnl_dropped_cmpds_list = [['Cefsulodin', 'Relebactam', 'Ceftriaxone', 'Doxycycline'],
                                     ['Ciprofloxacin', 'Cefsulodin', 'Relebactam', 'Ceftriaxone', 'Doxycycline'],
                                     ['Ciprofloxacin', 'Relebactam', 'Ceftriaxone', 'Doxycycline'],
                                     ['Ciprofloxacin', 'Cefsulodin', 'Ceftriaxone', 'Doxycycline'],
                                     ['Ciprofloxacin', 'Cefsulodin', 'Relebactam', 'Doxycycline'],
                                     ['Ciprofloxacin', 'Cefsulodin', 'Relebactam', 'Ceftriaxone']]


        clf = LocalOutlierFactor(n_neighbors=9, metric='cosine', novelty=True, p=2)

        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(7,5))


        for ax_coords, m_idx, m_n, clr in zip([(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)], [2,3,4,0,5,1], moa_name_list, ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']):

            ax_i, ax_j = ax_coords

            detect_val_fp_all = []
            detect_val_tp_all = []
            ### Identify outliers in dropped compounds to identify true positives ###
            detected_outliers = 0
            num_drugs = 0


            for rep in [0,1,2,3]:
                detect_val_tp = []
                detect_val_fp = []
                idx = self.index([self.moa_id, self.mic_id, self.plate_id], [[m_idx], [0,1,2,3,4], [rep]])

                feat_vecs_ = self.feat_vecs[idx]

                moa_labels_ = np.array(self.moa_labels)[idx]
                labels_ = np.array(self.labels)[idx]
                labels_as_name_ = np.array(self.labels_as_name)[idx]

                dropped_cmpds = self.moa_dict_inv[m_n]

                # Choose one random compound from existing dataset
                idx_drop = self.index([labels_as_name_], [dropped_cmpds + addtnl_dropped_cmpds_list[m_idx]])
                feat_vecs_train = feat_vecs_[~idx_drop]
                labels_train = moa_labels_[~idx_drop]

                X_train, X_test, y_train, y_test = train_test_split(feat_vecs_train, labels_train, test_size=0.1, random_state=42)

                scaler = StandardScaler(with_std=True)
                X_all = scaler.fit_transform(feat_vecs_)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)

                clf.fit(X_train)

                for dropped_cmpd_ in dropped_cmpds:
                    idx_ = self.index([self.moa_id, self.mic_id, self.plate_id], [[m_idx], [4], [rep]])
                    feat_vecs_ = self.feat_vecs[idx_]

                    labels_as_name_ = np.array(self.labels_as_name)[idx_]

                    X_all = scaler.transform(feat_vecs_)

                    idx_drop_ = self.index([labels_as_name_], [[dropped_cmpd_]])

                    y_pred = clf.decision_function(X_all[idx_drop_])

                    detect_val_tp.append(y_pred)

                for dropped_cmpd_ in addtnl_dropped_cmpds_list[m_idx]:
                    idx_ = self.index([self.moa_id, self.mic_id, self.plate_id], [[m_idx], [4], [rep]])
                    feat_vecs_ = self.feat_vecs[idx_]

                    labels_as_name_ = np.array(self.labels_as_name)[idx_]

                    X_all = scaler.transform(feat_vecs_)

                    idx_drop_ = self.index([labels_as_name_], [[dropped_cmpd_]])

                    y_pred = clf.decision_function(X_all[idx_drop_])

                    detect_val_fp.append(y_pred)

                detect_val_tp_all.append(detect_val_tp)
                detect_val_fp_all.append(detect_val_fp)

            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)

            for i, (fp, tp) in enumerate(zip(np.array(detect_val_fp_all).reshape(4,-1), np.array(detect_val_tp_all).reshape(4,-1))):
                fp = list(-1*fp)
                tp = list(-1*tp)
                y_score = np.array(fp + tp)
                y_true = np.array([0] * len(fp) + [1] * len(tp))
                fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
                tprs.append(np.interp(mean_fpr, fpr, tpr))
                roc_auc = metrics.auc(fpr, tpr)
                aucs.append(roc_auc)

            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = metrics.auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            ax[ax_i, ax_j].plot(mean_fpr, mean_tpr,
                     label=f'AUC={mean_auc:0.2f}',
                     lw=2, alpha=1, color=clr)

            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            ax[ax_i, ax_j].fill_between(mean_fpr, tprs_lower, tprs_upper, alpha=0.2, color=clr)


            ax[ax_i, ax_j].plot([0, 1], [0, 1], linestyle='--', lw=1.5, color='tab:red',alpha=.8) #             label='Random classifier',

            ax[ax_i, ax_j].set_xlim([-0.05, 1.05])
            ax[ax_i, ax_j].set_ylim([-0.05, 1.05])
            ax[ax_i, ax_j].set_xlabel('False Positive Rate')
            ax[ax_i, ax_j].set_ylabel('True Positive Rate')
            ax[ax_i, ax_j].set_title(f'{m_n}', fontsize=10)
            ax[ax_i, ax_j].legend(loc="lower right", frameon=False, fontsize=10)

        fig.tight_layout()
        plt.subplots_adjust(hspace=0.4, wspace=0.4)

        plt.savefig('novelty_detection_roc_curves.svg')
        plt.show()

    def plot_novelty_score(
        self
    ):
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn import metrics
        from sklearn.neighbors import LocalOutlierFactor

        cmpd_name_list = ['Ciprofloxacin', 'Norfloxacin', 'Levofloxacin', 'Colistin', 'PolymyxinB', 'PenicillinG', 'Cefsulodin', 'Sulbactam', 'Mecillinam', 'Avibactam', 'Meropenem', 'Clavulanate', 'Relebactam', 'Cefepime', 'Ceftriaxone', 'Aztreonam', 'Doxycycline', 'Kanamycin', 'Clarithromycin', 'Chloramphenicol']

        moa_name_list = ['Cell wall (PBP 1)', 'Cell wall (PBP 2)', 'Cell wall (PBP 3)', 'Gyrase', 'Ribosome', 'Membrane integrity']

        addtnl_dropped_cmpds_list = [['Cefsulodin', 'Relebactam', 'Ceftriaxone', 'Doxycycline'],
                                     ['Ciprofloxacin', 'Cefsulodin', 'Relebactam', 'Ceftriaxone', 'Doxycycline'],
                                     ['Ciprofloxacin', 'Relebactam', 'Ceftriaxone', 'Doxycycline'],
                                     ['Ciprofloxacin', 'Cefsulodin', 'Ceftriaxone', 'Doxycycline'],
                                     ['Ciprofloxacin', 'Cefsulodin', 'Relebactam', 'Doxycycline'],
                                     ['Ciprofloxacin', 'Cefsulodin', 'Relebactam', 'Ceftriaxone']]


        clf = LocalOutlierFactor(n_neighbors=9, metric='cosine', novelty=True, p=2)

        detect_val_fp_all = []
        detect_val_tp_all = []
        detected_cmpd_name_all = []

        for rep in [0,1,2,3]:

            detect_val_tp = []
            detected_cmpd_name = []
            ### Identify outliers in dropped compounds to identify true positives ###
            detected_outliers = 0
            num_drugs = 0

            for m_idx, m_n in zip([2,3,4,0,5,1], moa_name_list):

                idx = self.index([self.moa_id, self.mic_id, self.plate_id], [[m_idx], [0,1,2,3,4], [rep]])

                feat_vecs_ = self.feat_vecs[idx]

                moa_labels_ = np.array(self.moa_labels)[idx]
                labels_ = np.array(self.labels)[idx]
                labels_as_name_ = np.array(self.labels_as_name)[idx]

                dropped_cmpds = self.moa_dict_inv[m_n]

                idx_drop = self.index([labels_as_name_], [dropped_cmpds + addtnl_dropped_cmpds_list[m_idx]])

                feat_vecs_train = feat_vecs_[~idx_drop]
                labels_train = moa_labels_[~idx_drop]

                X_train, X_test, y_train, y_test = train_test_split(feat_vecs_train, labels_train, test_size=0.1, random_state=42)

                scaler = StandardScaler()
                X_all = scaler.fit_transform(feat_vecs_)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)

                clf.fit(X_train)

                y_pred = clf.predict(X_test)


                for dropped_cmpd_ in dropped_cmpds:
                    idx_ = self.index([self.moa_id, self.mic_id, self.plate_id], [[m_idx], [0,4], [rep]])
                    feat_vecs_ = self.feat_vecs[idx_]
                    # feat_vecs_ = pca.transform(feat_vecs[idx_])

                    labels_as_name_ = np.array(self.labels_as_name)[idx_]

                    X_all = scaler.transform(feat_vecs_)

                    idx_drop_ = self.index([labels_as_name_], [[dropped_cmpd_]])

                    y_pred = clf.predict(X_all[idx_drop_])

                    detect_val = list(y_pred).count(-1) / len(list(y_pred))

                    detect_val_tp.append(detect_val)
                    detected_cmpd_name.append(dropped_cmpd_)

            detect_val_tp_all.append(detect_val_tp)
            detected_cmpd_name_all.append(detected_cmpd_name)

        detect_val_matrix = np.zeros((4, len(detect_val_tp_all[0])))

        for i in range(4):
            detect_val_matrix[i,:] = detect_val_tp_all[i]

        plt.figure(figsize=(7,5))
        clr_list = ['tab:blue'] * 3 + ['tab:orange'] * 5 + ['tab:green'] * 3 + ['tab:red'] * 3 + ['tab:purple'] * 4 +  ['tab:brown'] * 2

        plt.scatter(np.median(detect_val_matrix, axis=0), [i for i in range(20)], marker='D', color=clr_list, s=60, label='MoA not in trainin data')
        for p in range(4):
            plt.scatter(detect_val_matrix[p], [i for i in range(20)], alpha=0.25, marker='D', color=clr_list)

        plt.yticks([i for i in range(20)], detected_cmpd_name_all[p], fontsize=12)
        plt.xticks(fontsize=14)
        plt.grid('on')
        plt.xlabel('Novelty probability', fontsize=14)
        plt.gca().invert_yaxis()
        plt.title(f'Detecting previously unseen MoA', fontsize=18)
        plt.savefig('novelty_probability_plot.svg')
        plt.show()

    def plot_umap(
        self,
        selected_moa='Gyrase'
    ):
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn import metrics
        from sklearn.neighbors import LocalOutlierFactor
        import umap

        moa_names = ['Gyrase', 'Membrane integrity', 'Cell wall (PBP 1)', 'Cell wall (PBP 2)', 'Cell wall (PBP 3)', 'Ribosome']
        moa_idx_ = moa_names.index(selected_moa)

        cmpd_name_list = ['Ciprofloxacin', 'Norfloxacin', 'Levofloxacin', 'Colistin', 'PolymyxinB', 'PenicillinG', 'Cefsulodin', 'Sulbactam', 'Mecillinam', 'Avibactam', 'Meropenem', 'Clavulanate', 'Relebactam', 'Cefepime', 'Ceftriaxone', 'Aztreonam', 'Doxycycline', 'Kanamycin', 'Clarithromycin', 'Chloramphenicol']

        moa_name_list = ['Cell wall (PBP 1)', 'Cell wall (PBP 2)', 'Cell wall (PBP 3)', 'Gyrase', 'Ribosome', 'Membrane integrity']

        addtnl_dropped_cmpds_list = [['Cefsulodin', 'Relebactam', 'Ceftriaxone', 'Doxycycline'],
                                     ['Ciprofloxacin', 'Cefsulodin', 'Relebactam', 'Ceftriaxone', 'Doxycycline'],
                                     ['Ciprofloxacin', 'Relebactam', 'Ceftriaxone', 'Doxycycline'],
                                     ['Ciprofloxacin', 'Cefsulodin', 'Ceftriaxone', 'Doxycycline'],
                                     ['Ciprofloxacin', 'Cefsulodin', 'Relebactam', 'Doxycycline'],
                                     ['Ciprofloxacin', 'Cefsulodin', 'Relebactam', 'Ceftriaxone']]


        clf = LocalOutlierFactor(n_neighbors=9, metric='cosine', novelty=True, p=2)

        detect_val_fp_all = []
        detect_val_tp_all = []
        detected_cmpd_name_all = []


        for rep in [1]:

            detect_val_tp = []
            detected_cmpd_name = []
            detected_outliers = 0
            num_drugs = 0

            for m_idx, m_n in zip([moa_idx_], [selected_moa]):

                idx = self.index([self.moa_id, self.mic_id, self.plate_id], [[m_idx], [0,1,2,3,4], [rep]])

                feat_vecs_ = self.feat_vecs[idx]

                moa_labels_ = np.array(self.moa_labels)[idx]
                labels_ = np.array(self.labels)[idx]
                labels_as_name_ = np.array(self.labels_as_name)[idx]

                dropped_cmpds = self.moa_dict_inv[m_n]

                idx_drop = self.index([labels_as_name_], [dropped_cmpds + addtnl_dropped_cmpds_list[m_idx]])

                feat_vecs_train = feat_vecs_[~idx_drop]
                labels_train = moa_labels_[~idx_drop]

                X_train, X_test, y_train, y_test = train_test_split(feat_vecs_train, labels_train, test_size=0.1, random_state=42)

                scaler = StandardScaler()
                X_all = scaler.fit_transform(feat_vecs_)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)

                clf.fit(X_train)

                y_pred = clf.predict(X_test)

                for dropped_cmpd_ in dropped_cmpds:
                    idx_ = self.index([self.moa_id, self.mic_id, self.plate_id], [[m_idx], [0,4], [rep]])
                    feat_vecs_ = self.feat_vecs[idx_]

                    labels_as_name_ = np.array(self.labels_as_name)[idx_]

                    X_all_ = scaler.transform(feat_vecs_)

                    idx_drop_ = self.index([labels_as_name_], [[dropped_cmpd_]])

                    y_pred = clf.predict(X_all_[idx_drop_])

                    detect_val = list(y_pred).count(-1) / len(list(y_pred))

                    detect_val_tp.append(detect_val)
                    detected_cmpd_name.append(dropped_cmpd_)

            detect_val_tp_all.append(detect_val_tp)
            detected_cmpd_name_all.append(detected_cmpd_name)

        umap_ = umap.UMAP(n_components=2, random_state=1, n_neighbors=25, min_dist=0.3, metric='cosine')
        X = umap_.fit_transform(X_all)

        plt.figure(figsize=(6,6))

        moa_classes = ['Control', 'Cell wall (PBP 1)', 'Cell wall (PBP 2)' 'Cell wall (PBP 3)', 'Gyrase', 'Membrane integrity', 'RNA polymerase', 'DNA synthesis']

        for m_idx, m_n, clr in zip([i for i in [0,1,2,3,4,6,7,8] if i not in [self.moa_to_num[selected_moa]]], self.moa_classes, ['gainsboro', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:brown', 'tab:pink', 'tab:olive']):

            idx_ref = self.index([np.array(self.moa_labels)[idx], np.array(self.mic_id)[idx], np.array(self.plate_id)[idx]], [[m_idx], [0,1,2,3,4], [1]])
            X_ref = X[idx_ref]
            plt.scatter(X_ref[:,0], X_ref[:,1], color='tab:grey', alpha=0.2)


        idx_moa_drop = self.index([np.array(self.moa_labels)[idx], np.array(self.mic_id)[idx], np.array(self.plate_id)[idx]], [[self.moa_to_num[selected_moa]], [0,4], [1]])

        X_scores = clf.score_samples(X_all)[idx_moa_drop]
        radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())

        X_moa_drop = X[idx_moa_drop]

        plt.scatter(X_moa_drop[:,0], X_moa_drop[:,1], color='tab:purple', alpha=1, label=f'{selected_moa} (unseen)', marker='*', s=100, edgecolor='black', linewidth=1)

        scatter = plt.scatter(
            X_moa_drop[:, 0],
            X_moa_drop[:, 1],
            s=2000 * radius,
            edgecolors="tab:red",
            facecolors="none",
            linestyle='solid',
            label="Outlier scores",
            alpha=0.2
        )

        idx_neg_ctrl = self.index([np.array(self.labels_as_name)[idx], np.array(self.mic_id)[idx], np.array(self.plate_id)[idx]], [addtnl_dropped_cmpds_list[moa_idx_], [0,4], [1]])

        X_scores = clf.score_samples(X_all)[idx_neg_ctrl]
        radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())

        X_neg_ctrl = X[idx_neg_ctrl]

        plt.scatter(X_neg_ctrl[:,0], X_neg_ctrl[:,1], color='tab:grey', alpha=0.75, label='Negative control', marker='p', s=75, edgecolor='black', linewidth=1)

        scatter = plt.scatter(
            X_neg_ctrl[:, 0],
            X_neg_ctrl[:, 1],
            s=2000 * radius,
            edgecolors="tab:blue",
            facecolors="none",
            linestyle='solid',
            label="Outlier scores",
            alpha=0.2
        )

        plt.xticks([])
        plt.yticks([])
        plt.title(selected_moa, fontsize=18)
        plt.savefig('lof_plot_ribo.svg')

        plt.show()
