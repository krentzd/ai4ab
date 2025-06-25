import numpy as np
import matplotlib.pyplot as plt

class ResultsPlotter:

    def __init__(self,
                 feat_vecs,
                 labels,
                 preds,
                 plate_id,
                 channel_id,
                 test_outputs,
                 classes,
                 ch_name_list):

        self.feat_vecs = feat_vecs
        self.labels = labels
        self.preds = preds
        self.plate_id = plate_id
        self.channel_id = channel_id
        self.test_outputs = test_outputs
        self.classes = classes
        self.ch_name_list = ch_name_list

        self.moa_classes = ['Control',
                            'Cell wall (PBP 1)',
                            'Cell wall (PBP 2)',
                            'Cell wall (PBP 3)',
                            'Gyrase',
                            'Ribosome',
                            'Membrane integrity',
                            'RNA polymerase',
                            'DNA synthesis']

        self.moa_dict = {'Avibactam_0.125xIC50': 'Cell wall (PBP 2)',
                          'Avibactam_0.25xIC50': 'Cell wall (PBP 2)',
                          'Avibactam_0.5xIC50': 'Cell wall (PBP 2)',
                          'Avibactam_1xIC50': 'Cell wall (PBP 2)',
                          'Aztreonam_0.125xIC50': 'Cell wall (PBP 3)',
                          'Aztreonam_0.25xIC50': 'Cell wall (PBP 3)',
                          'Aztreonam_0.5xIC50': 'Cell wall (PBP 3)',
                          'Aztreonam_1xIC50': 'Cell wall (PBP 3)',
                          'Cefepime_0.125xIC50': 'Cell wall (PBP 3)',
                          'Cefepime_0.25xIC50': 'Cell wall (PBP 3)',
                          'Cefepime_0.5xIC50': 'Cell wall (PBP 3)',
                          'Cefepime_1xIC50': 'Cell wall (PBP 3)',
                          'Cefsulodin_0.125xIC50': 'Cell wall (PBP 1)',
                          'Cefsulodin_0.25xIC50': 'Cell wall (PBP 1)',
                          'Cefsulodin_0.5xIC50': 'Cell wall (PBP 1)',
                          'Cefsulodin_1xIC50': 'Cell wall (PBP 1)',
                          'Ceftriaxone_0.125xIC50': 'Cell wall (PBP 3)',
                          'Ceftriaxone_0.25xIC50': 'Cell wall (PBP 3)',
                          'Ceftriaxone_0.5xIC50': 'Cell wall (PBP 3)',
                          'Ceftriaxone_1xIC50': 'Cell wall (PBP 3)',
                          'Chloramphenicol_0.125xIC50': 'Ribosome',
                          'Chloramphenicol_0.25xIC50': 'Ribosome',
                          'Chloramphenicol_0.5xIC50': 'Ribosome',
                          'Chloramphenicol_1xIC50': 'Ribosome',
                          'Ciprofloxacin_0.125xIC50': 'Gyrase',
                          'Ciprofloxacin_0.25xIC50': 'Gyrase',
                          'Ciprofloxacin_0.5xIC50': 'Gyrase',
                          'Ciprofloxacin_1xIC50': 'Gyrase',
                          'Clarithromycin_0.125xIC50': 'Ribosome',
                          'Clarithromycin_0.25xIC50': 'Ribosome',
                          'Clarithromycin_0.5xIC50': 'Ribosome',
                          'Clarithromycin_1xIC50': 'Ribosome',
                          'Clavulanate_0.125xIC50': 'Cell wall (PBP 2)',
                          'Clavulanate_0.25xIC50': 'Cell wall (PBP 2)',
                          'Clavulanate_0.5xIC50': 'Cell wall (PBP 2)',
                          'Clavulanate_1xIC50': 'Cell wall (PBP 2)',
                          'Colistin_0.125xIC50': 'Membrane integrity',
                          'Colistin_0.25xIC50': 'Membrane integrity',
                          'Colistin_0.5xIC50': 'Membrane integrity',
                          'Colistin_1xIC50': 'Membrane integrity',
                          'DMSO': 'Control',
                          'Doxycycline_0.125xIC50': 'Ribosome',
                          'Doxycycline_0.25xIC50': 'Ribosome',
                          'Doxycycline_0.5xIC50': 'Ribosome',
                          'Doxycycline_1xIC50': 'Ribosome',
                          'Kanamycin_0.125xIC50': 'Ribosome',
                          'Kanamycin_0.25xIC50': 'Ribosome',
                          'Kanamycin_0.5xIC50': 'Ribosome',
                          'Kanamycin_1xIC50': 'Ribosome',
                          'Levofloxacin_0.125xIC50': 'Gyrase',
                          'Levofloxacin_0.25xIC50': 'Gyrase',
                          'Levofloxacin_0.5xIC50': 'Gyrase',
                          'Levofloxacin_1xIC50': 'Gyrase',
                          'Mecillinam_0.125xIC50': 'Cell wall (PBP 2)',
                          'Mecillinam_0.25xIC50': 'Cell wall (PBP 2)',
                          'Mecillinam_0.5xIC50': 'Cell wall (PBP 2)',
                          'Mecillinam_1xIC50': 'Cell wall (PBP 2)',
                          'Meropenem_0.125xIC50': 'Cell wall (PBP 2)',
                          'Meropenem_0.25xIC50': 'Cell wall (PBP 2)',
                          'Meropenem_0.5xIC50': 'Cell wall (PBP 2)',
                          'Meropenem_1xIC50': 'Cell wall (PBP 2)',
                          'Norfloxacin_0.125xIC50': 'Gyrase',
                          'Norfloxacin_0.25xIC50': 'Gyrase',
                          'Norfloxacin_0.5xIC50': 'Gyrase',
                          'Norfloxacin_1xIC50': 'Gyrase',
                          'PenicillinG_0.125xIC50': 'Cell wall (PBP 1)',
                          'PenicillinG_0.25xIC50': 'Cell wall (PBP 1)',
                          'PenicillinG_0.5xIC50': 'Cell wall (PBP 1)',
                          'PenicillinG_1xIC50': 'Cell wall (PBP 1)',
                          'PolymyxinB_0.125xIC50': 'Membrane integrity',
                          'PolymyxinB_0.25xIC50': 'Membrane integrity',
                          'PolymyxinB_0.5xIC50': 'Membrane integrity',
                          'PolymyxinB_1xIC50': 'Membrane integrity',
                          'Relebactam_0.125xIC50': 'Cell wall (PBP 2)',
                          'Relebactam_0.25xIC50': 'Cell wall (PBP 2)',
                          'Relebactam_0.5xIC50': 'Cell wall (PBP 2)',
                          'Relebactam_1xIC50': 'Cell wall (PBP 2)',
                          'Rifampicin_0.125xIC50': 'RNA polymerase',
                          'Rifampicin_0.25xIC50': 'RNA polymerase',
                          'Rifampicin_0.5xIC50': 'RNA polymerase',
                          'Rifampicin_1xIC50': 'RNA polymerase',
                          'Sulbactam_0.125xIC50': 'Cell wall (PBP 1)',
                          'Sulbactam_0.25xIC50': 'Cell wall (PBP 1)',
                          'Sulbactam_0.5xIC50': 'Cell wall (PBP 1)',
                          'Sulbactam_1xIC50': 'Cell wall (PBP 1)',
                          'Trimethoprim_0.125xIC50': 'DNA synthesis',
                          'Trimethoprim_0.25xIC50': 'DNA synthesis',
                          'Trimethoprim_0.5xIC50': 'DNA synthesis',
                          'Trimethoprim_1xIC50': 'DNA synthesis'}

        self.dose_dict = {'0.125xIC50': 1,
                          '0.25xIC50': 2,
                          '0.5xIC50': 3,
                          '1xIC50': 4}

        self.moa_reduced_dict = {'Cell wall (PBP 1)': 'Cell wall',
                                 'Cell wall (PBP 2)': 'Cell wall',
                                 'Cell wall (PBP 3)': 'Cell wall',
                                 'Ribosome (30S)': 'Ribosome',
                                 'Ribosome (50S)': 'Ribosome',}

        self.labels_srtd_by_moa = ['DMSO',
                                  'Cefsulodin',
                                  'PenicillinG',
                                  'Sulbactam',
                                  'Mecillinam',
                                  'Avibactam',
                                  'Meropenem',
                                  'Clavulanate',
                                  'Relebactam',
                                  'Aztreonam',
                                  'Cefepime',
                                  'Ceftriaxone',
                                  'Doxycycline',
                                  'Chloramphenicol',
                                  'Clarithromycin',
                                  'Kanamycin',
                                  'Ciprofloxacin',
                                  'Levofloxacin',
                                  'Norfloxacin',
                                  'Rifampicin',
                                  'Trimethoprim',
                                  'PolymyxinB',
                                  'Colistin']

        self.moa_labels_strd = ['Control', 'Cell wall (PBP 1)', 'Cell wall (PBP 2)', 'Cell wall (PBP 3)', 'Ribosome', 'Gyrase', 'RNA polymerase', 'DNA synthesis', 'Membrane integrity']

        self.make_convenience_dicts()

    def index(self, input_maps, input_choices):
        "Returns boolean list to index array"
        # ch_name_list = ['BF', 'FM4', 'Hoechst', 'FM4_BF', 'Hoechst_BF', 'Hoechst_FM4', 'Hoechst_FM4_BF']
        ch_name_list = ['BF', 'Hoechst_FM4_BF']

        idx_list_ = []
        for maps, choices in zip(input_maps, input_choices):
            if isinstance(choices[0], str) and choices[0] in self.ch_name_list:
                choices = [self.ch_name_list.index(c) for c in choices]
            idx_list_.append(np.logical_or.reduce([np.array(maps) == c for c in choices]))

        return np.logical_and.reduce(idx_list_)

    def make_convenience_dicts(self):
        self.moa_dict_w_dose = {k: (v if v in self.moa_reduced_dict.keys() else v, self.dose_dict[k.split('_')[1]] if k not in ['DMSO'] else 0) for k, v in self.moa_dict.items()}
        self.moa_to_num = dict(zip(self.moa_classes, [i for i in range(len(self.moa_classes))]))

        self.label_to_name = dict(zip([i for i in range(len(self.classes))], self.classes))
        self.mic_id = [self.moa_dict_w_dose[self.label_to_name[l]][1] for l in self.labels]
        self.moa_labels = [self.moa_to_num[self.moa_dict_w_dose[self.label_to_name[l]][0]] for l in self.labels]
        self.moa_preds = [self.moa_to_num[self.moa_dict_w_dose[self.label_to_name[l]][0]] for l in self.preds]

        self.mic_id = [self.moa_dict_w_dose[self.label_to_name[l]][1] for l in self.labels]

        self.labels_as_name = [self.label_to_name[l].split('_')[0] for l in self.labels]
        self.moa_labels_as_name = [[self.moa_dict_w_dose[self.label_to_name[l]][0]][0] for l in self.labels]


    def make_confusion_matrix(self, labels, preds, classes_true, classes_pred, save_name, mode='normalize', title='Confusion matrix', label_name='compound', title_fontsize=20, tick_fontsize=12, label_fontsize=14):
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
            print(label_dict)

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


    def p_conditional(self, dose, channel, plate):
        "Computes P(cmpd|dose) and returns array reduced to number of classes + DMSO"
        from scipy import special
        idx_list = self.index([self.plate_id, self.channel_id, self.mic_id], [[plate], [channel], [0, dose]])
        p_cmpd_and_dose = special.softmax(self.test_outputs[idx_list])

        idx_list_2 = self.index([[self.moa_dict_w_dose[c][1] for c in self.classes]], [[0, dose]])
        p_dose = (p_cmpd_and_dose[:,idx_list_2]).sum()
        p_cond = p_cmpd_and_dose[:,idx_list_2] / p_dose

        return p_cond, np.array(self.classes)[idx_list_2]

    def plot_cond_confusion_matrix(self, dose, channel, plate, save_name='cond_cmpd_conf_matrix.svg', save=False, title='Confusion matrix', **kwargs):
        from sklearn import metrics
        idx_list = self.index([self.plate_id, self.channel_id, self.mic_id], [[plate], [channel], [0, dose]])
        cond_classes = self.p_conditional(dose, channel, plate)[1]
        cond_labels_dict = dict(zip([self.classes.index(c_n) for c_n in cond_classes], [i for i in range(len(cond_classes))]))
        cond_labels = [cond_labels_dict[l] for l in self.labels[idx_list]]
        cond_preds = [p.argmax() for p in self.p_conditional(dose, channel, plate)[0]]

        cond_classes_ = [c.split('_')[0] for c in cond_classes]

        self.make_confusion_matrix(cond_labels, cond_preds, cond_classes_, cond_classes_, save_name=save_name, title=title, **kwargs)

    def plot_cond_moa_confusion_matrix(self, dose, channel, plate, save_name='cond_moa_conf_matrix.svg', save=False, title='Confusion matrix', **kwargs):
        from sklearn import metrics
        idx_list = self.index([self.plate_id, self.channel_id, self.mic_id], [[plate], [channel], [0, dose]])
        cond_classes = self.p_conditional(dose, channel, plate)[1]
        cond_labels_dict = dict(zip([self.classes.index(c_n) for c_n in cond_classes], [i for i in range(len(cond_classes))]))
        cond_labels = [cond_labels_dict[l] for l in self.labels[idx_list]]
        cond_preds = [p.argmax() for p in self.p_conditional(dose, channel, plate)[0]]

        # moa_cond_dict = {k: moa_reduced_dict[v] if v in moa_reduced_dict.keys() else v for k, v in moa_dict.items()}
        moa_cond_dict = {k: v if v in self.moa_reduced_dict.keys() else v for k, v in self.moa_dict.items()}

        moa_cond_labels = [moa_cond_dict[cond_classes[l]] for l in cond_labels]
        moa_cond_preds = [moa_cond_dict[cond_classes[l]] for l in cond_preds]

        self.make_confusion_matrix(moa_cond_labels, moa_cond_preds, sorted(set(moa_cond_dict.values())), sorted(set(moa_cond_dict.values())), save_name=save_name, title=title, **kwargs)
