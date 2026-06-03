import os 
from glob import glob
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import random
from PIL import Image

from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib
from scipy import linalg

def load_json(path):
    """Load data from a JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def index(
    input_maps,
    input_choices
):
    "Returns boolean list to index array"
    idx_list_ = []
    for maps, choices in zip(input_maps, input_choices):
        idx_list_.append(np.logical_or.reduce([np.array(maps) == c for c in choices]))

    return np.logical_and.reduce(idx_list_)

def cosine_similarity(
    A,
    B
):
    return np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))

def get_umap(
    data,
    n_components=2,
    n_neighbors=500,
    min_dist=1.,
    metric='cosine'
):
    import umap
    umap_ = umap.UMAP(
        n_components=n_components,
        random_state=0,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric
    )
    umap_data = umap_.fit_transform(data)
    return umap_data

def get_median_vecs(
    feat_vecs,
    labels_as_name,
    labels_as_dose,
    dose=4
):
    feat_vecs_srtd_ = []
    feat_vecs_med_srtd_ = []
    labels_as_name_srtd_ = []
    labels_as_name_srtd_no_med_ = []
    for l in list(set(labels_as_name)):
        if l in ['DMSO', 'Water']:
            idx =  index([labels_as_dose, labels_as_name], [[0], [l]])
        else:
            idx =  index([labels_as_dose, labels_as_name], [[dose], [l]])
        feat_vecs_srtd_.append(feat_vecs[idx])
        feat_vecs_med_srtd_.append(np.median(feat_vecs[idx], axis=0))
        labels_as_name_srtd_.append(l)
        labels_as_name_srtd_no_med_.append([l] * feat_vecs[idx].shape[0])

    feat_vecs_srtd = np.vstack(feat_vecs_srtd_)
    feat_vecs_med_srtd = np.vstack(feat_vecs_med_srtd_)
    labels_as_name_srtd = np.hstack(labels_as_name_srtd_)
    labels_as_name_srtd_no_med = np.hstack(labels_as_name_srtd_no_med_)

    feat_vecs_srtd_concat = np.vstack([feat_vecs_srtd, feat_vecs_med_srtd])

    return feat_vecs_srtd_concat, labels_as_name_srtd, labels_as_name_srtd_no_med


def plot_umap(
    feat_vecs,
    labels_as_name,
    labels_as_dose,
    dose=4,
    n_components=2,
    n_neighbors=500,
    min_dist=1,
    metric='cosine',
    use_moa_labels=True,
    save_name='umap_plot_e_coli_on_klebs.pdf',
    title='UMAP'
):

    import matplotlib.patches as mpatches
    import matplotlib as mpl

    from matplotlib.colors import ListedColormap
    import textalloc as ta

    moa_col_dict = load_json('plotting_params/moa_col_dict.json')
    moa_col_dict_names = load_json('plotting_params/moa_col_dict_names.json')
    
    cmap = ListedColormap(colors=mpl.colormaps['tab10'].colors + mpl.colormaps['tab20c'].colors)

    feat_vecs_srtd_concat, labels_as_name_srtd, labels_as_name_srtd_no_med = get_median_vecs(
        feat_vecs=feat_vecs,
        labels_as_name=labels_as_name,
        labels_as_dose=labels_as_dose,
        dose=dose
    )

    X = get_umap(
        feat_vecs_srtd_concat,
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric
    )

    X_no_med, X_med = X[:-labels_as_name_srtd.shape[0]], X[-labels_as_name_srtd.shape[0]:]

    # Make UMAP with MoA colours 
    handles = []
    if use_moa_labels:
        for key, value in moa_col_dict_names.items():
            handles.append(mpatches.Patch(color=value, label=key))
    else:
        for l, c in enumerate(labels_as_name_srtd):
            handles.append(mpatches.Patch(color=cmap(int(l)), label=c))

    fig, ax = plt.subplots(figsize=(10,10))

    for l, c in enumerate(labels_as_name_srtd):
        idx = index([labels_as_name_srtd_no_med], [[c]])
        if use_moa_labels:
            col = moa_col_dict[c]
        else:
            col = cmap(int(l))
        ax.scatter(X_no_med[idx][:,0], X_no_med[idx][:,1], s=50, alpha=0.4, color= col, edgecolor='grey')

    x_pos = []
    y_pos = []
    text_list = []
    for i, (l, x) in enumerate(zip(labels_as_name_srtd, X_med)):
        x_pos.append(x[0])
        y_pos.append(x[1])
        text_list.append(l)
        if use_moa_labels:
            col = moa_col_dict[l]
        else:
            col = cmap(int(i))
        ax.scatter(x[0], x[1], s=150, edgecolor='black',alpha=1, label=l, color=col)
    
    ta.allocate(
        ax,x_pos,y_pos,
        text_list,
        x_scatter=x_pos, y_scatter=y_pos,
        textsize=10,
        linecolor='tab:grey'
    )
    plt.xticks([])
    plt.yticks([])
    plt.legend(handles=handles)
    plt.title(title)
    plt.savefig(save_name)
    plt.show()

def cluster_feat_vecs(dose):

    dose_dict = load_json('plotting_params/dose_dict.json')
    
    path_pattern = f'../../../DATA/E_coli_on_K_pneumoniae/AvgPoolCNN_cross_val_BF/Plate_Klebs/'
    path = glob(path_pattern)[0]
    
    feat_vecs = np.loadtxt(os.path.join(path, 'feat_vecs.txt'))
    labels = np.loadtxt(os.path.join(path, 'labels.txt'))
    with open(os.path.join(path, 'classes.txt'), 'r') as f:
        classes = json.load(f)

    label_to_name = dict(zip([i for i in range(len(classes))], [c if c not in ['DMSO'] else 'DMSO_0' for c in classes]))
    labels_as_name = np.asarray([label_to_name[l].split('_')[0] for l in labels])
    labels_as_dose = np.asarray([dose_dict[label_to_name[l].split('_')[-1]] for l in labels])

    idx_ctrl = index([labels_as_name], [['DMSO']])
    
    x = feat_vecs[idx_ctrl]    
    xc = x - np.mean(x, axis=0)
    xc = xc.T
    xcov = np.cov(xc, rowvar=True, bias=True)
    w, v = linalg.eigh(xcov) 
    diagw = np.diag(1/((w+0.005)**0.5))
    diagw = diagw.real.round(4) 
    xrot = np.dot(v, xc)
    wpca = np.dot(np.dot(diagw, v.T), xc)
    wzca = np.dot(np.dot(np.dot(v, diagw), v.T), xc)
    
    x_all = feat_vecs
    xc_all = x_all - np.mean(x_all, axis=0)
    xc_all = xc_all.T
    wzca_all = np.dot(np.dot(np.dot(v, diagw), v.T), xc_all)

    feat_vecs = wzca_all.T
    
    plot_umap(
        feat_vecs=feat_vecs,
        labels_as_name=labels_as_name,
        labels_as_dose=labels_as_dose,
        dose=dose_dict[dose],
        use_moa_labels=False,
        save_name='umap_plot_e_coli_on_klebs_cmpd.svg',
    )

    plot_umap(
        feat_vecs=feat_vecs,
        labels_as_name=labels_as_name,
        labels_as_dose=labels_as_dose,
        dose=dose_dict[dose],
        use_moa_labels=True,
        save_name='umap_plot_e_coli_on_klebs_moa.svg',
    )
    
    feat_vecs_med = []
    labels_med = []
    for l in list(set(labels_as_name)):
        idx = index([labels_as_dose, labels_as_name], [[dose_dict[dose], 0], [l]])
        feat_vecs_med.append(np.median(feat_vecs[idx], axis=0))
        labels_med.append(l)
    
    feat_vecs_med = np.vstack(feat_vecs_med)
    labels_med = np.hstack(labels_med)

    feat_vecs_med = feat_vecs_med - np.mean(feat_vecs_med, axis=0)


    sim_matrix = np.empty((len(labels_med), len(labels_med)))
    for x, fvec1 in enumerate(feat_vecs_med):
        for y, fvec2 in enumerate(feat_vecs_med):
            sim_matrix[x,y] = cosine_similarity(fvec1, fvec2)

    clustermap = sns.clustermap(
        sim_matrix,
        xticklabels=labels_med,
        yticklabels=labels_med,
        cmap='coolwarm',
        figsize=(7,7)
    )

    plt.savefig('clustermap_klebs.svg')

    Z = clustermap.dendrogram_col.linkage

    
    fig = plt.figure(figsize=(25, 2))
    ax = fig.add_subplot(1, 1, 1)
    
    matplotlib.rcParams['lines.linewidth'] = 1
    dn = dendrogram(Z, color_threshold=1,
                    labels=labels_med,
                    ax=ax)

    plt.savefig('cluster_dendrogram_klebs.svg')