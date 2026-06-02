import torch
import tifffile
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

from gradcam import GradCAM

class_dict = {'Avibactam_0.125xIC50': 0, 'Avibactam_0.25xIC50': 1, 'Avibactam_0.5xIC50': 2, 'Avibactam_1xIC50': 3, 'Aztreonam_0.125xIC50': 4, 'Aztreonam_0.25xIC50': 5, 'Aztreonam_0.5xIC50': 6, 'Aztreonam_1xIC50': 7, 'Cefepime_0.125xIC50': 8, 'Cefepime_0.25xIC50': 9, 'Cefepime_0.5xIC50': 10, 'Cefepime_1xIC50': 11, 'Cefsulodin_0.125xIC50': 12, 'Cefsulodin_0.25xIC50': 13, 'Cefsulodin_0.5xIC50': 14, 'Cefsulodin_1xIC50': 15, 'Ceftriaxone_0.125xIC50': 16, 'Ceftriaxone_0.25xIC50': 17, 'Ceftriaxone_0.5xIC50': 18, 'Ceftriaxone_1xIC50': 19, 'Chloramphenicol_0.125xIC50': 20, 'Chloramphenicol_0.25xIC50': 21, 'Chloramphenicol_0.5xIC50': 22, 'Chloramphenicol_1xIC50': 23, 'Ciprofloxacin_0.125xIC50': 24, 'Ciprofloxacin_0.25xIC50': 25, 'Ciprofloxacin_0.5xIC50': 26, 'Ciprofloxacin_1xIC50': 27, 'Clarithromycin_0.125xIC50': 28, 'Clarithromycin_0.25xIC50': 29, 'Clarithromycin_0.5xIC50': 30, 'Clarithromycin_1xIC50': 31, 'Clavulanate_0.125xIC50': 32, 'Clavulanate_0.25xIC50': 33, 'Clavulanate_0.5xIC50': 34, 'Clavulanate_1xIC50': 35, 'Colistin_0.125xIC50': 36, 'Colistin_0.25xIC50': 37, 'Colistin_0.5xIC50': 38, 'Colistin_1xIC50': 39, 'DMSO_0.125xIC50': 40, 'DMSO_0.25xIC50': 40, 'DMSO_0.5xIC50': 40, 'DMSO_1xIC50': 40, 'Doxycycline_0.125xIC50': 41, 'Doxycycline_0.25xIC50': 42, 'Doxycycline_0.5xIC50': 43, 'Doxycycline_1xIC50': 44, 'Kanamycin_0.125xIC50': 45, 'Kanamycin_0.25xIC50': 46, 'Kanamycin_0.5xIC50': 47, 'Kanamycin_1xIC50': 48, 'Levofloxacin_0.125xIC50': 49, 'Levofloxacin_0.25xIC50': 50, 'Levofloxacin_0.5xIC50': 51, 'Levofloxacin_1xIC50': 52, 'Mecillinam_0.125xIC50': 53, 'Mecillinam_0.25xIC50': 54, 'Mecillinam_0.5xIC50': 55, 'Mecillinam_1xIC50': 56, 'Meropenem_0.125xIC50': 57, 'Meropenem_0.25xIC50': 58, 'Meropenem_0.5xIC50': 59, 'Meropenem_1xIC50': 60, 'Norfloxacin_0.125xIC50': 61, 'Norfloxacin_0.25xIC50': 62, 'Norfloxacin_0.5xIC50': 63, 'Norfloxacin_1xIC50': 64, 'PenicillinG_0.125xIC50': 65, 'PenicillinG_0.25xIC50': 66, 'PenicillinG_0.5xIC50': 67, 'PenicillinG_1xIC50': 68, 'PolymyxinB_0.125xIC50': 69, 'PolymyxinB_0.25xIC50': 70, 'PolymyxinB_0.5xIC50': 71, 'PolymyxinB_1xIC50': 72, 'Relebactam_0.125xIC50': 73, 'Relebactam_0.25xIC50': 74, 'Relebactam_0.5xIC50': 75, 'Relebactam_1xIC50': 76, 'Rifampicin_0.125xIC50': 77, 'Rifampicin_0.25xIC50': 78, 'Rifampicin_0.5xIC50': 79, 'Rifampicin_1xIC50': 80, 'Sulbactam_0.125xIC50': 81, 'Sulbactam_0.25xIC50': 82, 'Sulbactam_0.5xIC50': 83, 'Sulbactam_1xIC50': 84, 'Trimethoprim_0.125xIC50': 85, 'Trimethoprim_0.25xIC50': 86, 'Trimethoprim_0.5xIC50': 87, 'Trimethoprim_1xIC50': 88}


def load_tensor_from_tiff(path):
    img = tifffile.imread(path)
    
    return torch.FloatTensor(img / (2 ** 8 - 1))

def make_composite(img_1, img_2):
    if isinstance(img_1, torch.FloatTensor):
        img_1 = Image.fromarray(np.array(img_1))
    if isinstance(img_2, torch.FloatTensor):
        img_2 = Image.fromarray(np.array(img_2))

    img_2_crop = img_2.crop((img_2.size[0] // 2, 0, img_2.size[0], img_2.size[1]))
    img_1.paste(img_2_crop, (img_1.size[0] // 2, 0))
    
    return torch.FloatTensor(np.array(img_1))

def plot_gradcam(
    model,
    input_tensor,
    cmpd_name
):

    gradcam = GradCAM(model=model)
    gradcam.hook_target_layer(target_layer=3)
    
    gradcam_map = gradcam(input_tensor, target_class_idx=class_dict[cmpd_name])
    
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(10,10))
    
    min_gradcam_map = gradcam_map.min().item()
    max_gradcam_map = gradcam_map.max().item()
    
    for i in range(9):
    
        r = i // 3
        c = i % 3
    
        ax[r,c].imshow(input_tensor[0,i,0].cpu().numpy(), cmap='gray')
    
        gradcam_resized = cv2.resize(gradcam_map[i].cpu().numpy(), (256, 256), interpolation=cv2.INTER_CUBIC)
        ax[r,c].imshow(gradcam_resized, cmap='jet', alpha=0.2, vmin=min_gradcam_map, vmax=max_gradcam_map)
        ax[r,c].axis('off')
    
    fig.tight_layout()
    fig.suptitle(f'{cmpd_name.split("_")[0]} ({cmpd_name.split("_")[1]})', y=1.015, fontsize=20)
    plt.show()

def plot_gradcam_counterfactual(
    model,
    input_tensor,
    cmpd_name_1='Doxycycline_1xIC50',
    cmpd_name_2='Mecillinam_1xIC50'
    
):
    gradcam = GradCAM(model=model)
    gradcam.hook_target_layer(target_layer=3)
    
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(6,9))
    
    tgt_cmpds = []
    gradcam_map_dict = dict()
    
    for col_idx, tgt_cmpd in enumerate([cmpd_name_1, cmpd_name_2]):
        gradcam_map = gradcam(input_tensor, target_class_idx=class_dict[tgt_cmpd])
        
        min_gradcam_map = gradcam_map.min().item()
        max_gradcam_map = gradcam_map.max().item()
    
        gradcam_map_dict[tgt_cmpd] = gradcam_map
        
        for row_idx, i in enumerate([1,4,7]):
        
            ax[row_idx, col_idx].imshow(input_tensor[0,i,0].cpu().numpy(), cmap='gray')
        
            gradcam_resized = cv2.resize(gradcam_map[i].cpu().numpy(), (256, 256), interpolation=cv2.INTER_CUBIC)
            ax[row_idx, col_idx].imshow(gradcam_resized, cmap='jet', alpha=0.2, vmin=min_gradcam_map, vmax=max_gradcam_map)
            ax[row_idx, col_idx].axis('off')
    
        ax[0,col_idx].set_title(f'Target: {tgt_cmpd}')
    
        tgt_cmpds.append(tgt_cmpd)
        
    fig.tight_layout()
    plt.show()

def plot_relative_gradcam_activation(
    model,
    input_tensor,
    cmpd_name_left='Doxycycline_1xIC50',
    cmpd_name_right='Mecillinam_1xIC50',
    
):

    gradcam = GradCAM(model=model)
    gradcam.hook_target_layer(target_layer=3)
        
    tgt_cmpds = []
    gradcam_map_dict = dict()
    
    for col_idx, tgt_cmpd in enumerate([cmpd_name_left, cmpd_name_right]):
        gradcam_map = gradcam(input_tensor, target_class_idx=class_dict[tgt_cmpd])
        
        min_gradcam_map = gradcam_map.min().item()
        max_gradcam_map = gradcam_map.max().item()
    
        gradcam_map_dict[tgt_cmpd] = gradcam_map
        
    for crop in [1, 4, 7]:
        left_left = gradcam_map_dict[cmpd_name_left][crop].cpu().numpy()[:,:16].sum() / (gradcam_map_dict[cmpd_name_left][crop].cpu().numpy()[:,:16].sum() + gradcam_map_dict[cmpd_name_right][crop].cpu().numpy()[:,:16].sum())
        left_right = gradcam_map_dict[cmpd_name_left][crop].cpu().numpy()[:,16:].sum() / (gradcam_map_dict[cmpd_name_left][crop].cpu().numpy()[:,16:].sum() + gradcam_map_dict[cmpd_name_right][crop].cpu().numpy()[:,16:].sum())
        
        right_left = gradcam_map_dict[cmpd_name_right][crop].cpu().numpy()[:,:16].sum() / (gradcam_map_dict[cmpd_name_left][crop].cpu().numpy()[:,:16].sum() + gradcam_map_dict[cmpd_name_right][crop].cpu().numpy()[:,:16].sum())
        right_right = gradcam_map_dict[cmpd_name_right][crop].cpu().numpy()[:,16:].sum() / (gradcam_map_dict[cmpd_name_left][crop].cpu().numpy()[:,16:].sum() + gradcam_map_dict[cmpd_name_right][crop].cpu().numpy()[:,16:].sum())
    
        plt.figure(figsize=(8,3))
        
        plt.bar([0 , 1.25, 2.75, 4],[left_left, left_right, right_left, right_right], color=['tab:blue', 'tab:orange', 'tab:blue', 'tab:orange'])
        plt.xticks([0 , 1.25, 2.75, 4], [cmpd_name_left.split('_')[0], cmpd_name_right.split('_')[0], cmpd_name_left.split('_')[0], cmpd_name_right.split('_')[0]], rotation=90)
        plt.ylabel('Relative Grad-CAM activation')
        plt.show()