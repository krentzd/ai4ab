from skimage import filters
from PIL import Image
import glob
import os
import numpy as np
from tqdm import tqdm
import argparse
import shutil

def estimate_cell_density(im):
    thresh = filters.threshold_otsu(im)
    fg = im > thresh
    bg = im < thresh
    return np.array(fg.sum() / (bg.sum() + fg.sum())), fg, bg

def make_dir(dir):
    """Create directories including subdirectories"""
    dir_lst = dir.split('/')
    for idx in range(1, len(dir_lst) + 1):
        if not os.path.exists(os.path.join(*dir_lst[:idx])):
            os.mkdir(os.path.join(*dir_lst[:idx]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plate', required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--low_density_dir', required=True)
    parser.add_argument('--medium_density_dir', required=True)
    parser.add_argument('--high_density_dir', required=True)

    args = parser.parse_args()


    cmpd_dir_paths = glob.glob(os.path.join(args.data_dir, f'Plate_{args.plate}', '*'))

    cmpd_name_list = []
    cell_instance_dict = dict()

    for cmpd_dir in tqdm(cmpd_dir_paths):
        cmpd_name = os.path.basename(cmpd_dir)
        cmpd_name_list.append(cmpd_name)
        img_paths = glob.glob(os.path.join(cmpd_dir, '*.png'))
        instance_list = []
        for img_path in img_paths:
            img = np.array(Image.open(img_path))
            cell_dnst, __, __ = estimate_cell_density(img[...,1])

            img_base_path = os.path.basename(img_path)

            instance_list.append((img_base_path, cell_dnst))

        cell_instance_dict[cmpd_name] = instance_list


    source_dir = os.path.join(args.data_dir, f'Plate_{args.plate}')

    make_dir(args.low_density_dir)
    make_dir(args.high_density_dir)
    make_dir(args.medium_density_dir)

    for key in cell_instance_dict.keys():
        cell_density = np.hstack([i[1] for i in cell_instance_dict[key]])
        p_low = np.percentile(cell_density, 30)
        p_high = np.percentile(cell_density, 70)

        target_low_density_dir = os.path.join(args.low_density_dir, key)
        target_high_density_dir = os.path.join(args.high_density_dir, key)
        target_medium_density_dir = os.path.join(args.medium_density_dir, key)

        make_dir(target_low_density_dir)
        make_dir(target_high_density_dir)
        make_dir(target_medium_density_dir)

        low_count = 0
        high_count = 0
        med_count = 0

        for item in cell_instance_dict[key]:
            if item[1] <= p_low:
                low_count += 1

                source_path = os.path.join(source_dir, key, item[0])
                target_path = os.path.join(target_low_density_dir, item[0])

                shutil.copy(source_path, target_path)

            elif item[1] >= p_high:
                high_count += 1

                source_path = os.path.join(source_dir, key, item[0])
                target_path = os.path.join(target_high_density_dir, item[0])

                shutil.copy(source_path, target_path)

            else:
                med_count += 1

                source_path = os.path.join(source_dir, key, item[0])
                target_path = os.path.join(target_medium_density_dir, item[0])

                shutil.copy(source_path, target_path)

        print(f'{key}: n_low is {low_count}, n_med is {med_count} and n_high is {high_count}')
