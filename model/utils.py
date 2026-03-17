import math
import os
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt

class OverlappingCropMultiChannel:
    """
    Generate overlapping and evenly spaced crops.
    Returns list of PIL Images (see FiveCrop documentation)
    """
    def __init__(self, crop_size, stride, pad=True, mode='RGB'):
        assert isinstance(crop_size, (int, tuple))
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)
        else:
            assert len(crop_size) == 2
            self.crop_size = crop_size
        self.stride = stride
        self.mode = mode
        self.pad = pad

    def __call__(self, image):
        ch, h, w = image.size()
        new_h, new_w = self.crop_size

        # Pad image
        if self.pad and (h % new_h != 0 or w % new_w != 0):
            old_h = h
            old_w = w
            h = math.ceil(h / new_h) * new_h
            w = math.ceil(w / new_w) * new_w

            pad_h = h - old_h
            pad_w = w - old_w
            pad_h_top = math.floor(pad_h / 2)
            pad_h_bottom = math.ceil(pad_h / 2)
            pad_w_left = math.floor(pad_w / 2)
            pad_w_right = math.ceil(pad_w / 2)

            image = np.stack([np.pad(image[i,:,:], ((pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right)), 'constant', constant_values=0) for i in range(ch)], axis=2)

        # Generate list of overlapping crops
        image_crops = [image[:, i:i+new_h, j:j+new_w] for i in range(0, h, self.stride)
                                                            for j in range(0, w, self.stride)]
        # Filter image crops
        image_crops = [crop for crop in image_crops if crop.size()[1:] == (new_h, new_w)]

        return image_crops

def plot_sample_batch(img, save_dir, save_name='sample_images.png'):

    img = img * 0.5 + 0.5

    npimg = img.numpy()

    fig = plt.figure(figsize = (30, 30))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(os.path.join(save_dir, save_name))
    plt.close()

def make_dir(dir):
    """Create directories including subdirectories"""
    dir_lst = dir.split('/')
    for idx in range(1, len(dir_lst) + 1):
        if not os.path.exists(os.path.join(*dir_lst[:idx])):
            os.mkdir(os.path.join(*dir_lst[:idx]))

def convert_to_list(x):
    return [x_i.unsqueeze(0) for x_i in x]

def intersect_dicts(class_merge_dict, moa_dict):
    intrsct_dict = dict()
    for k, v in class_merge_dict.items():
        if v in moa_dict.keys():
            intrsct_dict[k] = moa_dict[v]

    return {**intrsct_dict, **moa_dict}

def get_class_weights(root_train):
    class_dirs = []
    for dir in root_train:
        class_dirs += os.listdir(dir)
    class_dirs = [x for x in class_dirs if x not in dropped_classes]
    class_dirs = [class_merge_dict[x] if x in class_merge_dict.keys() else x for x in class_dirs]
    class_weights = [1/class_dirs.count(x) for x in sorted(list(set(class_dirs)))]

    return class_weights

def make_model_directories(save_dir,
                           test_dir=None
                           test_dir_ext=None):
    if test_dir:
        if test_dir_ext is None:
            test_dir_ext = datetime.datetime.now().strftime("%y%m%d_%H%M")
        save_dir_ = os.path.join(save_dir, f"{test_dir}_{test_dir_ext}")
        make_dir(save_dir_)
    else:
        if not os.path.exists(save_dir):
            make_dir(save_dir)
        if not os.path.exists(os.path.join(save_dir, 'ckpts')):
            os.mkdir(os.path.join(save_dir, 'ckpts'))

def parse_train_val_test_dir(data_dir,
                             train_dir,
                             test_dir,
                             val_dir,
                             dose):
    all_dir = os.listdir(data_dir)

    if dose:
        all_dir = [os.path.join(d, dose) for d in all_dir]

    if val_dir[0] == 'None':
        val_dir = None
    elif len(val_dir) == 0:
        val_dir = [random.choice([dir for dir in all_dir if dir not in test_dir])]

    if len(train_dir) == 0:
        if val_dir:
            train_dir = [dir for dir in all_dir if dir not in test_dir + val_dir]
        else:
            train_dir = [dir for dir in all_dir if dir not in test_dir]

    return [train_dir, val_dir, test_dir]

def load_ckpt(ckpt, save_dir):
    if ckpt > 0:
        ckpt_path = glob.glob(os.path.join(save_dir, 'ckpts', '*_' + str(ckpt) + '_*.tar'))[0]
    elif ckpt == -1:
        ckpt_paths = glob.glob(os.path.join(save_dir, 'ckpts', '*.tar'))
        ckpt_path = sorted(ckpt_paths, key=lambda s: os.path.basename(s).split('_')[3])[0]

    return torch.load(ckpt_path)
