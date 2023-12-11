import math
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, metrics, decomposition
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

class OverlappingCrop:
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

    def __call__(self, data):
        image = np.array(data)
        h, w = image.shape[:2]
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

            image = np.stack([np.pad(image[:,:,i], ((pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right)), 'constant', constant_values=0) for i in range(image.shape[2])], axis=2)
            # image = np.stack([np.pad(image[:,:,i], ((pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right)), 'symmetric') for i in range(image.shape[2])], axis=2)

        # Generate list of overlapping crops
        image_crops = [image[i:i+new_h, j:j+new_w, :] for i in range(0, h, self.stride)
                                                            for j in range(0, w, self.stride)]
        # Filter image crops
        image_crops = [Image.fromarray(crop.astype('uint8'), self.mode) for crop in image_crops if crop.shape[:2] == (new_h, new_w)]

        return image_crops

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
            # image = np.stack([np.pad(image[:,:,i], ((pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right)), 'symmetric') for i in range(image.shape[2])], axis=2)

        # Generate list of overlapping crops
        image_crops = [image[:, i:i+new_h, j:j+new_w] for i in range(0, h, self.stride)
                                                            for j in range(0, w, self.stride)]
        # Filter image crops
        image_crops = [crop for crop in image_crops if crop.size()[1:] == (new_h, new_w)]

        return image_crops

def plot_sample_batch(img, save_dir, save_name='sample_images.png'):
    # img = img * 0.1012 + 0.6281     # unnormalize            transforms.Normalize(mean=0.6281, std=0.1012)
    img = img * 0.5 + 0.5     # unnormalize            transforms.Normalize(mean=0.6281, std=0.1012)

    npimg = img.numpy()

    fig = plt.figure(figsize = (30, 30))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(os.path.join(save_dir, save_name))
    plt.close()

# Also add probabilities
def plot_predictions(images, labels, preds, n_images, test_acc, save_dir, classes, save_name='predicted_images.png'):
    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))

    fig = plt.figure(figsize = (25, 20))
    # plt.title('Test predictions (accuracy = {})'.format(test_acc))

    for i in range(rows*cols):

        ax = fig.add_subplot(rows, cols, i+1)

        image = images[i]
        true_label = labels[i]
        pred = preds[i]
        image = image / 2 + 0.5     # unnormalize
        image = image.permute(1, 2, 0)

        ax.imshow(image.cpu().numpy(), cmap='gray')
        ax.set_title(f'true label: {classes[true_label]}\n'\
                     f'pred label: {classes[pred]}',
                     fontsize="large")
        ax.axis('off')
    ax.text(0.5,-0.15, 'Test accuracy = {0:.3f}'.format(test_acc), size=14, ha="center",
         transform=ax.transAxes, color='red')
    fig.subplots_adjust(hspace = 0.4)
    plt.savefig(os.path.join(save_dir, save_name), dpi=150)
    plt.close()

def get_tsne(data, n_components=2):
    tsne = manifold.TSNE(n_components=n_components, random_state=0)
    tsne_data = tsne.fit_transform(data)
    return tsne_data

def get_umap(data, n_components=2):
    import umap

    umap_ = umap.UMAP(n_components=n_components, random_state=0)
    umap_data = umap_.fit_transform(data)
    return umap_data

def get_pca(data, n_components=2):
    pca = decomposition.PCA(n_components=n_components)
    pca_data = pca.fit_transform(data)
    # print('Explained variance', pca.explained_variance)
    return pca_data

def imscatter(x, y, images, labels, classes, ax=None, zoom=1):
    cmap_list = plt.get_cmap('tab20', len(classes)).colors
    if ax is None:
        ax = plt.gca()

    x, y = np.atleast_1d(x, y)
    artists = []
    for i, (x0, y0) in enumerate(zip(x, y)):
        image = images[i]
        image = image / 2 + 0.5     # unnormalize
        image = image.permute(1, 2, 0)
        im = OffsetImage(image.cpu().numpy(), zoom=zoom, cmap='gray')
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=True, bboxprops = dict(edgecolor=cmap_list[labels[i]], lw=10))
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    # ax.legend(labels=classes)

def plot_image_representations(data, images, labels, classes, save_dir, save_name='tsne_image_plot.svg'):
    fig = plt.figure(figsize = (50, 50))
    ax = fig.add_subplot(111)
    # scatter = ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='tab20', s=100)
    imscatter(data[:, 0], data[:, 1], images, labels, classes, zoom=0.7, ax=ax)
    # for i in range(data.shape[0]):
    #     ax.annotate(str(i), (data[i, 0], data[i, 1]))

    # handles, labels = scatter.legend_elements()
    # legend = ax.legend(labels=classes)
    plt.savefig(os.path.join(save_dir, save_name), dpi=150)
    plt.close()

def plot_representations(data, labels, classes, save_dir, save_name='tsne_plot.png', dropped_label=None):
    classes_from_labels = [classes[int(i)] for i in np.unique(labels)]
    print(classes_from_labels)
    print(len(classes))
    fig = plt.figure(figsize = (20, 20))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(data[:, 0], data[:, 1], s=250, c=labels, cmap='tab20')
    if dropped_label is not None:
        dropped_pnts = data[np.where(labels==dropped_label)]
        ax.scatter(dropped_pnts[:,0], dropped_pnts[:,1], s=250, c='None', edgecolor='firebrick', linewidths=4)
    # for i in range(data.shape[0]):
        # ax.annotate(str(i), (data[i, 0], data[i, 1]))
    handles, labels = scatter.legend_elements()
    print(handles)
    print(labels)
    legend = ax.legend(handles=scatter.legend_elements(num=len(classes_from_labels))[0], labels=classes_from_labels, fontsize="large")
    plt.savefig(os.path.join(save_dir, save_name), dpi=150)
    plt.close()

def plot_data_distribution(loader):
    """
    Iterate through dataloader and compute mean and variance of pixels in dataset
    """
    dataiter = iter(train_loader)
    images, __ = dataiter.next()
