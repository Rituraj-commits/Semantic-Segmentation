import torch.nn.functional as F
import numpy as np
import scipy.io
import torch

import matplotlib.pyplot as plt

def onehot_to_rgb(onehot, colormap):
    """Function to decode encoded mask labels
    Inputs:
        onehot - one hot encoded image matrix (height x width x num_classes)
        colormap - dictionary of color to label id
    Output: Decoded RGB image (height x width x 3)
    """
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros(onehot.shape[:2] + (3,))
    for k in colormap.keys():
        output[single_layer == k] = colormap[k]
    return output


def rgb_to_onehot(rgb, colormap):
    """Function to one hot encode RGB mask labels
    Inputs:
        rgb - RGB mask (height x width x 3)
        colormap - dictionary of color to label id
    Output: Encoded one hot vector (height x width x num_classes)
    """

    rgb = rgb.numpy()
    mask = np.zeros((rgb.shape[0], rgb.shape[1]))
    for k, v in colormap.items():
        mask[np.all(rgb == v, axis=2)] = k
    mask = F.one_hot(torch.from_numpy(mask).long(), num_classes=len(colormap)).numpy()
    return mask


def one_hot_encoded():
    """Function to create colormaps for one hot encoding"""

    mapping = scipy.io.loadmat("/media/ri2raj/External HDD/gta5_new/train/mapping.mat")

    label_names = []
    label_codes = []

    for i, name in enumerate(mapping.get("classes")):
        for j, value in enumerate(name):
            label_names = np.append(label_names, value)

    x = list(mapping.get("cityscapesMap"))

    for i in x:
        label_codes.append(i)

    label_codes = tuple(map(tuple, label_codes))

    code2id = {v: k for k, v in enumerate(label_codes)}
    id2code = {k: v for k, v in enumerate(label_codes)}

    name2id = {v: k for k, v in enumerate(label_names)}
    id2name = {k: v for k, v in enumerate(label_names)}

    return code2id, id2code, name2id, id2name

def plot_net_predictions(imgs, true_masks, masks_pred, batch_size, colormap):
    
    fig, ax = plt.subplots(3, batch_size, figsize=(20, 15))
    
    for i in range(batch_size):
        
        img  = np.transpose(imgs[i].squeeze().cpu().detach().numpy(), (1,2,0))
        mask_pred = masks_pred[i].cpu().detach().numpy()
        mask_true = true_masks[i].cpu().detach().numpy()
    
        ax[0,i].imshow(img)
        ax[1,i].imshow(onehot_to_rgb(mask_pred,colormap))
        ax[1,i].set_title('Predicted')
        ax[2,i].imshow(onehot_to_rgb(mask_true,colormap))
        ax[2,i].set_title('Ground truth')
        
    return fig