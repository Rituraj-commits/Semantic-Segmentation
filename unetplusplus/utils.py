import torch.nn as nn
import numpy as np
import scipy.io

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice

def iou_score(predictions, labels):
    predictions = nn.Softmax(dim=1)(predictions)
    pred = predictions.data.cpu().numpy()
    ious = []
    pred[pred <= 0.5] = 0
    pred[pred > 0.5] = 1
    testlabel = labels.numpy()[0][0].astype(bool)
    pred = pred.astype(bool)
    # Compute IOU
    overlap = testlabel * pred
    union = testlabel + pred
    iou = overlap.sum() / float(union.sum())
    ious.append(iou)
    return ious

def dice_score(predictions,labels):
    predictions = nn.Softmax(dim=1)(predictions)
    pred = predictions.data.cpu().numpy()
    dices = []
    pred[pred <= 0.5] = 0
    pred[pred > 0.5] = 1
    target = labels.numpy()[0][0].astype(bool)
    numerator = 2 * np.sum(pred * target)
    denominator = np.sum(pred + target)
    dice = (numerator + 1) / (denominator + 1)
    dices.append(dice)
    return dices


def rgb_to_onehot(rgb_image, colormap):
    '''Function to one hot encode RGB mask labels
        Inputs: 
            rgb_image - image matrix (eg. 256 x 256 x 3 dimension numpy ndarray)
            colormap - dictionary of color to label id
        Output: One hot encoded image of dimensions (height x width x num_classes) where num_classes = len(colormap)
    '''
    num_classes = len(colormap)
    shape = rgb_image.shape[:2]+(num_classes,)
    encoded_image = np.zeros( shape, dtype=np.int8 )
    rgb_image = np.expand_dims(rgb_image, axis=0)
    for i, cls in enumerate(colormap):
        encoded_image[:,:,i] = np.all(rgb_image.reshape( (-1,3) ) == colormap[i], axis=1).reshape(shape[:2])
    encoded_image = encoded_image.transpose(2,0,1)
    return encoded_image


def onehot_to_rgb(onehot, colormap):
    '''Function to decode encoded mask labels
        Inputs: 
            onehot - one hot encoded image matrix (height x width x num_classes)
            colormap - dictionary of color to label id
        Output: Decoded RGB image (height x width x 3) 
    '''
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros( onehot.shape[:2]+(3,) )
    for k in colormap.keys():
        output[single_layer==k] = colormap[k]
    return np.uint8(output)

def one_hot_encoded():

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

    return code2id,id2code,name2id,id2name