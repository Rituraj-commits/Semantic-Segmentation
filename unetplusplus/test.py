import scipy.io
import numpy as np
from utils import *
from loader import *
import torch

from segmentation_models_pytorch import UnetPlusPlus


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

def test_generator():

    dataset = GTAVDataset(mode="train",classes=19,dataset_path='/media/ri2raj/External HDD/gta5_new/')
    batch = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=2)
    x,y = next(iter(batch))
   
    mask_encoded = [rgb_to_onehot(y[X,:,:,:], id2code) for X in range(y.shape[0])]
    mask_encoded = np.array(mask_encoded)

    return x, mask_encoded

model = UnetPlusPlus(encoder_name='efficientnet-b1', encoder_weights=None, classes=35, activation=None)
#print(model)


# Example of target with class indices
loss = nn.CrossEntropyLoss()

# Example of target with class probabilities
input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5).softmax(dim=1)
output = loss(input, target)
output.backward()




