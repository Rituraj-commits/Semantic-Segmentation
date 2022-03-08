import numpy as np
import os
import cv2
from torch.utils.data import Dataset

class GTAVDataset(Dataset):
    def __init__(self, mode, classes=1, dataset_path="./"):
        self.mode = mode
        self.classes = classes
        self.dataset_path = dataset_path

    def __len__(self):
        return len(os.listdir(self.dataset_path + self.mode + "/images/"))

    def __getitem__(self, idx):
        filename = os.listdir(self.dataset_path + self.mode + "/images/")
        img = cv2.imread(self.dataset_path + self.mode + "/images/" + filename[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = np.transpose(img, (2, 0, 1))
        img = img / 255.0
        img = img.astype(np.float32)
        label = cv2.imread(
            self.dataset_path + self.mode + "/labels/" + filename[idx]
        )
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        label = cv2.resize(label, (224, 224))    
        label = label / 255.0
        return img, label

