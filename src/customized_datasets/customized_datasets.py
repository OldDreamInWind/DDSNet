import h5py
import pandas as pd
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

class PCamDataset(Dataset):
    def __init__(self, datafile, labelsfile, transform=None):
        self.data = h5py.File(datafile,'r')
        self.img_labels = h5py.File(labelsfile,'r')
        self.transform = transform

    def __len__(self):
        return len(self.img_labels['y'])

    def __getitem__(self, idx):
        # image = torch.tensor(self.data['x'][idx]).permute(2, 0, 1).to(torch.float)
        image = np.array(self.data['x'][idx]).astype(np.float32)
        label = self.img_labels['y'][idx][0][0][0]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class SkinDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, split='train', test_size=0.2, random_state=42):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        image_files = self.data_frame.iloc[1:,1].values
        labels = self.data_frame.iloc[1:,2].values
        self.label_dict = {'akiec':0, 'bcc':1, 'bkl':2, 'df':3, 'mel':4, 'nv':5, 'vasc':6}

        label_indices = [self.label_dict[label] for label in labels]

        train_files, val_files, train_labels, val_labels = train_test_split(
            image_files, label_indices, test_size=test_size, random_state=random_state, stratify=label_indices
        )

        if split=='train':
            self.image_files = train_files
            self.labels = train_labels
        elif split=='val':
            self.image_files = val_files
            self.labels = val_labels
        else:
            raise ValueError("split should be train or val")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.image_files[idx]) + '.jpg'
        image = Image.open(img_name)

        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)

        return image, label