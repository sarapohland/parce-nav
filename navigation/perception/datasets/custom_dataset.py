import os
import tqdm
import pickle
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

from navigation.perception.utils.segment import mask_images


class CustomDataset(Dataset):
    def __init__(self, data_dir, key):
        # Load saved data
        print('Loading {} dataset from {}...'.format(key, data_dir))
        file = os.path.join(data_dir, 'dataset.npz')
        dataset = np.load(open(file, 'rb'))
        self.data = dataset[key + '_data']
        self.labels = dataset[key + '_labels']

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        data = torch.from_numpy(self.data[idx,:,:,:]).float()
        labels = torch.from_numpy(self.labels[idx,:])
        return data, labels

class SegmentedDataset(Dataset):
    def __init__(self, data_dir, key, params):
        # Load saved data
        print('Loading {} dataset from {}...'.format(key, data_dir))
        file = os.path.join(data_dir, 'dataset.npz')
        dataset = np.load(open(file, 'rb'))
        orig_data = dataset[key + '_data']

        # Segment and mask images
        self.data, self.labels = mask_images(orig_data, params)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        data = torch.from_numpy(self.data[idx,:,:,:]).float()
        labels = torch.from_numpy(self.labels[idx,:,:,:]).float()
        return data, labels
    