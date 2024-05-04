import string
import numpy as np

import torch
import torchvision
from torch.utils.data import ConcatDataset

from navigation.perception.datasets.custom_dataset import CustomDataset, SegmentedDataset

def get_class_names(data):
    if data == 'lunar-nav':
        return ['sunny smooth', 'gray smooth', 'sunny bumpy', 'gray bumpy', 'light crater',
                'dark crater', 'edge of crater', 'hill'] + ['structures']
    else:
        raise NotImplementedError('The {} dataset is not currently available.'.format(data))

def get_num_classes(data):
    if data == 'lunar-nav':
        return 8
    else:
        raise NotImplementedError('The {} dataset is not currently available.'.format(data))

def make_weights_for_balanced_classes(labels):
    total_lbls = len(labels)
    unique_lbls = np.unique(labels)
    weights = np.zeros(len(labels))
    for lbl in unique_lbls:
        count = len(np.where(labels.flatten() == lbl)[0])
        weights[labels.flatten() == lbl] = total_lbls / count                           
    return weights 

def setup_loader(data, batch_size=None, train=False, val=False, test=False, ood=False, params=None):
    
    if data == 'lunar-nav':
        batch_size = 64 if batch_size is None else batch_size
        if train:
            if params is None:
                train_dataset = CustomDataset('./data/Lunar-Nav/', 'train')
                weights = make_weights_for_balanced_classes(train_dataset.labels)
                sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
                loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                                        sampler=sampler)
            else:
                train_dataset = SegmentedDataset('./data/Lunar-Nav/', 'train', params)
                loader = torch.utils.data.DataLoader(train_dataset,
                            batch_size=batch_size, shuffle=True)
            
        elif val:
            if params is None:
                val_dataset = CustomDataset('./data/Lunar-Nav/', 'val')
            else:
                val_dataset = SegmentedDataset('./data/Lunar-Nav/', 'val', params)
            loader = torch.utils.data.DataLoader(val_dataset,
                        batch_size=batch_size, shuffle=True)
            
        elif test:
            if params is None:
                test_dataset = CustomDataset('./data/Lunar-Nav/', 'test')
            else:
                test_dataset = SegmentedDataset('./data/Lunar-Nav/', 'test', params)
            loader = torch.utils.data.DataLoader(test_dataset,
                        batch_size=batch_size, shuffle=False)
            
        elif ood:
            if params is None:
                ood_dataset = CustomDataset('./data/Lunar-Nav/', 'ood')
            else:
                ood_dataset = SegmentedDataset('./data/Lunar-Nav/', 'ood', params)
            loader = torch.utils.data.DataLoader(ood_dataset,
                        batch_size=batch_size, shuffle=False)
    
    else:
        raise NotImplementedError('The {} dataset is not currently available.'.format(data))
    
    return loader