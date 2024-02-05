from random import shuffle
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as model
import random
import numpy as np
from exp_mode import EXP_MODES
from torch.utils.data import Subset
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
import random
import os

valid_rate = 0.2
dataset_load_func = {
    'mnist': torchvision.datasets.MNIST,
    'fmnist':torchvision.datasets.FashionMNIST,
    'cifar10': torchvision.datasets.CIFAR10,
    'cifar100': torchvision.datasets.CIFAR100,
}

transform_original = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

transform_aug = transforms.Compose([
            transforms.RandomChoice([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(180),
                transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                ]),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

class CustomCIFAR10Dataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset, original_transform, augmented_transform, augmentation_rate):
        self.original_dataset = original_dataset
        self.original_transform = original_transform
        self.augmented_transform = augmented_transform
        self.augmentation_rate = augmentation_rate

    def __getitem__(self, index):
        if torch.rand(1).item() < self.augmentation_rate:
            return self.augmented_transform(self.original_dataset[index][0]), self.original_dataset[index][1]
        else:
            return self.original_transform(self.original_dataset[index][0]), self.original_dataset[index][1]

    def __len__(self):
        return len(self.original_dataset)
    

def prepare_aug_data(dataset_name):
    data_dir = "./data/" + dataset_name
    
    aug_dataset = dataset_load_func[dataset_name](root=data_dir, 
                                              train=True, 
                                              download=True,
                                              transform=transform_aug
                                              )
    aug, _ = train_test_split(aug_dataset, train_size=(1-valid_rate), shuffle=False)
    
    return aug

def prepare_train_aug_data_per_epoch(dataset_name, train_dataset, aug_val, bs, shuffleFlag, mode):
    aug_rate = aug_val/100
    
    aug_data = prepare_aug_data(dataset_name)

    ''' Fo me, I want to use below code insted of calling "prepare_aug_data"
    num_samples = len(aug_data)
    num_subset = int(num_samples * 0.8)
    aug_data = Subset(aug_data, range(num_subset))
    '''

    if aug_rate<1:
        if mode == EXP_MODES.ORIG_PLUS_DYNAMIC_AUG_1X:
            train_dataset, _ = train_test_split(train_dataset, train_size=1-aug_rate, shuffle=shuffleFlag)
        aug, _ = train_test_split(aug_data, train_size=aug_rate, shuffle=shuffleFlag)
        my_train_dataset = train_dataset + aug
    else:
        aug = list(aug_data)        
        if mode == EXP_MODES.ORIG_PLUS_DYNAMIC_AUG_1X:
            my_train_dataset = aug
        elif mode == EXP_MODES.ORIG_PLUS_DYNAMIC_AUG_2X:
            my_train_dataset = train_dataset + aug
        
    train_loader = torch.utils.data.DataLoader(my_train_dataset, batch_size=bs, shuffle=True, num_workers=4)   
    
    return train_loader

def prepare_train_aug_data_per_epoch2(train_dataset, aug_data, aug_val, bs, shuffleFlag, mode):
    aug_rate = aug_val/100
   
    if aug_rate<1:
        if mode == EXP_MODES.ORIG_PLUS_DYNAMIC_AUG_1X:
            train_dataset, _ = train_test_split(train_dataset, train_size=1-aug_rate, shuffle=shuffleFlag)
            aug, _ = train_test_split(aug_data, train_size=aug_rate, shuffle=shuffleFlag)
        elif mode == EXP_MODES.ORIG_PLUS_DYNAMIC_AUG_2X:
            aug, _ = train_test_split(aug_data, train_size=aug_rate, shuffle=True)
        my_train_dataset = train_dataset + aug
    else:
        aug = list(aug_data)        
        if mode == EXP_MODES.ORIG_PLUS_DYNAMIC_AUG_1X:
            my_train_dataset = aug
        elif mode == EXP_MODES.ORIG_PLUS_DYNAMIC_AUG_2X:
            my_train_dataset = train_dataset + aug
        
    train_loader = torch.utils.data.DataLoader(my_train_dataset, batch_size=bs, shuffle=True, num_workers=4)   
    
    return train_loader



def load_original_and_aug_Data(dataset_name = None):
    data_dir = './data/{}'.format(dataset_name)
    data_download_Flag = False
    if not os.path.exists(data_dir):
        data_download_Flag = True

    original_train_dataset = dataset_load_func[dataset_name](root=data_dir, 
                                              train=True, 
                                              download=data_download_Flag,
                                              transform=transform_original
                                              )
    
    train_orig_dataset, valid_orig_dataset = train_test_split(original_train_dataset, test_size=valid_rate, shuffle=False)

    test_orig_dataset = dataset_load_func[dataset_name](root=data_dir, 
                                                   train=False, 
                                                   download=True,
                                                   transform=transform_original)
    
    # Load CIFAR-10 dataset
    aug_train_dataset = dataset_load_func[dataset_name](root=data_dir, train=True, download=True, transform=transform_aug)

    # Split into training and validation sets
    train_size = int((1-valid_rate) * len(aug_train_dataset))

    train_aug_dataset = torch.utils.data.Subset(aug_train_dataset, range(train_size))

    return train_orig_dataset, valid_orig_dataset, test_orig_dataset, train_aug_dataset


def load_original_Data(dataset_name = None):

    data_dir = './data/{}'.format(dataset_name)
    data_download_Flag = False
    if not os.path.exists(data_dir):
        data_download_Flag = True

    dataset = dataset_load_func[dataset_name](root=data_dir, 
                                              train=True, 
                                              download=data_download_Flag,
                                              transform=transform_original
                                              )
    
    train_dataset, valid_dataset = train_test_split(dataset, test_size=0.2, shuffle=False)

    test_dataset = dataset_load_func[dataset_name](root=data_dir, 
                                                   train=False, 
                                                   download=True,
                                                   transform=transform_original)

    return train_dataset, valid_dataset, test_dataset

def getCustomDataset(train_dataset, augmentation_rate):
    custom_train_dataset = CustomCIFAR10Dataset(train_dataset, transform_original, transform_aug, augmentation_rate)
    return custom_train_dataset

    
def load_aug_Data(dataset_name = None):

    data_dir = './data/{}'.format(dataset_name)
    data_download_Flag = False
    if not os.path.exists(data_dir):
        data_download_Flag = True
    
    # Load CIFAR-10 dataset
    full_dataset = dataset_load_func[dataset_name](root='./data', train=True, download=True, transform=transform_original)

    # Split the dataset into training and validation sets
    #train_dataset, valid_dataset = train_test_split(full_dataset, test_size=0.2, shuffle=False)

    # Split into training and validation sets
    train_size = int((1-valid_rate) * len(full_dataset))
    valid_size = len(full_dataset) - train_size

    train_dataset = torch.utils.data.Subset(full_dataset, range(train_size))
    valid_dataset = torch.utils.data.Subset(full_dataset, range(train_size, len(full_dataset)))

    test_dataset = dataset_load_func[dataset_name](root=data_dir, 
                                                   train=False, 
                                                   download=True,
                                                   transform=transform_original)
    
    return train_dataset, valid_dataset, test_dataset