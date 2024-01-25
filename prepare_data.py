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

from sklearn.model_selection import train_test_split
import random
import os

dataset_load_func = {
    'mnist': torchvision.datasets.MNIST,
    'fmnist':torchvision.datasets.FashionMNIST,
    'cifar10': torchvision.datasets.CIFAR10,
    'cifar100': torchvision.datasets.CIFAR100,
}

def prepare_aug_data(dataset_name):
    transform_t = transforms.Compose([
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
    
    data_dir = "./data/" + dataset_name
    
    aug_dataset = dataset_load_func[dataset_name](root=data_dir, 
                                              train=True, 
                                              download=True,
                                              transform=transform_t
                                              )
    
    return aug_dataset

def prepare_train_aug_data_per_epoch(dataset_name, train_dataset, aug_val, bs, shuffleFlag, mode):
    aug_rate = aug_val/100
    
    aug_data = prepare_aug_data(dataset_name)
    
    if aug_rate<1:
        if mode == EXP_MODES.NEW_AUG:
            train_dataset, _ = train_test_split(train_dataset, train_size=1-aug_rate, shuffle=shuffleFlag)
            aug, _ = train_test_split(aug_data, train_size=aug_rate, shuffle=shuffleFlag)
        else:
            aug, _ = train_test_split(aug_data, train_size=aug_rate, shuffle=shuffleFlag)
    else:
        aug = list(aug_data)
        
    if mode == EXP_MODES.NEW_AUG and aug_rate>=1:
        aug_train_dataset = aug
    else:
        aug_train_dataset = train_dataset + aug
        
    train_loader = torch.utils.data.DataLoader(aug_train_dataset, batch_size=bs, shuffle=True, num_workers=4)   
    
    return train_loader


def prepare_train_data_per_epoch(train_mode, train_dataset, valid_dataset, bs, dataset, valid_shuffle=False):
    
    if dataset != 'cifar100':
        num_label = 10
    else:
        num_label = 100
    
    label = [i for i in range(num_label)]

    
    train_loader = []
    
    if train_mode == EXP_MODES.RANDOM_1 or train_mode == EXP_MODES.RANDOM_5:
        length_train = len(train_dataset)
        selection_rate = 1
        if train_mode == EXP_MODES.RANDOM_1: 
            selection_rate = 0.99
        elif train_mode == EXP_MODES.RANDOM_5:
            selection_rate = 0.95
        random_indices_train = random.sample(range(length_train), (int)(length_train*selection_rate))
        reduced_train_dataset = [train_dataset[i] for i in random_indices_train]
        train_loader = torch.utils.data.DataLoader(reduced_train_dataset, batch_size=bs, shuffle=True, num_workers=4)   
        if valid_shuffle:  
            length_valid = len(valid_dataset)
            random_indices_valid = random.sample(range(length_valid), (int)(length_valid*selection_rate))
            reduced_valid_dataset = [valid_dataset[i] for i in random_indices_valid]
            valid_dataset = reduced_valid_dataset


    elif train_mode == EXP_MODES.EVEN_5 or train_mode == EXP_MODES.EVEN_LABEL:
        train_data_len_by_label = []
        valid_data_len_by_label = []
        train_data_by_label = []
        valid_data_by_label = []

        for i in label:
            train_data_by_label.append([])
            valid_data_by_label.append([])
        for i in train_dataset:
            train_data_by_label[i[1]].append(i)
        for i in valid_dataset:
            valid_data_by_label[i[1]].append(i)
        for i in label:
            random.shuffle(train_data_by_label[i])
            random.shuffle(valid_data_by_label[i])
            train_data_len_by_label.append(len(train_data_by_label[i]))
            valid_data_len_by_label.append(len(valid_data_by_label[i]))
        train_min_len = min(train_data_len_by_label)
        if train_mode == EXP_MODES.EVEN_5: 
            my_length = len(train_dataset)
            temp = (int)((my_length/num_label)*.95)
            if temp<train_min_len:
                train_min_len = temp
        
        for i in train_data_by_label:
            t = i[:train_min_len]
            train_loader += t
        train_loader = torch.utils.data.DataLoader(tuple(train_loader), batch_size=bs, shuffle=True, num_workers=4)

        if valid_shuffle:     
            val_min_len = min(valid_data_len_by_label)
            val_temp = []
            for i in range(val_min_len):
                w = []
                for t in valid_data_by_label:
                    w.append(t.pop())
                val_temp+=w
            valid_dataset = val_temp


        '''
        val_min_len = min(valid_len)
        val_temp = []

        for i in range(val_min_len):
            w = []
            for t in valid_data_by_label:
                w.append(t.pop())
            val_temp+=w
        valid_dataset = val_temp       
        '''
        '''                                                                                                               # 균등
        temp = []
        for i in range(min_len):
            w = []
            for t in train_data_by_label:
                w.append(t.pop())
            temp+=w
        train_loader = torch.utils.data.DataLoader(tuple(temp), batch_size=bs, shuffle=False, num_workers=4)
        '''
    elif train_mode == EXP_MODES.DYNAMIC_AUG:
        aug_data = torch.load("aug_{}.pt".format(dataset_name))
        aug, _ = train_test_split(aug_data, train_size=aug_rate, shuffle=False)
        aug_train_dataset = train_dataset + aug
        train_loader = torch.utils.data.DataLoader(aug_train_dataset, batch_size=bs, shuffle=True, num_workers=4)   

    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=bs, shuffle=True, num_workers=4)
    
    return train_loader, valid_loader


def loadData(dataset_name = None, applyDataAug = False, aug_rate = 0.2, aug_sh = True):
    if dataset_name == "cifar10" or dataset_name == "cifar100":
        transform_t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform_t = transforms.ToTensor()

    data_dir = './data/{}'.format(dataset_name)
    data_download_Flag = False
    if not os.path.exists(data_dir):
        data_download_Flag = True

    dataset = dataset_load_func[dataset_name](root=data_dir, 
                                              train=True, 
                                              download=data_download_Flag,
                                              transform=transform_t
                                              )
    
    train_dataset, valid_dataset = train_test_split(dataset, test_size=0.2, shuffle=False)

    test_dataset = dataset_load_func[dataset_name](root=data_dir, 
                                                   train=False, 
                                                   download=True,
                                                   transform=transform_t)

    return train_dataset, valid_dataset, test_dataset

def load_aug_Data(dataset_name = None):
    if dataset_name == "cifar10" or dataset_name == "cifar100":
        transform_t = transforms.Compose([
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
        transform_v = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform_t = transforms.ToTensor()

    data_dir = './data/{}'.format(dataset_name)
    data_download_Flag = False
    if not os.path.exists(data_dir):
        data_download_Flag = True

    dataset = dataset_load_func[dataset_name](root=data_dir, 
                                              train=True, 
                                              download=data_download_Flag,
                                              transform=None
                                              )
    
    train_dataset, valid_dataset = train_test_split(list(range(len(dataset))), test_size=0.2, shuffle=False)

    test_dataset = dataset_load_func[dataset_name](root=data_dir, 
                                                   train=False, 
                                                   download=True,
                                                   transform=transform_v)
    
    train_dataset = Subset(dataset, train_dataset)
    valid_dataset = Subset(dataset, valid_dataset)
    
    train_dataset.dataset.transform = transform_t
    valid_dataset.dataset.transform = transform_v

    return train_dataset, valid_dataset, test_dataset
