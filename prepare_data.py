import torch
import torchvision
import torchvision.transforms as transforms
from UniformAugment import UniformAugment

import os
from sklearn.model_selection import train_test_split

from exp_mode import EXP_MODES
from autoaug import ImageNetPolicy
from aug_transform import *
from utils import *

dataset_load_func = {
    'mnist': torchvision.datasets.MNIST,
    'fmnist':torchvision.datasets.FashionMNIST,
    'cifar10': torchvision.datasets.CIFAR10,
    'cifar100': torchvision.datasets.CIFAR100,
    'stl10': torchvision.datasets.STL10,
}

d_size = {'cifar10':32, 'cifar100':32, 'stl10':96, 'caltech101':224}

mean = {
    'cifar10': [0.4914, 0.4822, 0.4465],
    'cifar100': [0.4914, 0.4822, 0.4465],
    'stl10': [0.4467106, 0.43980986, 0.40664646],
    'caltech101': [0.54558825, 0.52854234, 0.50220895]
}

std = {
    'cifar10': [0.2023, 0.1994, 0.2010],
    'cifar100': [0.2023, 0.1994, 0.2010],
    'stl10': [0.22414584, 0.22148906, 0.22389975],
    'caltech101': [0.23349172, 0.23052077, 0.23274076]
}

my_exp_transforms = [transform_aug0_rc, transform_uniform, transform_autoaug, transform_aug1]

def prepare_train_aug_data_per_epoch(train_dataset, aug_data, aug_val, bs, mode):
    aug_rate = aug_val/100
   
    if aug_rate<1:
        if mode == EXP_MODES.ORIG_PLUS_VALID_AUG_1X:
            train_dataset, _ = train_test_split(train_dataset, train_size=1-aug_rate, shuffle=True)
        aug, _ = train_test_split(aug_data, train_size=aug_rate, shuffle=True)
        my_train_dataset = train_dataset + aug
    else:
        aug = list(aug_data)        
        if mode == EXP_MODES.ORIG_PLUS_VALID_AUG_1X:
            my_train_dataset = aug
        elif mode == EXP_MODES.ORIG_PLUS_VALID_AUG_2X:
            my_train_dataset = train_dataset + aug
            
    train_loader = torch.utils.data.DataLoader(my_train_dataset, batch_size=bs, shuffle=True, num_workers=8)   
    
    return train_loader

def load_caltech(ori, data, transform_index = 0):

    if ori:
        transform_d = transform_original
    else:   
        transform_d = my_exp_transforms[transform_index]
        transform_d.transforms.append(transforms.Normalize(mean[data], std[data]))
        
    if transform_index == 2:
        transform_d.transforms.pop(1)
        transform_d.transforms.insert(0, ImageNetPolicy())
        
    transform_d.transforms.insert(0, transforms.Resize((224, 224)))

    if not os.path.isdir('./data/caltech101'):
        os.system(f"git clone https://github.com/MachineLearning2020/Homework2-Caltech101.git")
        
        classes = []
        rootdir = 'Homework2-Caltech101/101_ObjectCategories'
        dirs = "./data/caltech101"

        for file in os.listdir(rootdir):
            d = os.path.join(rootdir, file)
            if os.path.isdir(d):
                label = d.split("/")[-1]
                if label == "BACKGROUND_Google":
                    continue
                classes.append(label)
                
        os.makedirs((dirs))
        split_files(rootdir, dirs, classes)
        
    train_data = torchvision.datasets.ImageFolder(root='./data/caltech101/train', transform=transform_d)
    valid_data = torchvision.datasets.ImageFolder(root='./data/caltech101/validation', transform=transform_d)
    test_data = torchvision.datasets.ImageFolder(root='./data/caltech101/test', transform=transform_d)

    if os.path.isdir('./Homework2-Caltech101'):
        os.system(f"rm -rf './Homework2-Caltech101'")
        
    return train_data, valid_data, test_data

def load_original_Data(dataset_name = None,valid_rate=0.2):

    data_dir = './data/{}'.format(dataset_name)
    data_download_Flag = False
    
    if not os.path.exists(data_dir):
        data_download_Flag = True
        
    transform_original.transforms.append(transforms.Normalize(mean[dataset_name], std[dataset_name]))
    
    if dataset_name == "stl10":
        dataset = dataset_load_func[dataset_name](root=data_dir, 
                                                  split="train", 
                                                  download=data_download_Flag,
                                                  transform=transform_original
                                                  )

        train_dataset, valid_dataset = train_test_split(dataset, test_size=valid_rate, shuffle=False)

        test_dataset = dataset_load_func[dataset_name](root=data_dir, 
                                                       split="test", 
                                                       download=True,
                                                       transform=transform_original)
    elif dataset_name == "caltech101":
        train_dataset, valid_dataset, test_dataset = load_caltech(True, dataset_name)
    else:
        dataset = dataset_load_func[dataset_name](root=data_dir, 
                                                  train=True, 
                                                  download=data_download_Flag,
                                                  transform=transform_original
                                                  )

        train_dataset, valid_dataset = train_test_split(dataset, test_size=valid_rate, shuffle=False)

        test_dataset = dataset_load_func[dataset_name](root=data_dir, 
                                                       train=False, 
                                                       download=True,
                                                       transform=transform_original)

    return train_dataset, valid_dataset, test_dataset

def load_exp_aug_Data(dataset_name,transform_index,valid_rate):
    
    data_dir = './data/{}'.format(dataset_name)
    
    transform_uniform.transforms.insert(0, transforms.RandomCrop(d_size[dataset_name], padding=4))
    transform_uniform.transforms.insert(2, UniformAugment())
    
    trans = my_exp_transforms[transform_index]
    
    if dataset_name == "caltech101":
        train_dataset, valid_dataset, _ = load_caltech(False, dataset_name, transform_index)
        if valid_rate == 0:
            aug_train_dataset = train_dataset + valid_dataset
        else:
            aug_train_dataset = train_dataset
    elif dataset_name == "stl10":
        trans.transforms.append(transforms.Normalize(mean[dataset_name], std[dataset_name]))
        aug_train_dataset = dataset_load_func[dataset_name](root=data_dir, split="train", download=True, transform=trans)
    else:
        trans.transforms.append(transforms.Normalize(mean[dataset_name], std[dataset_name]))
        aug_train_dataset = dataset_load_func[dataset_name](root=data_dir, train=True, download=True, transform=trans)
        
    train_size = int((1-valid_rate) * len(aug_train_dataset))
    train_aug_dataset = torch.utils.data.Subset(aug_train_dataset, range(train_size))
    
    return train_aug_dataset