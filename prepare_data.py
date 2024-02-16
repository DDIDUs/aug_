from random import shuffle
import torch
import torchvision
import torchvision.transforms as transforms
from exp_mode import EXP_MODES

from sklearn.model_selection import train_test_split
import os

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

# rc: random choice 과거 버전
transform_aug_rc_old = transforms.Compose([
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

# rc: random choice 
transform_aug0_rc = transforms.Compose([
            transforms.RandomChoice([
                transforms.RandomHorizontalFlip(p=1),
                transforms.RandomRotation(10),
                transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                ]),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

# aug: 우리가 일반적으로 augmentation 이라고 하는 변형 
transform_aug1 = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# s1: single 1 -> augmentation 1개만 적용
transform_aug2_s1 = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
# s2: single 1 -> augmentation 1개만 적용
transform_aug3_s2 = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# s3: single 1 -> augmentation 1개만 적용
transform_aug4_s3 = transforms.Compose([
    transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# s4: single 1 -> augmentation 1개만 적용
transform_aug5_s4 = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

my_exp_transforms = [transform_aug0_rc,transform_aug1,transform_aug2_s1,transform_aug3_s2,transform_aug4_s3,transform_aug5_s4]

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
    
def getCustomDataset(train_dataset, augmentation_rate,transform_aug):
    custom_train_dataset = CustomCIFAR10Dataset(train_dataset, transform_original, transform_aug, augmentation_rate)
    return custom_train_dataset
    

def prepare_train_aug_data_per_epoch(train_dataset, aug_data, aug_val, bs, mode):
    aug_rate = aug_val/100
   
    if aug_rate<1:
        if mode == EXP_MODES.ORIG_PLUS_DYNAMIC_AUG_1X or mode == EXP_MODES.ORIG_PLUS_VALID_AUG_1X:
            train_dataset, _ = train_test_split(train_dataset, train_size=1-aug_rate, shuffle=True)
        aug, _ = train_test_split(aug_data, train_size=aug_rate, shuffle=True)
        my_train_dataset = train_dataset + aug
    else:
        aug = list(aug_data)        
        if mode == EXP_MODES.ORIG_PLUS_DYNAMIC_AUG_1X or mode == EXP_MODES.ORIG_PLUS_VALID_AUG_1X:
            my_train_dataset = aug
        elif mode == EXP_MODES.ORIG_PLUS_DYNAMIC_AUG_2X or mode == EXP_MODES.ORIG_PLUS_VALID_AUG_2X:
            my_train_dataset = train_dataset + aug
        
    train_loader = torch.utils.data.DataLoader(my_train_dataset, batch_size=bs, shuffle=True, num_workers=8)   
    
    return train_loader


def load_original_Data(dataset_name = None,valid_rate=0.2):

    data_dir = './data/{}'.format(dataset_name)
    data_download_Flag = False
    if not os.path.exists(data_dir):
        data_download_Flag = True

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
    
    # Load CIFAR-10 dataset
    aug_train_dataset = dataset_load_func[dataset_name](root=data_dir, train=True, download=True, transform=my_exp_transforms[transform_index])

    # Split into training and validation sets
    train_size = int((1-valid_rate) * len(aug_train_dataset))
    train_aug_dataset = torch.utils.data.Subset(aug_train_dataset, range(train_size))
    return train_aug_dataset

