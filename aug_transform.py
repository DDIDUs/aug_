import torchvision.transforms as transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

transform_original = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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

transform_autoaug = transforms.Compose([
            AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

transform_uniform = transforms.Compose([
    transforms.RandomCrop(96, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])