from torchvision import transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

def get_basic(x):
    basic = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return basic(x)

def get_aug(x):
    if isinstance(x, tuple):
        aug = A.Compose([
            A.RandomResizedCrop(height=224, width=224, scale=(0.7, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            A.ToGray(p=0.1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        aug_imgs = aug(image=x[0].numpy(), mask=x[1].numpy())
        return (aug_imgs['image'], aug_imgs['mask'])
    
    else:
        aug = T.Compose(
            [
                T.RandomResizedCrop(224, scale=(0.7, 1.0)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.3, 0.3, 0.3, 0.3),
                T.RandomGrayscale(p=0.1),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return aug(x)
