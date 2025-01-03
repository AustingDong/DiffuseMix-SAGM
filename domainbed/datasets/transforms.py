from torchvision import transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
from domainbed.utils.diffusemix_utils import AdaptiveDiffuseMixUtils
import torch
import numpy as np


basic_A = A.Compose([
        A.Resize(height=224, width=224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], additional_targets={'image_gen': 'image'}
)

basic_T = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

aug_A = A.Compose([
        A.RandomResizedCrop(height=224, width=224, scale=(0.7, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
        A.ToGray(p=0.1),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], additional_targets={'image_gen': 'image'})

aug_T = T.Compose(
            [
                T.RandomResizedCrop(224, scale=(0.7, 1.0)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.3, 0.3, 0.3, 0.3),
                T.RandomGrayscale(p=0.1),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

diffusemix_utils = AdaptiveDiffuseMixUtils

def get_basic(x):
    if isinstance(x, tuple):
        x = np.array(x)
        basic_imgs = basic_A(image=x[0], image_gen=x[1])
        return basic_imgs['image'], basic_imgs['image_gen']
    
    else:
        return basic_T(x)

def get_aug(x, diffusemix=False, diffusemix_args=None):
    if isinstance(x, tuple):
        if diffusemix:
            blended_img = diffusemix_utils.create_image(x[0], x[1], diffusemix_args["fractal_root"], diffusemix_args["num_slices"], diffusemix_args["alpha"])
            x = (x[0], blended_img)
        x = np.array(x)
        aug_imgs = aug_A(image=x[0], image_gen=x[1])
        return aug_imgs['image'], aug_imgs['image_gen']
    
    else:
        return aug_T(x)
