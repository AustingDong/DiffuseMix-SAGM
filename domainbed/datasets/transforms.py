from torchvision import transforms as T
import torch

def get_basic(seed):
    return T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

def get_aug(seed):
    T.functional.set_seed(seed)
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
    return aug
