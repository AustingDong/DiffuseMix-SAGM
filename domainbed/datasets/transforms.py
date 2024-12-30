from torchvision import transforms as T
import torchvision.transforms.v2 as v2
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
        seed = torch.randint(0, 10000, (1,)).item()
        torch.manual_seed(seed)
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
        return tuple(aug(xx) for xx in x)
    
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
