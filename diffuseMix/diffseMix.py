import torch
from torch.utils.data import Dataset
from utils import Utils
from PIL import Image
class DiffuseMix(Dataset):

    def __init__(self):
        pass

    
    def augmentation(self):
        pass


    def __len__(self):
        pass


    def __getitem__(self, idx):
        return self.augmented_images[idx]
