import numpy as np
import os
import json
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as albu
from lib import get_mask

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn)
    ]
    return albu.Compose(_transform)


class CustomDataset(Dataset):
    def __init__(self, data_path, masks=True, preprocessing=None, transforms=None):
        if masks:
            self.annotations = json.load(open(f"{data_path}/coco_annotations.json", "r"))
        else:
            self.files = os.listdir(data_path)
        self.path = data_path    
        self.masks = masks
        self.transforms = transforms
        self.preprocessing = preprocessing

    def __getitem__(self, idx):
        if self.masks:
            mask = get_mask(idx, self.annotations)
            mask = mask.reshape(mask.shape[0], mask.shape[1])
            file_name = self.annotations["images"][idx]['file_name']
            img = cv2.imread(f"{self.path}/images/{file_name}")
        else:
            img = cv2.imread(f"{self.path}/{self.files[idx]}")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.masks:
            if self.transforms is not None:
                augmented = self.transforms(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask']
                
            if self.preprocessing is not None:
                preprocessed = self.preprocessing(image=img, mask=mask)
                img = preprocessed['image']
                mask = preprocessed['mask']
        else:
            if self.preprocessing is not None:
                preprocessed = self.preprocessing(image=img)
                img = preprocessed['image']

        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()
        
        if self.masks:
            mask = torch.from_numpy(mask)        
            mask = (mask > 0).float()
            return img, mask
        else:
            return img

    def __len__(self):
        if self.masks:
            return len(self.annotations["images"])
        else:
            return len(self.files)