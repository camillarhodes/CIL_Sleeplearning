from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
from pathlib import Path
import numpy as np

import torch
import cv2


class SegDataset(Dataset):
    def __init__(self, transform, data_path='./data/training', split_percent=0.85, validation=False):
        img_dir = Path(f'{data_path}/images')
        mask_dir = Path(f'{data_path}/groundtruth')
        self.img_files = list(img_dir.glob('*.png'))
        self.mask_files = list(mask_dir.glob('*.png'))
        
        total_len = len(self.img_files)
        training_len = int(split_percent * total_len)
        
        if validation:
            self.img_files = self.img_files[training_len:]
            self.mask_files = self.mask_files[training_len:]
        else:
            self.img_files = self.img_files[:training_len]
            self.mask_files = self.mask_files[:training_len]
            
        self.transform = transform
        
    def __len__(self):
        return len(self.img_files)
        
    def __getitem__(self, idx):

        img = cv2.imread(str(self.img_files[idx]))
        mask = cv2.imread(str(self.mask_files[idx]))

        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']
            
            
        img = (img.transpose(2,0,1) / 255).astype(np.float32)
        mask = (mask[:,:,:1].transpose(2,0,1) / 255).astype(np.float32)
        return img, mask
    
    
class MasDataset(Dataset):
    def __init__(self, transform, data_path='./Massachusetts/tiff'):
        data_dir = Path(data_path)
        self.img_files = list(data_dir.rglob('*.tiff'))
        self.mask_files = list(data_dir.rglob('*.tif'))
        self.transform = transform
        
    def __len__(self):
        return len(self.img_files)
        
    def __getitem__(self, idx):

        img = cv2.imread(str(self.img_files[idx]))
        mask = cv2.imread(str(self.mask_files[idx]))

        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']
            
            
        img = (img.transpose(2,0,1) / 255).astype(np.float32)
        mask = (mask[:,:,:1].transpose(2,0,1) / 255).astype(np.float32)
        return img, mask
    
    
    
    
def get_train_val_dataloaders(split_percent=0.85, batch_size=4, num_workers=4, transform_train=None, transform_val=None, include_massachusetts=True):
    train_dataset = SegDataset(transform_train, split_percent=split_percent, validation=False)
    val_dataset = SegDataset(transform_val, split_percent=split_percent, validation=True)
    
    if include_massachusetts:
        mas_dataset = MasDataset(transform_train)
        train_dataset = ConcatDataset([train_dataset, mas_dataset])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
    
    return train_dataloader, val_dataloader