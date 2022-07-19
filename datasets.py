from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
from pathlib import Path
import numpy as np

import torch
import cv2


class SegDataset(Dataset):
    def __init__(self, transform, data_path='./data'):
        training_img_dir = Path(f'{data_path}/training/images')
        training_mask_dir = Path(f'{data_path}/training/groundtruth')
        self.training_img_files = list(training_img_dir.glob('*.png'))
        self.training_mask_files = list(training_mask_dir.glob('*.png'))
        self.transform = transform

    def __len__(self):
        return len(self.training_img_files)

    def __getitem__(self, idx):

        img = cv2.imread(str(self.training_img_files[idx]))
        mask = cv2.imread(str(self.training_mask_files[idx]))

        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']

        img = (img.transpose(2, 0, 1) / 255).astype(np.float32)
        mask = (mask[:, :, :1].transpose(2, 0, 1) / 255).astype(np.float32)
        return img, mask


class MasDataset(Dataset):
    def __init__(self, transform, data_path='./Massachusetts/tiff'):
        data_dir = Path(data_path)
        self.training_img_files = list(data_dir.rglob('*.tiff'))
        self.training_mask_files = list(data_dir.rglob('*.tif'))
        self.transform = transform

    def __len__(self):
        return len(self.training_img_files)

    def __getitem__(self, idx):

        img = cv2.imread(str(self.training_img_files[idx]))
        mask = cv2.imread(str(self.training_mask_files[idx]))

        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']

        img = (img.transpose(2, 0, 1) / 255).astype(np.float32)
        mask = (mask[:, :, :1].transpose(2, 0, 1) / 255).astype(np.float32)
        return img, mask


def get_train_val_dataloaders(split_percent=0.85, batch_size=4, num_workers=4, transform=None, include_massachusetts=True):
    seg_dataset = SegDataset(transform)

    seg_dataset_len = len(seg_dataset)
    train_size = int(seg_dataset_len*split_percent)
    val_size = seg_dataset_len - train_size

    train_dataset, val_dataset = random_split(
        seg_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    if include_massachusetts:
        mas_dataset = MasDataset(transform)
        train_dataset = ConcatDataset([train_dataset, mas_dataset])
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=num_workers)

    return train_dataloader, val_dataloader
