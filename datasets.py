from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split


class SegDataset(Dataset):
    def __init__(
        self,
        transform,
        data_path="./data/training",
        split_percent=0.85,
        validation=False,
        upscaling=None,
    ):
        """Dataset class for the main dataset.

        Parameters:
            transform : str
                The type of data augmentations to perform.
            data_path : str
                The path to the training data.
            split_percent: float
                The ratio of training data to validation data. This is ignored if validation is set to False.
            validation : bool
                A flag whether to use validation.
            upscaling : str, optional
                The type of upscaling to use, if any. Supported values are None and "vdsr".
        """

        if upscaling == "vdsr":
            img_dir = Path(f"{data_path}/images_800")
            mask_dir = Path(f"{data_path}/groundtruth_800")
        else:
            img_dir = Path(f"{data_path}/images")
            mask_dir = Path(f"{data_path}/groundtruth")

        self.img_files = list(img_dir.glob("*.png"))
        self.mask_files = list(mask_dir.glob("*.png"))

        # Split data into training and validation if required
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

        # Apply data augmentation, if any
        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        img = (img.transpose(2, 0, 1) / 255).astype(np.float32)
        mask = (mask[:, :, :1].transpose(2, 0, 1) / 255).astype(np.float32)

        return img, mask


class MasDataset(Dataset):
    """Dataset class for the Massachusets dataset.

    Parameters:
        transform : str
            The type of data augmentations to perform.
        data_path : str
            The path to the training data.
        return_canny : bool
            A flag whether to return canny.
    """

    def __init__(
        self, transform, data_path="./data/Massachusetts/tiff", return_canny=False
    ):
        self.return_canny = return_canny
        data_dir = Path(data_path)
        unfiltered_img_files = sorted(
            list(data_dir.rglob("*.tiff")), key=lambda x: x.stem
        )
        unfiltered_mask_files = sorted(
            list(data_dir.rglob("*.tif")), key=lambda x: x.stem
        )
        filtered_img_files = []
        filtered_mask_files = []

        for (img_path, mask_path) in zip(unfiltered_img_files, unfiltered_mask_files):
            mask = cv2.imread(str(mask_path))
            if mask.mean() > 25:
                filtered_img_files.append(img_path)
                filtered_mask_files.append(mask_path)

        self.img_files = filtered_img_files
        self.mask_files = filtered_mask_files
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):

        img = cv2.imread(str(self.img_files[idx]))
        mask = cv2.imread(str(self.mask_files[idx]))

        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        if self.return_canny:
            canny = cv2.Canny(img, 300, 300)

        img = (img.transpose(2, 0, 1) / 255).astype(np.float32)
        mask = (mask[:, :, :1].transpose(2, 0, 1) / 255).astype(np.float32)

        if self.return_canny:
            canny = (canny[None, :, :] / 255).astype(np.float32)
            return img, mask, canny
        return img, mask


def get_train_val_dataloaders(
    split_percent=0.85,
    batch_size=4,
    num_workers=4,
    transform_train=None,
    transform_val=None,
    include_massachusetts=False,
    upscaling=None,
    return_canny=False,
):
    """A method to get the training and validation dataloaders.

    Parameters:
        split_percent: float
            The ratio of training data to validation data. This is ignored if validation is set to False.
        batch_size : int
            The batch size.
        num_workers : int
            The amount of workers to employ for the dataloader. Set this to 1 if using Windows.
        transform_train : str, optional
            The type of data augmentations to perform on the training set.
        transform_val : str, optional
            The type of data augmentations to perform on the validation set.
        include_massachusetts : boolean
            A flag whether to include the Massachusetts dataset.
        data_path : str
            The path to the training data.
        upscaling : str, optional
            The type of upscaling to use, if any. Supported values are None and "vdsr".
        return_canny : bool
            A flag whether to return canny.

    Returns:
        train_dataloader, val_dataloader
            The training and validation dataloaders. If validation is set to False, the validation dataloader will be empty.
    """
    train_dataset = SegDataset(
        transform_train,
        split_percent=split_percent,
        validation=False,
        upscaling=upscaling,
    )
    val_dataset = SegDataset(
        transform_val, split_percent=split_percent, validation=True, upscaling=upscaling
    )

    if include_massachusetts:
        mas_dataset = MasDataset(transform_train, return_canny=return_canny)
        train_dataset = ConcatDataset([train_dataset, mas_dataset])
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True
    )

    return train_dataloader, val_dataloader
