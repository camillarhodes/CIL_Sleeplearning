import albumentations as A

   
crop_only = [
    A.RandomCrop(width=256, height=256),
]

crop_flip_constrast = [
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
]

crop_flip = [
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
]

resize_to_256 = [
    A.geometric.resize.Resize(256,256),
]

def get_transforms(augmentation_type, **kwargs):
    if augmentation_type == 'none' or augmentation_type == None:
        return A.Compose([])
    if augmentation_type == 'c':
        return A.Compose(crop_only)
    if augmentation_type == 'cf':
        return A.Compose(crop_flip)
    if augmentation_type == 'cfc':
        return A.Compose(crop_flip_constrast)
    if augmentation_type == 'r256':
        return A.Compose(resize_to_256)
    raise NotImplementedError('Unsupported augmentation type')