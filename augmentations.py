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

crop_flip_scale = [
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.Affine(scale=(0.7,1.3), keep_ratio=True)
]

rotate_crop_flip = [
    A.Affine(rotate=(-180,180)),
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
]

rotate_crop_flip_512 = [
    A.Affine(rotate=(-180,180)),
    A.RandomCrop(width=512, height=512),
    A.HorizontalFlip(p=0.5),
]

rotate_crop_resize_to_128 = [
    A.Affine(rotate=(-180,180)),
    A.RandomCrop(width=256, height=256),
    A.geometric.resize.Resize(128,128)
]

rotate_crop_resize_to_512 = [
    A.Affine(rotate=(-180,180)),
    A.RandomCrop(width=256, height=256),
    A.geometric.resize.Resize(512,512)
]

center_crop = [
    A.CenterCrop(width=256, height=256)
]

center_crop_512 = [
    A.CenterCrop(width=512, height=512)
]

center_crop_resize_to_128 = [
        A.CenterCrop(width=256, height=256),
        A.geometric.resize.Resize(128,128),
]

center_crop_resize_to_512 = [
        A.CenterCrop(width=256, height=256),
        A.geometric.resize.Resize(512,512),
]

resize_to_256 = [
    A.geometric.resize.Resize(256,256),
]

resize_to_384 = [
    A.geometric.resize.Resize(384,384),
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
    if augmentation_type == 'cfs':
        return A.Compose(crop_flip_scale)
    if augmentation_type == 'rcf':
        return A.Compose(rotate_crop_flip)
    if augmentation_type == 'rcf512':
        return A.Compose(rotate_crop_flip_512)
    if augmentation_type == 'rcr128':
        return A.Compose(rotate_crop_resize_to_128)
    if augmentation_type == 'rcr512':
        return A.Compose(rotate_crop_resize_to_512)
    if augmentation_type == 'r256':
        return A.Compose(resize_to_256)
    if augmentation_type == 'center_c':
        return A.Compose(center_crop)
    if augmentation_type == 'center_c512':
        return A.Compose(center_crop_512)
    if augmentation_type == 'center_cr128':
        return A.Compose(center_crop_resize_to_128)
    if augmentation_type == 'center_cr512':
        return A.Compose(center_crop_resize_to_512)
    if augmentation_type == 'resize_384':
        return A.Compose(resize_to_384)
    raise NotImplementedError('Unsupported augmentation type')
