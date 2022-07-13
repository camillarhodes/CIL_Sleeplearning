import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
import segmentation_models_pytorch as smp

class PreprocessLikePretraining(ImageOnlyTransform):
    def __init__(self, encoder_name, encoder_weights):
        super().__init__()
        self.fn = smp.encoders.get_preprocessing_fn(encoder_name=encoder_name, pretrained=encoder_weights)
    
    def apply(self, img, **params):
        return self.fn(img)
    
    
crop_only = [
    A.RandomCrop(width=256, height=256),
]

crop_flip_constrast = [
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
]

def get_transforms(augmentation_type, **kwargs):
    if augmentation_type == 'none' or augmentation_type == None:
        return A.Compose([])
    if augmentation_type == 'c':
        return A.Compose(crop_only)
    if augmentation_type == 'cfc':
        return A.Compose(crop_flip_constrast)
    if augmentation_type == 'cfcp':
        encoder_name = kwargs.pop('encoder_name')
        encoder_weights = kwargs.pop('encoder_weights')
        p = PreprocessLikePretraining(encoder_name, encoder_weights)
        return A.Compose(crop_flip_constrast + [p])
    raise NotImplementedError('Unsupported augmentation type')