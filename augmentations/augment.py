from albumentations import JpegCompression, OneOf, Compose, HorizontalFlip
from albumentations.augmentations.transforms import Resize

def base_transform(height, width, mappings, p=2/3):
    return Compose([
        OneOf([
            JpegCompression(quality_lower=20, quality_upper=90, p=0.5),
            Resize(height//4,width//4, interpolation=1, p=0.5)
        ], p=1.0),
        HorizontalFlip(p=0.5)
    ], p=p,
    additional_targets=mappings)

base_aug = base_transform, (2, 1/3)