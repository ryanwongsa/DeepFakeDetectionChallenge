from albumentations import JpegCompression, OneOf, Compose, HorizontalFlip
from albumentations.augmentations.transforms import Resize, Downscale
import albumentations as A
import random

def prep_transform(height, width, mappings, p=2/3):
    scale = random.randint(2, 4)
    return Compose([
        OneOf([
            JpegCompression(quality_lower=20, quality_upper=70, p=0.5),
            Downscale(scale_min=0.25, scale_max=0.50, interpolation=1, p=0.5),
            Resize(height//scale,width//scale, interpolation=1, p=1.0)
        ], p=1.0),
        HorizontalFlip(p=0.5)
    ], p=p,
    additional_targets=mappings)


def base_transform(height, width, mappings, p=2/3):
    return Compose([
        OneOf([
            JpegCompression(quality_lower=20, quality_upper=70, p=0.5),
            Downscale(scale_min=0.25, scale_max=0.50, interpolation=1, p=0.5),
            Resize(height//4,width//4, interpolation=1, p=0.5)
        ], p=1.0),
        HorizontalFlip(p=0.5)
    ], p=p,
    additional_targets=mappings)

base_aug = base_transform, (2, 1/3)


def more_transform(height, width, mappings, p=2/3):
    scale = random.randint(2, 4)
    return Compose([
        OneOf([
            JpegCompression(quality_lower=20, quality_upper=70, p=0.5),
            Downscale(scale_min=0.25, scale_max=0.50, interpolation=1, p=0.5),
            Resize(height//scale,width//scale, interpolation=1, p=1.0)
        ], p=1.0),
        HorizontalFlip(p=0.5),
        A.augmentations.transforms.GaussNoise(p=0.2),
        A.RandomBrightnessContrast(p=0.2),    
        A.RandomGamma(p=0.2),    
        A.CLAHE(p=0.2),
        A.ChannelShuffle(p=0.2),
        A.MultiplicativeNoise(multiplier=[0.5, 1.5], elementwise=True, p=0.1),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.2),
    ], p=p,
    additional_targets=mappings)

more_aug = more_transform, (2,1/3)

def even_more_transform(height, width, mappings, p=2/3):
    scale = random.randint(2, 4)
    return Compose([
        OneOf([
            JpegCompression(quality_lower=20, quality_upper=70, p=0.5),
            Downscale(scale_min=0.25, scale_max=0.50, interpolation=1, p=0.5),
            Resize(height//scale,width//scale, interpolation=1, p=1.0)
        ], p=0.6),
        HorizontalFlip(p=0.5),
        A.augmentations.transforms.GaussNoise(p=0.2),
        A.RandomBrightnessContrast(p=0.2),    
        A.RandomGamma(p=0.2),    
        A.CLAHE(p=0.2),
        A.ChannelShuffle(p=0.2),
        A.MultiplicativeNoise(multiplier=[0.5, 1.5], elementwise=True, p=0.1),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.2),
    ], p=0.9,
    additional_targets=mappings)

even_more_aug = even_more_transform, (2,1/3)