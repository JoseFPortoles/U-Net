import albumentations as A
from albumentations.augmentations.crops.transforms import RandomCrop
from albumentations.augmentations.transforms import Normalize
from albumentations.augmentations.geometric.rotate import Rotate
from albumentations.pytorch.transforms import ToTensorV2

def reshape_img_size(img_size):
    if type(img_size) is tuple:
        return img_size
    elif type(img_size) is int:
        return (img_size, img_size)
    else:
        raise TypeError(f"Expected argument img_size is tuple(int,int) or int. A {type(img_size)} was passed instead")

def transform(img_size):
    img_size = reshape_img_size(img_size)    
    return A.Compose([
        RandomCrop(*img_size, p=0.5),
        A.HorizontalFlip(p=0.5),
        Rotate(limit=15, crop_border=True),
        A.Resize(*img_size),
        Normalize(),
        ToTensorV2()
    ])

def transform_ham10k(img_size):
    img_size = reshape_img_size(img_size)    
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        Rotate(limit=15, crop_border=True),
        A.Resize(*img_size),
        ToTensorV2()
    ])

def val_transform(crop_size):
    crop_size = reshape_img_size(crop_size)    
    return A.Compose([
                RandomCrop(*crop_size),
                Normalize(),
                ToTensorV2()
            ])