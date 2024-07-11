import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.augmentations.geometric.resize import SmallestMaxSize
import cv2
import numpy as np
from PIL import Image

VOC12_PIXEL_WEIGHTLIST = {
    'background': 0.000904033501971689, 
    'aeroplane': 0.05580971950578698,   
    'bicycle': 0.15512901475841676, 
    'bird': 0.04740651893772769, 
    'boat': 0.07187987804855628, 
    'bottle': 0.06605527448142, 
    'bus': 0.023431610850140144, 
    'car': 0.03190424653970079, 
    'cat': 0.019373015794332062, 
    'chair': 0.06140854291701378, 
    'cow': 0.039445520421651, 
    'diningtable': 0.046379051274731395, 
    'dog': 0.02286588678863056, 
    'horse': 0.042673821654842, 
    'motorbike': 0.042815967498724766, 
    'person': 0.009404935990145382, 
    'potted plant': 0.08560038227768499, 
    'sheep': 0.04730992453835607, 
    'sofa': 0.03801971776752205, 
    'train': 0.02666702480980015, 
    'tv/monitor': 0.056240339221151625, 
    'contour': 0.009275572421693995
    }

class VOCSegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, crop_size=256, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.crop_size = crop_size
        self.num_classes = 22

    def __getitem__(self, index):
        img = cv2.imread(self.image_paths[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.
        mask = np.array(Image.open(self.mask_paths[index]))
        mask = np.where(mask==255, 21, mask)

        if self.transform:
            if min(img.shape[:2]) < self.crop_size:
                resize_transform = SmallestMaxSize(max_size=self.crop_size)
                aug = resize_transform(image=img, mask=mask)
                img, mask = aug['image'], aug['mask']
            aug = self.transform(image=img, mask=mask)
            img = aug['image']
            mask = aug['mask'].to(torch.long)

        return img, mask
        
        

    def __len__(self):
        return len(self.image_paths)