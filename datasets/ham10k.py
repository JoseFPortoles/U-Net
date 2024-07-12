import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.augmentations.geometric.resize import SmallestMaxSize
import cv2
import numpy as np

class HAM10kSegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, crop_size=256, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.crop_size = crop_size
        self.num_classes = 2

    def __getitem__(self, index):
        img = cv2.imread(self.image_paths[index])
        mask = cv2.imread(self.mask_paths[index], cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis]

        if self.transform:
            if min(img.shape[:2]) < self.crop_size:
                resize_transform = SmallestMaxSize(max_size=self.crop_size)
                aug = resize_transform(image=img, mask=mask)
                img, mask = aug['image'], aug['mask']
            aug = self.transform(image=img, mask=mask)
            img = aug['image'].to(torch.float32)
            mask = aug['mask'].permute(2,0,1).to(torch.float32)/255.

        return img, mask
        
    def __len__(self):
        return len(self.image_paths)