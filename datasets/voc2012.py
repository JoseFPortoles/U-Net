import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.augmentations.geometric.resize import SmallestMaxSize
import cv2
import numpy as np
from PIL import Image

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