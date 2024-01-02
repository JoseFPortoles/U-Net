from transforms.transforms import transform
from datasets.voc2012 import VOCSegmentationDataset
from datasets.helpers import tensor2img
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(prog='img_mask_alignment.py', 
                                 description='Show effect of applying transforms simultaneously on image and mask.',)
parser.add_argument('--VOC_root', type=str, help='root path for VOC dataset')
parser.add_argument('--img_size', type=int, default=224, help='Size of the square input image')
parser.add_argument('--img_filename', type=str, help='File name including file extension')
parser.add_argument('--mask_filename', type=str, help='File name including file extension')
args = parser.parse_args()

voc_root = args.VOC_root
img_size = args.img_size
img_filename = args.img_filename
mask_filename = args.mask_filename

# Image and mask from files

img_1 = cv2.imread(os.path.join(voc_root, 'JPEGImages', img_filename))
img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
mask_1 = np.array(Image.open(os.path.join(voc_root, 'SegmentationClass', mask_filename)))
augmented_img = transform(img_size)(image=img_1, mask=mask_1)
img_1 = augmented_img['image'].permute(1,2,0).numpy()
mask_1 = augmented_img['mask'].unsqueeze(-1).numpy()

# Image and mask from Dataset object

img_paths = os.listdir(os.path.join(voc_root, 'JPEGImages'))
mask_paths = os.listdir(os.path.join(voc_root, 'SegmentationClass'))
val_dataset = VOCSegmentationDataset(img_paths, mask_paths, crop_size=img_size, transform=transform(img_size))


aug = val_dataset[0]
img_2 = tensor2img(aug[0])
mask_2 = aug[1].numpy()


plt.figure(figsize=(10,5))
plt.subplot(121)
plt.imshow(img_1)
plt.subplot(122)
plt.imshow(mask_1)

plt.figure(figsize=(10,5))
plt.subplot(121)
plt.imshow(img_2)
plt.subplot(122)
plt.imshow(mask_2)