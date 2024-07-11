from datasets.ham10k import HAM10kSegmentationDataset
from transforms.transforms import transform
from datasets.helpers import tensor2img
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

img_folder = "/home/jose/datos/HAM10000/HAM10000_images/"
mask_folder = "/home/jose/datos/HAM10000/HAM10000_segmentations_lesion_tschandl/HAM10000_segmentations_lesion_tschandl/"

img_pathlist = sorted([str(x) for x in Path(img_folder).glob("**/*.jpg")])
mask_pathlist = sorted([str(x) for x in Path(mask_folder).glob("**/*.png")])

dataset = HAM10kSegmentationDataset(img_pathlist, mask_pathlist, crop_size=224, transform=transform((224,224)))

for img, mask in dataset:

    img = tensor2img(img)
    mask = mask.numpy().astype(np.uint8)
    plt.figure()
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(mask)
    plt.show()
    plt.pause(5)