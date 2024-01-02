from datasets.voc2012 import VOCSegmentationDataset
from transforms.transforms import transform
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse
import os

parser = argparse.ArgumentParser(prog='img_mask_alignment.py', 
                                 description='Show effect of applying transforms simultaneously on image and mask.',)
parser.add_argument('--VOC_root', type=str, help='root path for VOC dataset')
parser.add_argument('--img_size', type=int, default=224, help='Size of the square input image')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
args = parser.parse_args()

voc_root = args.VOC_root
img_size = args.img_size
batch_size = args.batch_size

image_train = os.path.join(voc_root, 'JPEGImages')
mask_train = os.path.join(voc_root, 'SegmentationClass')

train_dataset = VOCSegmentationDataset(image_train, mask_train, crop_size=img_size, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

weights = np.zeros(22)

for _, masks in tqdm(train_loader):
    values, counts = np.unique(masks, return_counts=True)
    for v, c in zip(values,counts):
        weights[v] += c 

for c in range(22):
    print(f"Clase {c} -> #pixels = {weights[c]}\n")

weights = np.reciprocal(weights)
weights = weights/np.sum(weights)


printed_weights = [x for x in weights]
print(f"Weights vector for pascal-voc-2012:\n\nweights = torch.Tensor({printed_weights})")