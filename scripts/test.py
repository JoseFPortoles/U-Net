import PIL.Image
import torch
import torchvision
import argparse
import os
from models.ext_unet import UNet
from transforms.transforms import test_transform
import PIL
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser(prog='test.py', description='Test U-Net on test dataset images')
parser.add_argument('--input_size', type=int, default=224, help="Input image size.")
parser.add_argument('--weights_path', type=str, default=None, help='Weights file path.')
parser.add_argument('--test_img_path', type=str, help='Path to test image.')
parser.add_argument('--output_path', type=str, default='./out', help='Folder where results are saved.')
parser.add_argument('--num_segment_categories', type=int, default=1, help='Number of segmentation categories, including background.')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

unet = UNet(weights=torchvision.models.VGG16_BN_Weights.IMAGENET1K_V1, out_channels=args.num_segment_categories).to(device)

if args.weights_path:
    if Path(args.weights_path).is_file():
        try:
            checkpoint = torch.load(args.weights_path, map_location=device)
            unet.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint at {args.weights_path}")
        except Exception as e:
            print(f"ERROR: Attempt at loading weights from {args.weights_path} threw an exception {e}.")    
    else:
        raise ValueError(f"Specified weights path {args.weights_path} does not exist, model was xavier initialised")

input_img = np.array(PIL.Image.open(args.test_img_path).convert('RGB'))

pred = unet(test_transform(args.input_size)(image=input_img)['image'].unsqueeze(0)).squeeze(0)

pred_mask = ((pred - pred.min())/(pred.max() - pred.min())*255.).to(torch.int8).squeeze(0).numpy()

print(f"type: {type(pred)}, shape: {pred.shape}")
if Path(args.output_path).is_dir() is False:
    Path(args.output_path).mkdir(parents=True, exist_ok=True)
PIL.Image.fromarray(pred_mask, mode='L').save(Path(args.output_path)/f"pred_mask_{Path(args.test_img_path).stem}.png")