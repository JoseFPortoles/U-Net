import PIL.Image
import PIL
import matplotlib.cm as cm
import torch
import torchvision
from torchvision import transforms as T
import numpy as np
import argparse
from pathlib import Path
from models.ext_unet import UNet
from transforms.transforms import test_transform

ALPHA = 0.7

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

input_img = PIL.Image.open(args.test_img_path).convert('RGB')
original_size = input_img.size

input_array = np.array(input_img)
pred = unet(test_transform(args.input_size)(image=input_array)['image'].unsqueeze(0)).squeeze(0)

pred_mask = ((pred - pred.min())/(pred.max() - pred.min())*255.).to(torch.uint8).squeeze(0).numpy()
resized_input_img = input_img.resize(pred_mask.shape[::-1])

print(f"type: {type(pred)}, shape: {pred.shape}")
if Path(args.output_path).is_dir() is False:
    Path(args.output_path).mkdir(parents=True, exist_ok=True)


# Convert the grayscale image to a color heatmap
heatmap = cm.jet(pred_mask)
heatmap_rgb = np.delete(heatmap, 3, 2)*255.


# Save the color heatmap image
heatmap_img = PIL.Image.fromarray(heatmap_rgb.astype(np.uint8))
#PIL.Image.fromarray((heatmap_rgb * 255).astype(np.uint8)).save(Path(args.output_path)/f"pred_mask_{Path(args.test_img_path).stem}.png")


# Resize all three images to the original size
heatmap_img = heatmap_img.resize(original_size)
resized_input_img = resized_input_img.resize(original_size)
blended_image = PIL.Image.blend(heatmap_img, resized_input_img, ALPHA)

# Save the blended image
blended_image.save(Path(args.output_path)/f"blended_image_{Path(args.test_img_path).stem}.png")

# Save resized input image
# resized_input_img.save(Path(args.output_path)/f"resized_input_{Path(args.test_img_path).stem}.png")

composite_image = PIL.Image.fromarray(np.hstack((np.array(resized_input_img), np.array(heatmap_img), np.array(blended_image)))) 
composite_image.save(Path(args.output_path)/f"composite_image_{Path(args.test_img_path).stem}.png")