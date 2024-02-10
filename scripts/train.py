from models.ext_unet import UNet
from models.helpers import init_weights, freeze_encoder
from datasets.helpers import VOC12_PIXEL_WEIGHTLIST, get_file_paths, save_json_filelist
from datasets.voc2012 import VOCSegmentationDataset
from transforms.transforms import transform, val_transform
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchmetrics import JaccardIndex
from sklearn.model_selection import train_test_split
import os
import argparse
from tqdm import tqdm
import json


parser = argparse.ArgumentParser(prog='train.py', description='Train U-Net for segmentation task')

parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
parser.add_argument('--wd', type=float, default=1e-3, help='Weight decay.')
parser.add_argument('--input_size', type=int, default=224, help='Lateral size for the (square) input image')
parser.add_argument('--num_segment_categories', type=int, default=22, help='Number of segmentation categories, including background.')
parser.add_argument('--weights_path', type=str, default=None, help='Weights file path. If =None then initialise net w/ Xavier function.')
parser.add_argument('--data_root', type=str, help='Dataset root folder path')
parser.add_argument('--output_path', type=str, default='./checkpoints', help='Folder where trained weights are saved')
parser.add_argument('--repartition_set', action='store_true', help='Repartition dataset')
parser.add_argument('--partition_folder', type=str, )
parser.add_argument('--frozen_encoder', action='store_true', help='Freezes encoder')
parser.add_argument('--add_contour_loss_weight', type=float, default=0, help='Add extra weight to segmentation of contours (See original U-Net article)')

args = parser.parse_args()

def train_loop(num_epochs: int, batch_size: int, lr: float, wd: float, input_size: int, out_channels: int, weights_path: str, data_root: str, output_path: str,
               repartition_set: bool, partition_folder: str, frozen_encoder: bool, extra_contour_w: float):
    
    lr_start = lr

    writer = SummaryWriter()

    if os.path.exists(output_path) is False:
        os.mkdir(output_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    unet = UNet(weights=torchvision.models.VGG16_BN_Weights.IMAGENET1K_V1, out_channels=out_channels).to(device)
    unet.xavier_init_decoder()
    unet = freeze_encoder(unet, freeze=frozen_encoder)

    if weights_path:
        if os.path.exists(weights_path):
            try:
                unet.load_state_dict(torch.load(weights_path), map_location=device)
                print(f"Model was initialised with weights from {weights_path}")
            except Exception as e:
                unet.apply(init_weights)
                print("ERROR: Attempt at loading weights from {weights_path} threw an exception {e}.\nModel was xavier initialised instead!!!")    
        else:
            unet.apply(init_weights)
            print("Specified weights path does not exist, model was xavier initialised")
    
    mask_dir = os.path.join(data_root, 'SegmentationClass')
    mask_paths = get_file_paths(mask_dir) 
    jpg_dir = os.path.join(data_root, 'JPEGImages') 
    file_names = [os.path.splitext(os.path.basename(path))[0] for path in mask_paths]
    image_paths = [os.path.join(jpg_dir, name + ".jpg") for name in file_names]

    train_list_path = os.path.join(partition_folder, 'train.txt')
    val_list_path = os.path.join(partition_folder, 'val.txt')

    if repartition_set:
        image_train, image_val, mask_train, mask_val = train_test_split(image_paths, mask_paths, test_size=0.2, random_state=42)
        save_json_filelist(image_train, train_list_path)
        save_json_filelist(image_val, val_list_path)
    elif os.path.isfile(train_list_path) and os.path.isfile(val_list_path):
        print(f"Loading partition stored at {partition_folder}")
        with open(train_list_path, "r") as fp:
            train_filelist = json.load(fp)
            image_train = [os.path.join(jpg_dir, img_file) for img_file in train_filelist]
            mask_train = [os.path.join(mask_dir, file[:-3]+'png') for file in train_filelist]
        with open(val_list_path, "r") as fp:
            val_filelist = json.load(fp)
            image_val = [os.path.join(jpg_dir, img_file) for img_file in val_filelist]
            mask_val = [os.path.join(mask_dir, file[:-3]+'png') for file in val_filelist]


    train_dataset = VOCSegmentationDataset(image_train, mask_train, crop_size=input_size, transform=transform(input_size))
    val_dataset = VOCSegmentationDataset(image_val, mask_val, crop_size=input_size, transform=val_transform(input_size))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = Adam(unet.parameters(), lr=lr, weight_decay=wd)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, verbose=False)

    loss_weights = [VOC12_PIXEL_WEIGHTLIST[k] + (k == 'contour') * extra_contour_w for k in VOC12_PIXEL_WEIGHTLIST]

    loss_weights=torch.Tensor(loss_weights)
    criterion = nn.CrossEntropyLoss(weight=loss_weights).to(device)
    print(f"Loss function: Applied category pixel weights = {loss_weights}")

    best_iou = 0.0
    jaccard = JaccardIndex(task='multiclass', num_classes=out_channels).to(device)

    for epoch in range(num_epochs):
        last_lr = optimizer.param_groups[0]['lr']
        unet.train()  
        iter_epoch = len(train_loader)
        for idx, (images, masks) in enumerate(tqdm(train_loader)):
            iter = idx +  iter_epoch * epoch
            images = images.to(device)
            optimizer.zero_grad()
            outputs = unet(images)
            masks = masks.to(device)
            loss = criterion(outputs, masks)
            writer.add_scalar("train. loss (iter)", loss, iter)
            writer.add_scalar("lr (iter)", last_lr, iter)
            loss.backward()
            optimizer.step()
        writer.add_scalar("train. loss (epoch)", loss, epoch)

        unet.eval()

        with torch.no_grad():
            val_loss = 0.0
            val_iou = 0.0

            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = unet(images)
                val_loss += criterion(outputs, masks)
                val_iou += jaccard(outputs, masks)

            val_loss /= len(val_loader)
            val_iou /= len(val_loader)
            writer.add_scalar("val. loss (epoch)", val_loss, iter)
            writer.add_scalar("val. IoU (epoch)", val_iou, iter)
            
            if val_iou >= best_iou:
                best_iou = val_iou
                torch.save(unet.state_dict(), os.path.join(output_path, 'best_model.pth'))
        
        scheduler.step(val_iou)
    writer.add_hparams({'lr': lr_start, 'wd': wd, 'batch_size': batch_size, 'extra_contour_w': extra_contour_w, 'frozen_encoder': frozen_encoder},
                       {'last_val_loss': val_loss, 'last_val_iou': val_iou, 'best_val_iou': best_iou})
    writer.flush()

def main(args):
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    lr = args.lr
    wd = args.wd
    input_size = args.input_size
    out_channels = args.num_segment_categories
    weights_path = args.weights_path
    data_root = args.data_root
    output_path = args.output_path
    repartition_set = args.repartition_set
    partition_folder = args.partition_folder
    frozen_encoder = args.frozen_encoder
    extra_contour_w = args.add_contour_loss_weight
    
    train_loop(num_epochs, batch_size, lr, wd, input_size, out_channels, weights_path, data_root, output_path, repartition_set, partition_folder, frozen_encoder, extra_contour_w)


if __name__ == '__main__':
    main(args)