from models.ext_unet import UNet
from models.helpers import init_weights
from datasets.helpers import VOC12_PIXEL_WEIGHTLIST, get_file_paths, save_json_filelist
from datasets.VOC import VOCSegmentationDataset
from transforms.transforms import transform, val_transform
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
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

args = parser.parse_args()

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

    writer = SummaryWriter()

    if os.path.exists(output_path) is False:
        os.mkdir(output_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    unet = UNet(weights=torchvision.models.VGG16_BN_Weights.IMAGENET1K_V1, out_channels=out_channels).to(device)

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
    else:
        unet.apply(init_weights)
        print("No specified weights path, model was xavier initialised")

    mask_paths = get_file_paths(os.path.join(data_root, 'SegmentationClass')) 
    dir_jpg = os.path.join(data_root, 'JPEGImages') 
    file_names = [os.path.splitext(os.path.basename(path))[0] for path in mask_paths]
    image_paths = [os.path.join(dir_jpg, name + ".jpg") for name in file_names]

    train_list_path = os.path.join(partition_folder, 'train.txt')
    val_list_path = os.path.join(partition_folder, 'val.txt')

    if repartition_set:
        image_train, image_val, mask_train, mask_val = train_test_split(image_paths, mask_paths, test_size=0.2, random_state=42)
        save_json_filelist(image_train, train_list_path)
        save_json_filelist(image_val, val_list_path)
    elif os.path.isfile(train_list_path) and os.path.isfile(val_list_path):
        print(f"Loading partition stored at {partition_folder}")
        with open(os.path.join(partition_folder), "r") as fp:
            filelist = json.load(fp)
            image_train = [os.path.join(dir_jpg, img_file) for img_file in filelist]
            mask_train = [os.path.join(mask_paths, file[:-3]+'png') for file in filelist]

    train_dataset = VOCSegmentationDataset(image_train, mask_train, crop_size=input_size, transform=transform(input_size))
    val_dataset = VOCSegmentationDataset(image_val, mask_val, crop_size=input_size, transform=val_transform(input_size))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = Adam(unet.parameters(), lr=lr, weight_decay=wd)
    scheduler = ExponentialLR(optimizer, gamma=0.9, verbose=True)

    loss_weights=torch.Tensor(VOC12_PIXEL_WEIGHTLIST)
    criterion = nn.CrossEntropyLoss(weight=loss_weights).to(device)
    print(f"Loss function: Applied category pixel weights = {loss_weights}")

    best_iou = 0.0
    jaccard = JaccardIndex(task='multiclass', num_classes=out_channels).to(device)

    for epoch in range(num_epochs):
        unet.train()  

        for idx, (images, masks) in enumerate(tqdm(train_loader)):
            iter = idx + batch_size * epoch
            images = images.to(device)
            optimizer.zero_grad()
            outputs = unet(images)
            masks = masks.to(device)
            loss = criterion(outputs, masks)
            writer.add_scalar("train. loss (iter)", loss, iter)
            if idx%50 == 0:
                print(f"Training loss (iter {idx}) = {loss}")
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
        
        scheduler.step()
    writer.flush()

if __name__ == '__main__':
    main(args)