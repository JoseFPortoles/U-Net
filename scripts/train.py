from models.ext_unet import UNet
from models.helpers import init_weights
from datasets.helpers import VOC12_PIXEL_WEIGHTLIST, get_file_paths
from datasets.VOC import VOCSegmentationDataset
from transforms.transforms import transform, val_transform
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchmetrics import JaccardIndex
from sklearn.model_selection import train_test_split
import os
import argparse
import tqdm

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

    image_train, image_val, mask_train, mask_val = train_test_split(image_paths, mask_paths, test_size=0.2, random_state=42)

    train_dataset = VOCSegmentationDataset(image_train, mask_train, crop_size=input_size, transform=transform(input_size))
    val_dataset = VOCSegmentationDataset(image_val, mask_val, crop_size=input_size, transform=val_transform(input_size))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(unet.parameters(), lr=lr, weight_decay=wd)

    loss_weights=torch.Tensor(VOC12_PIXEL_WEIGHTLIST)
    criterion = nn.CrossEntropyLoss(weight=loss_weights).to(device)
    print(f"Loss function: Applied category pixel weights = {loss_weights}")

    best_iou = 0.0
    jaccard = JaccardIndex(task='multiclass', num_classes=out_channels).to(device)

    for epoch in range(num_epochs):
        unet.train()  

        for idx, (images, masks) in enumerate(tqdm(train_loader)):
            images = images.to(device)
            optimizer.zero_grad()
            outputs = unet(images)
            masks = masks.to(device)
            loss = criterion(outputs, masks)
            if idx%50 == 0:
                print(f"Training loss (iter {idx}) = {loss}")
            loss.backward()
            optimizer.step()

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

            if val_iou >= best_iou:
                best_iou = val_iou
                torch.save(unet.state_dict(), os.path.join(output_path, 'best_model.pth'))

        # Imprimir métricas de entrenamiento y validación
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")