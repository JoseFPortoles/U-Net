import torch
import torch.nn.functional as F 
import os
import numpy as np
import json

VOC12_PIXEL_WEIGHTLIST = {
    'background': 0.000904033501971689, 
    'aeroplane': 0.05580971950578698, 
    'bicycle': 0.15512901475841676, 
    'bird': 0.04740651893772769, 
    'boat': 0.07187987804855628, 
    'bottle': 0.06605527448142, 
    'bus': 0.023431610850140144, 
    'car': 0.03190424653970079, 
    'cat': 0.019373015794332062, 
    'chair': 0.06140854291701378, 
    'cow': 0.039445520421651, 
    'diningtable': 0.046379051274731395, 
    'dog': 0.02286588678863056, 
    'horse': 0.042673821654842, 
    'motorbike': 0.042815967498724766, 
    'person': 0.009404935990145382, 
    'potted plant': 0.08560038227768499, 
    'sheep': 0.04730992453835607, 
    'sofa': 0.03801971776752205, 
    'train': 0.02666702480980015, 
    'tv/monitor': 0.056240339221151625, 
    'contour': 0.009275572421693995
    }

def tensor2img(img_tensor):
    img = img_tensor[0].permute(1,2,0).numpy()
    img = (img-img.min())/(img.max()-img.min())*255.
    return img.astype(np.uint8)

def get_file_paths(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths

def save_json_filelist(pathlist, save_path):
    filelist = [os.path.basename(path) for path in pathlist] 
    with open(save_path, "w") as fp:
        fp.write(str(json.dumps(filelist))) 

class ToOneHotMask(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, mask):
        one_hot_mask = F.one_hot(mask.to(torch.int64).squeeze(0), num_classes=self.num_classes).permute(2,0,1)
        return one_hot_mask.float()