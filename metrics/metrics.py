import torch
import torch.nn.functional as F

def iou(pred_masks, true_masks):
    masks_one_hot = F.one_hot(true_masks.to(torch.int64), num_classes=22)
    intersection = torch.logical_and(pred_masks, masks_one_hot.permute(0,3,1,2)).sum(dim=(0,1,2))
    union = torch.logical_or(pred_masks, masks_one_hot.permute(0,3,1,2)).sum(dim=(0,1,2))
    iou = intersection.float() / union.float()
    return iou.mean().item()