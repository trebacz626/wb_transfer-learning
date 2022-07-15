import torch
from torch import nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, softmax=True, one_hot_encode=False, smooth=1):

        if softmax:
            inputs = F.softmax(inputs, dim=1)
        if one_hot_encode:
            max_idx = torch.argmax(inputs, 1, keepdim=True)
            inputs.zero_()
            inputs.scatter_(1, max_idx, 1)

        intersection = (inputs * targets).sum(dim=(2,3))
        dice = (2.*intersection + smooth)/(inputs.sum(dim=(2,3)) + targets.sum(dim=(2,3)) + smooth)

        return 1 - dice.mean()
