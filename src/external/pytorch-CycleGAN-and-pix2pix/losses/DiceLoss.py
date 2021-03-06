from torch import nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        inputs = F.softmax(inputs, dim=1)


        intersection = (inputs * targets).sum(dim=(2,3))
        dice = (2.*intersection + smooth)/(inputs.sum(dim=(2,3)) + targets.sum(dim=(2,3)) + smooth)

        return 1 - dice.mean()
