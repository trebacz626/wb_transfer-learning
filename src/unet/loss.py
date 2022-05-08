import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.cel = nn.CrossEntropyLoss()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        cel = self.cel(y_pred, y_true)
        return cel
        # y_pred = y_pred[:, 0].contiguous().view(-1)
        # y_true = y_true[:, 0].contiguous().view(-1)
        # intersection = (y_pred * y_true).sum()
        # dsc = (2.0 * intersection + self.smooth) / (
        #     y_pred.sum() + y_true.sum() + self.smooth
        # )
        # return cel + 1.0 - dsc
