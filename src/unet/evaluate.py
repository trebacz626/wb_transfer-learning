import torch
import torch.nn.functional as F
from tqdm import tqdm

from unet.loss import DiceLoss


def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    lossfn = DiceLoss()
    # iterate over the validation set
    for images, true_masks in tqdm(
        dataloader,
        total=num_val_batches,
        desc="Validation round",
        unit="batch",
        leave=False,
    ):

        # move images and labels to correct device and type
        images = images.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            masks_pred = net(images)

            dice_score += lossfn(
                masks_pred,
                true_masks,
            )

    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches
