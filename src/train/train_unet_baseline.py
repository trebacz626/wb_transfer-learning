# %%
import os

os.chdir("/home/mtizim/studia/s6/wbtl/wb_transfer-learning/src")
from data.heart_mutual_dataset import HeartMutualDataset
from data.heart_mutual_valid_dataset import HeartMutualValidDataset
from torch import optim
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp
from models.unet import UNet
import torch.nn.functional as F
import wandb
import logging
import PIL
import numpy as np
from losses.DiceLoss import DiceLoss

# %%
def dataselector(data):
    return data["T_scan"], data["T_labels"]


# %%

lossfn = DiceLoss()
learning_rate = 1e-4
epochs = 150
batch_size = 32
val_percent = 0.1
save_checkpoint = True
amp = True


experiment = wandb.init(project="unet", resume="allow", anonymous="must")
experiment.config.update(
    dict(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        save_checkpoint=save_checkpoint,
        amp=amp,
    )
)
# %%
class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


opt = dotdict({"load_size": 96, "no_flip": False})
dataset = HeartMutualDataset(opt)
datasetval = HeartMutualValidDataset(opt)

logging.info(
    f"""Starting training:
    Epochs:          {epochs}
    Batch size:      {batch_size}
    Learning rate:   {learning_rate}
    Training size:   {len(dataset)}
    Validation size: {len(datasetval)}
    Checkpoints:     {save_checkpoint}
    Device:          {"cuda"}
    Mixed Precision: {amp}
"""
)


# %%
unet = smp.Unet(
    encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
    activation="sigmoid",
    in_channels=1,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=8,  # model output channels (number of classes in your dataset)
)
# unet = UNet(in_channels=1, out_channels=1, init_features=32)
unet.to("cuda")

optimizer = optim.Adam(unet.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=2)
grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
global_step = 0


# %%
loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
train_loader = DataLoader(dataset, shuffle=True, **loader_args)
val_loader = DataLoader(datasetval, shuffle=False, drop_last=True, **loader_args)
# %%
def evaluate(net, dataloader, device, lossfn, dsel):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    # iterate over the validation set
    for data in tqdm(
        dataloader,
        total=num_val_batches,
        desc="Validation round",
        unit="batch",
        leave=False,
    ):
        images, true_masks = dsel(data)

        # move images and labels to correct device and type
        images = images.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            masks_pred = net(images)

            dice_score += lossfn(masks_pred, true_masks, softmax=False)

    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches


# %%
n_train = len(dataset)
device = "cuda"
for epoch in range(1, epochs + 1):
    unet.train()
    epoch_loss = 0
    with tqdm(total=n_train, desc=f"Epoch {epoch}/{epochs}", unit="img") as pbar:
        for data in train_loader:
            images, true_masks = dataselector(data)
            optimizer.zero_grad(set_to_none=True)

            images = images.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.float32)

            with torch.cuda.amp.autocast(enabled=True):
                masks_pred = unet(images)
                loss = lossfn(masks_pred, true_masks, softmax=False)

            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            pbar.update(images.shape[0])
            global_step += 1
            epoch_loss += loss.item()

            scheduler.step(loss)

            experiment.log({"train loss": loss.item()})
            pbar.set_postfix(**{"loss (batch)": f"{loss.item():16}"})

            if global_step % 250 == 0:

                val_score = evaluate(unet, val_loader, device, lossfn, dataselector)

                logging.info("Validation Dice loss: {}".format(val_score))
                masks = {}

                # just two masks for brevity
                for i in range(0, 2):
                    tim = [
                        PIL.Image.fromarray(
                            np.uint8(
                                (true_masks[n][i].float().cpu().detach()).numpy()
                                * 255.0
                            ),
                            mode="L",
                        )
                        for n in range(3)
                    ]
                    pim = [
                        PIL.Image.fromarray(
                            np.uint8(
                                (masks_pred[n].float().cpu()[i].detach()).numpy()
                                * 255.0
                            ),
                            mode="L",
                        )
                        for n in range(3)
                    ]
                    masks[f"true_{i}"] = [wandb.Image(t) for t in tim]
                    masks[f"pred_{i}"] = [wandb.Image(p) for p in pim]
                experiment.log(
                    {
                        "learning rate": optimizer.param_groups[0]["lr"],
                        "validation Dice": val_score,
                        "masks": masks,
                    }
                )

# %%
torch.save(unet, "../savedmodels/unet/300.pt")

# %%
