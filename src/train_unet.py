# !/bin/python3
# %%
from data import HeartDataset
from unet.unet import UNet, UNetAlt
from torch import optim
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from unet.evaluate import evaluate
import torch.nn.functional as F
import wandb
import logging
import PIL
import numpy as np
from unet.loss import DiceLoss

# %%
lossfn = DiceLoss()
learning_rate = 1e-5
epochs = 150
batch_size = 16
val_percent = 0.1
save_checkpoint = True
amp = True
dataset = HeartDataset(
    "/home/mtizim/programming/studia/wbtl/wb_transfer-learning/data/affregcommon2mm_roi_ct_train/"
)
n_val = int(len(dataset) * val_percent)
n_train = len(dataset) - n_val

experiment = wandb.init(project="unet", resume="allow", anonymous="must")
experiment.config.update(
    dict(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        val_percent=val_percent,
        save_checkpoint=save_checkpoint,
        amp=amp,
    )
)

logging.info(
    f"""Starting training:
    Epochs:          {epochs}
    Batch size:      {batch_size}
    Learning rate:   {learning_rate}
    Training size:   {n_train}
    Validation size: {n_val}
    Checkpoints:     {save_checkpoint}
    Device:          {"cuda"}
    Mixed Precision: {amp}
"""
)


# %%
unet = UNetAlt()
unet.to("cuda")

optimizer = optim.Adam(unet.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=2)
grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
global_step = 0

# %%

train_set, val_set = random_split(
    dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0)
)

# %%
loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
train_loader = DataLoader(train_set, shuffle=True, **loader_args)
val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

# %%
device = "cuda"
for epoch in range(1, epochs + 1):
    unet.train()
    epoch_loss = 0
    with tqdm(total=n_train, desc=f"Epoch {epoch}/{epochs}", unit="img") as pbar:
        for images, true_masks in train_loader:
            optimizer.zero_grad(set_to_none=True)

            images = images.to(device=device, dtype=torch.float32)
            true_masks = 1.0 - true_masks.to(device=device, dtype=torch.float32)

            with torch.cuda.amp.autocast(enabled=True):
                masks_pred = unet(images)
                loss = lossfn(
                    masks_pred,
                    true_masks,
                )

            # loss.backward()
            # optimizer.step()
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

                val_score = evaluate(unet, val_loader, device)

                logging.info("Validation Dice loss: {}".format(val_score))
                masks = {}

                # just two masks for brevity
                for i in range(1, 3):
                    tim = [
                        PIL.Image.fromarray(
                            np.uint8(
                                (true_masks[n][i].float().cpu().detach() > 0.5).numpy()
                                * 254.0
                            ),
                            mode="L",
                        )
                        for n in range(5)
                    ]
                    pim = [
                        PIL.Image.fromarray(
                            np.uint8(
                                (masks_pred[n].float().cpu()[i].detach() > 0.5).numpy()
                                * 254.0
                            ),
                            mode="L",
                        )
                        for n in range(5)
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
