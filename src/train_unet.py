# !/bin/python3
# %%
from data import HeartDataset
from unet.unet import UNet
from torch import optim
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, random_split

learning_rate = 1e-5
epochs = 5
batch_size = 1
val_percent = 0.1
img_scale = 0.5

dataset = HeartDataset("affregcommon2mm_roi_mr_train")
# %%
unet = UNet()

optimizer = optim.RMSprop(
    unet.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9
)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, "max", patience=2
)  # goal: maximize Dice score
grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
criterion = nn.CrossEntropyLoss()
global_step = 0

# %%
n_val = int(len(dataset) * val_percent)
n_train = len(dataset) - n_val
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

            assert images.shape[1] == unet.n_channels, (
                f"Network has been defined with {unet.n_channels} input channels, "
                f"but loaded images have {images.shape[1]} channels. Please check that "
                "the images are loaded correctly."
            )

            images = images.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.long)

            with torch.cuda.amp.autocast(enabled=amp):
                masks_pred = unet(images)
                loss = criterion(masks_pred, true_masks) + dice_loss(
                    F.softmax(masks_pred, dim=1).float(),
                    F.one_hot(true_masks, unet.n_classes).permute(0, 3, 1, 2).float(),
                    multiclass=True,
                )

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            pbar.update(images.shape[0])
            global_step += 1
            epoch_loss += loss.item()
            experiment.log(
                {"train loss": loss.item(), "step": global_step, "epoch": epoch}
            )
            pbar.set_postfix(**{"loss (batch)": loss.item()})

            # Evaluation round
            division_step = n_train // (10 * batch_size)
            if division_step > 0:
                if global_step % division_step == 0:

                    val_score = evaluate(unet, val_loader, device)
                    scheduler.step(val_score)

                    logging.info("Validation Dice score: {}".format(val_score))
                    print(
                        {
                            "learning rate": optimizer.param_groups[0]["lr"],
                            "validation Dice": val_score,
                            "images": wandb.Image(images[0].cpu()),
                            "masks": {
                                "true": wandb.Image(true_masks[0].float().cpu()),
                                "pred": wandb.Image(
                                    torch.softmax(masks_pred, dim=1)
                                    .argmax(dim=1)[0]
                                    .float()
                                    .cpu()
                                ),
                            },
                            "step": global_step,
                            "epoch": epoch,
                            **histograms,
                        }
                    )
