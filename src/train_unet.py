import time
from collections import defaultdict

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import lr_scheduler

import wandb
from losses.DiceLoss import DiceLoss

from models.unet import UNet
from options.train_options import TrainOptions
from data.heart_mutual_dataset import HeartMutualDataset
from data.heart_mutual_valid_dataset import HeartMutualValidDataset
from visualizer.visualizer import Visualizer


import segmentation_models_pytorch as smp

if __name__ == '__main__':
    experiment = wandb.init(project="unet", resume="allow")
    visualizer = Visualizer(experiment)
    opt = TrainOptions().parse()
    train_dataset = torch.utils.data.DataLoader(
            HeartMutualDataset(opt),
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads))

    valid_dataset = torch.utils.data.DataLoader(
            HeartMutualValidDataset(opt),
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads))

    dataset_size = len(train_dataset)
    print('The number of training images = %d' % dataset_size)

    device = 'cuda'
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1, classes=8
    ).to(device)#UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))
    scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
    criterion_dice = DiceLoss().to(device)
    criterion_ce = CrossEntropyLoss().to(device)
    softmax = nn.Softmax(dim=1).to(device)
    total_iters = 0                # the total number of training iterations

    best_validation_score = 0

    wandb.save(f"./checkpoints/{opt.name}/*")
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        for i, data in enumerate(train_dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            ##HERE do stuff
            optimizer.zero_grad()
            scan, labels = data["T_scan"].to(device), data["T_labels"].to(device)
            pred_labels = model(scan)

            loss = criterion_dice(pred_labels, labels) + criterion_ce(pred_labels, labels)
            loss = loss.mean()
            loss.backward()
            optimizer.step()

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                visualizer.log_images({
                    "scan": scan,
                    "labels":torch.unsqueeze(torch.argmax(labels, dim=1)/3.5 - 1, dim=1),
                    "pred_labels": torch.unsqueeze(torch.argmax(pred_labels, dim=1)/3.5 - 1, dim=1)
                })

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = {"train_combined": loss}
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                experiment.log(losses)
                print({"epoch":epoch, "epoch_iter":epoch_iter}, losses)


            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                # save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                # model.save_networks(save_suffix)

            iter_data_time = time.time()

        scheduler.step()
        if epoch % opt.save_epoch_freq == 0:

            with torch.no_grad():
                model.eval()
                valid_scores = defaultdict(list)
                for i, data in enumerate(valid_dataset):
                    scan, labels = data["T_scan"].to(device), data["T_labels"].to(device)
                    pred_labels = model(scan)
                    scores = {
                        "dice": 1 - criterion_dice(pred_labels, labels),
                        "dice_hard": 1 - criterion_dice(pred_labels, labels, one_hot_encode=True)
                    }
                    for key in scores:
                        valid_scores[key].append(scores[key])
                for key in valid_scores:
                    valid_scores[key] = sum(valid_scores[key])/len(valid_scores[key])
                experiment.log(valid_scores)
                print("Validation Scores:", valid_scores)
                model.train()

            if valid_scores["dice_hard"] > best_validation_score:
                best_validation_score = valid_scores["dice_hard"]
                print('saving the best model at the end of epoch %d, iters %d' % (epoch, total_iters))
                # model.save_networks_and_optimizers('latest')

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
