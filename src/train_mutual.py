import time
from collections import defaultdict

import torch
import wandb


from models.mutual_model import MutualModel
from options.train_options import TrainOptions
from data.heart_mutual_dataset import HeartMutualDataset
from data.heart_mutual_valid_dataset import HeartMutualValidDataset
from visualizer.visualizer import Visualizer

if __name__ == '__main__':
    experiment = wandb.init(project="mutual", resume="allow")
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

    model = MutualModel(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    total_iters = 0                # the total number of training iterations
    # model.load_networks_and_optimizers(10)

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in enumerate(train_dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                visuals = model.get_current_visuals()
                visualizer.log_images(visuals)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                experiment.log(losses)
                print({"epoch":epoch, "epoch_iter":epoch_iter})


            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()


        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks_and_optimizers('latest')
            model.save_networks_and_optimizers(epoch)

            with torch.no_grad():
                model.eval()
                valid_scores = defaultdict(list)
                for i, data in enumerate(valid_dataset):
                    model.set_input(data)
                    scores = model.validation_scores()
                    for key in scores:
                        valid_scores[key].append(scores[key])
                for key in valid_scores:
                    valid_scores[key] = sum(valid_scores[key])/len(valid_scores[key])
                experiment.log(valid_scores)
                model.train()

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
