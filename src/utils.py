from matplotlib import pyplot as plt
import torch


def plot_image_and_masks(X, y, number_in_batch=0):
    channels = y.shape[1]
    fig, axs = plt.subplots(1, channels + 1)
    fig.set_figheight(20)
    fig.set_figwidth(20)
    axs[0].imshow(X[number_in_batch, 0])
    for i in range(channels):
        axs[i + 1].imshow(y[number_in_batch, i])
