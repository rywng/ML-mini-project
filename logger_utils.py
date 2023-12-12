import matplotlib.pyplot as plt
import numpy as np
from torch.functional import Tensor
from torch.utils.data import DataLoader

FIGSIZE = (12, 12)


def matplotlib_imshow(img, one_channel=False):
    img = Tensor.cpu(img)
    if one_channel:
        img = img.mean(dim=0)
    # img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def plot_classes_preds(net, images, labels):
    """Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along with
    its probability, alongside the actual label, coloring this information
    based on whether the prediction was correct or not.

    Uses the "images_to_probs" function.
    """
    RANGE = min(12, len(labels))
    ROWS = 3
    images = images[:RANGE]
    labels = labels[:RANGE]
    probs = net(images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=FIGSIZE)
    for idx in np.arange(RANGE):
        prob = probs[idx].detach().to("cpu").numpy()
        label = labels[idx].detach().to("cpu").numpy()
        ax = fig.add_subplot(ROWS,
                             RANGE // ROWS,
                             idx + 1,
                             xticks=[],
                             yticks=[])
        matplotlib_imshow(images[idx], one_channel=False)
        ax.set_title(f"{prob}%, {label}%\n")
    return fig


def plot_random_batch(train_dataloader: DataLoader, batch_size: int):
    # sample some data to writer
    # get some random training images
    dataiter = iter(train_dataloader)
    image_samples, label_samples = next(dataiter)

    # write to tensorboard
    fig = plt.figure(figsize=FIGSIZE)
    for i in np.arange(batch_size):
        ax = fig.add_subplot(8, batch_size // 8, i + 1, xticks=[], yticks=[])
        matplotlib_imshow(image_samples[i])
        sample = label_samples[i]
        if len(sample) > 1:
            # TODO: this is broken rn
            sample = list(map(float, sample))
            sample = list(round(i, 4) for i in sample)
            ax.set_title(str(sample))
        else:
            ax.set_title(str(int(sample)))

    return fig
