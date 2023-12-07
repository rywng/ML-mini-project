# imports
import matplotlib.pyplot as plt
import numpy as np
from torch.functional import Tensor

# datasets
# helper functions


def matplotlib_imshow(img, one_channel=False):
    img = Tensor.cpu(img)
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    RANGE = 8
    images = images[:RANGE]
    labels = labels[:RANGE]
    probs = net(images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(RANGE):
        ax = fig.add_subplot(2, RANGE // 2, idx + 1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=False)
        ax.set_title(
            f"{float(probs[idx] * 100.0):.2f}%, {float(labels[idx] * 100):.2f}%\n"
            .format(color=("green" if abs(probs[idx] -
                                          labels[idx]) < 0.4 else "red")))
    # fig.show()
    # __import__('pdb').set_trace()
    return fig
