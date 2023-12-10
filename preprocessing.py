import os

import cv2
import numpy as np
from torchvision.transforms import transforms, v2

IMG_WIDTH = 180
IMG_HEIGHT = 192


def data_aug(img, labels, photometric_only=False):
    # Photometric
    res = v2.Grayscale(num_output_channels=3)(img)
    yield res, labels

    res = v2.ColorJitter()(img)
    yield res, labels

    if not photometric_only:
        res = v2.RandomPerspective()(img)
        yield res, labels
        res = v2.RandomPerspective(distortion_scale=0.2)(img)
        yield res, labels

        res = v2.RandomRotation(degrees=(0, 180), expand=True)(img)
        res = v2.Resize((IMG_HEIGHT, IMG_WIDTH))(res)
        yield res, labels

    yield img, labels


def load_data(data_dir):
    """
    Load image data from directory `data_dir/files`.
    Load labels from file `data_dir/labels.txt`

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory `labels` should be a list of 
    integer labels, representing whether the person in image is smiling
    """
    images = []
    labels = []
    data_path = os.path.join(data_dir, "files")
    with open(os.path.join(data_dir, "labels.txt")) as label_file:
        for image in os.listdir(data_path):
            img = cv2.imread(os.path.join(data_path, image))

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, dsize=(IMG_WIDTH, IMG_HEIGHT))
            res = transforms.ToTensor()(img)

            line_data = int(
                label_file.readline().split(" ")[0])  # convert labels to int

            # TODO: add multi-output
            for img_new, label in data_aug(res,
                                           line_data,
                                           photometric_only=False):
                img_new = np.array(img_new)
                images.append(img_new)
                labels.append(label)

    return (images, labels)
