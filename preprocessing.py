import os

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import transforms, v2

IMG_WIDTH = 180
IMG_HEIGHT = 192


def data_aug(x_train, y_train, photometric_only=False):
    x_new = []
    y_new = []
    for i in range(len(x_train)):
        for res in generate_aug_data(torch.tensor(x_train[i]),
                                     photometric_only=photometric_only):
            x_new.append(res)
            y_new.append(y_train[i])
    return np.array(x_new), np.array(y_new)


def generate_aug_data(img, photometric_only=False):

    def rand():
        return torch.rand((1, 1))

    # Photometric
    if rand() > 0.7:
        res = v2.Grayscale(num_output_channels=3)(img)
        yield res

    if rand() > 0.7:
        res = v2.ColorJitter(.3, .3, .3, .2)(img)
        yield res

    res = img
    if rand() > 0.8:
        res = v2.RandomPosterize(2, 1)(img)
        yield res

    if rand() > 0.6:
        res = v2.RandomAutocontrast(1)(res)
        yield res

    if rand() > 0.6:
        res = v2.RandomEqualize(1)(res)
        yield res

    if not photometric_only:
        if rand() > 0.5:
            res = v2.RandomPerspective(0.5, p=1)(res)
            yield res

        if rand() > 0.7:
            res = v2.RandomPerspective(p=1)(img)
            yield res

        if rand() > 0.6:
            res = v2.RandomRotation(degrees=(-90, 90), expand=True)(img)
            res = v2.Resize((IMG_HEIGHT, IMG_WIDTH), antialias=True)(res)
            yield res

    yield img


def load_data(data_dir, photometric_only=False):
    """Load image data from directory `data_dir/files`. Load labels from file
    `data_dir/labels.txt`

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

            if not photometric_only: # smile
                line_data = int(label_file.readline().split()
                                [0])  # convert labels to int
            else: # pose
                line_data = label_file.readline().split()[1:]
                line_data = torch.tensor(list(map(float, line_data)), dtype=torch.float64)

            images.append(res)
            labels.append(line_data)

    return (images, labels)


def get_dataloaders(data_dir,
                    dev,
                    batch_size=64,
                    test_size=0.3,
                    photometric_only=False):
    # Get image arrays and labels for all image files
    image_samples, label_samples = load_data(data_dir, photometric_only=photometric_only)

    # Split data into training and testing sets
    # TODO: don't use scikit, use torch, and keep the tensor format
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(image_samples), np.array(label_samples), test_size=test_size)

    x_train, y_train = data_aug(x_train, y_train, photometric_only=photometric_only)

    # Convert data to PyTorch tensors and normalize
    print(f"Using cuda: {torch.cuda.is_available()}")
    x_train = torch.tensor(x_train).float().to(dev)
    y_train = torch.tensor(y_train).float().to(dev)
    x_test = torch.tensor(x_test).float().to(dev)
    y_test = torch.tensor(y_test).float().to(dev)


    # Create dataloaders
    train_dataloader = DataLoader(TensorDataset(x_train, y_train),
                                  shuffle=True,
                                  batch_size=batch_size)
    test_dataloader = DataLoader(TensorDataset(x_test, y_test),
                                 batch_size=batch_size)
    return train_dataloader, test_dataloader
