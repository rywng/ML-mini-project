from datetime import date
import os
import sys

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

EPOCHS = 50
IMG_WIDTH = 128
IMG_HEIGHT = 128
# NUM_CATEGORIES = 43
# NUM_CATEGORIES = 2
TEST_SIZE = 0.4
BATCH_SIZE = 64


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python face.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    # labels = keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(np.array(images),
                                                        np.array(labels),
                                                        test_size=TEST_SIZE)

    # Convert data to PyTorch tensors and normalize
    x_train = torch.tensor(x_train).float()
    y_train = torch.tensor(y_train).float().unsqueeze(1)
    x_test = torch.tensor(x_test).float()
    y_test = torch.tensor(y_test).float().unsqueeze(1)

    # Create dataloaders
    train_dataloader = DataLoader(TensorDataset(x_train, y_train),
                                  batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(TensorDataset(x_test, y_test),
                                 batch_size=BATCH_SIZE)

    # Get a compiled neural network
    model = get_model()

    # Define the loss function
    criterion = nn.BCELoss()

    # Define the optimizer
    optimizer = optim.Adam(model.parameters())

    # Fit model on training data
    for epoch in range(EPOCHS):
        for inputs, targets in train_dataloader:
            print(f"epoch: {epoch}", end="")
            optimizer.zero_grad()
            outputs = model(inputs)
            print(len(outputs))
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            print("loss: {loss}")

    # Evaluate neural network performance
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    print(
        f"Accuracy of the network on the test images: {100 * correct / total}")

    # Save model to file
    if len(sys.argv) == 3:
        filename = f"{date.today()}-{EPOCHS}.pt"
        torch.save(model.state_dict(), filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions 3 x IMG_WIDTH x IMG_HEIGHT. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []
    data_path = os.path.join(data_dir, "files")
    with open(os.path.join(data_dir, "labels.txt")) as label_file:
        for image in os.listdir(data_path):
            img = cv2.imread(os.path.join(data_path, image))
            res = cv2.resize(img, dsize=(IMG_WIDTH, IMG_HEIGHT))
            # Convert image to PyTorch format (channel, height, width)
            res = np.transpose(res, (2, 0, 1))
            # Add image
            images.append(res)
            # Add line
            line_data = int(
                label_file.readline().split(" ")[0])  # convert labels to int
            labels.append(line_data)

    return (images, labels)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = Net()
    return model


if __name__ == "__main__":
    main()
