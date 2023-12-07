from datetime import date
import torchvision
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
import os
import sys
import torchvision.transforms as transforms

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

EPOCHS = 60
IMG_WIDTH = 180
IMG_HEIGHT = 192
TEST_SIZE = 0.3
BATCH_SIZE = 64


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python face.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    # TODO: don't use scikit, use torch, and keep the tensor format
    x_train, x_test, y_train, y_test = train_test_split(np.array(images),
                                                        np.array(labels),
                                                        test_size=TEST_SIZE)

    # Convert data to PyTorch tensors and normalize
    print(f"Using cuda: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        dev = "cuda"
    else:
        dev = "cpu"

    x_train = torch.tensor(x_train).float().to(dev)
    y_train = torch.tensor(y_train).float().unsqueeze(1).to(dev)
    x_test = torch.tensor(x_test).float().to(dev)
    y_test = torch.tensor(y_test).float().unsqueeze(1).to(dev)

    # Create dataloaders
    train_dataloader = DataLoader(
        TensorDataset(x_train, y_train),
        # shuffle=True,
        batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(TensorDataset(x_test, y_test),
                                 batch_size=BATCH_SIZE)

    writer = SummaryWriter('runs/face-smile')

    # sample some data to writer
    # get some random training images
    dataiter = iter(train_dataloader)
    images, labels = next(dataiter)

    # create grid of images
    img_grid = torchvision.utils.make_grid(images, normalize=True)

    # write to tensorboard
    writer.add_image('one_batch', img_grid)

    # Get a compiled neural network
    model = get_model().to(dev)

    # Define the loss function
    criterion = nn.BCELoss()

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.000001)

    # Fit model on training data
    with tqdm(range(EPOCHS)) as pbar:
        for epoch in range(EPOCHS):
            model.train()
            pbar.update()
            running_loss = 0.0
            for inputs, targets in train_dataloader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            average_loss = running_loss / len(train_dataloader)

            # Evaluate neural network performance
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in test_dataloader:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            average_val_loss = val_loss / len(test_dataloader)
            pbar.set_postfix(epoch=epoch,
                             loss=f"{average_loss:.4f}",
                             val_loss=f'{average_val_loss:.4f}')

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
            transform = transforms.ToTensor()
            res = transform(img)
            res = np.array(res)
            # Add image
            images.append(res)
            # Add line
            line_data = int(
                label_file.readline().split(" ")[0])  # convert labels to int
            labels.append(line_data)

    return (images, labels)


class SmilingClassifier(nn.Module):

    def __init__(self):
        super(SmilingClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(64 * 45 * 48, 512)
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 45 * 48)
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = SmilingClassifier()
    return model


if __name__ == "__main__":
    main()
