import torch
from torch import nn

class Classifier(nn.Module):
    pass

class SmilingClassifier(Classifier):

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
