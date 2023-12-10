import torch
from torch import nn
from torchvision import models
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from logger_utils import plot_classes_preds


class Classifier(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def get_output(self, input):
        return self(input)

    def train_model(self, train_dataloader: DataLoader,
                    test_dataloader: DataLoader, epochs: int, dev: str,
                    writer: SummaryWriter, name: str) -> nn.Module:
        model = self.to(dev)
        sample_input, sample_target = next(iter(test_dataloader))

        # Define the loss function
        criterion = nn.BCELoss()

        # Define the optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.000001)

        # Fit model on training data
        with tqdm(range(epochs)) as pbar:
            for epoch in range(epochs):
                model.train()

                running_loss = 0.0
                average_loss = 0.0
                for i, data in enumerate(train_dataloader):
                    inputs, targets = data
                    outputs = self.get_output(inputs)
                    loss = criterion(outputs, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    if i == len(train_dataloader) - 1:
                        average_loss = running_loss / len(train_dataloader)
                        # TODO: this code is shit
                        eval_loss, accuracy = self.eval_model(
                            test_dataloader, criterion)
                        writer.add_scalars("Training stats", {
                            "Training loss": average_loss,
                            "Test loss": eval_loss,
                        },
                                           global_step=epoch + 1)
                        writer.add_scalar("Accuracy", accuracy, epoch + 1)
                        writer.add_figure(
                            "predictions vs. actuals",
                            plot_classes_preds(model, sample_input,
                                               sample_target), epoch + 1)
                        running_loss = 0.0
                        torch.save(model.state_dict(),
                                   f"runs/{name}/{epoch}.pt")

                pbar.update()
                pbar.set_postfix(
                    epoch=epoch,
                    loss=f"{average_loss:.4f}",
                )

        return model

    def eval_model(self, test_dataloader: DataLoader, criterion):
        # Evaluate neural network performance
        self.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_dataloader:
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        average_val_loss = val_loss / len(test_dataloader)
        accuracy = 100 * correct / total
        return average_val_loss, accuracy


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


class Resnet(Classifier):

    def __init__(self) -> None:
        super(Resnet, self).__init__()
        self = models.resnet50(progress=True,
                               weights=models.ResNet50_Weights.DEFAULT)

    def get_output(self, input):
        return nn.Linear(1000, 1)(super().get_output(input))
