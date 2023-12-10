import argparse
import os
import shutil

import torch
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import models
from torchvision.models.resnet import nn

from logger_utils import plot_random_batch
from model_utils import train_model
from preprocessing import get_dataloaders

TEST_SIZE = 0.3
# Don't change this!
BATCH_SIZE = 64


def main(args):
    if args.clean:
        try:
            shutil.rmtree(os.path.join(os.path.dirname(__file__), "runs"))
        except FileNotFoundError:
            print("Already cleaned, skipping")

    if torch.cuda.is_available() and not args.nocuda:
        dev = "cuda"
    else:
        dev = "cpu"

    train_dataloader, test_dataloader = get_dataloaders(
        args.dataset_location, dev)

    writer = SummaryWriter('runs/face-smile')

    writer.add_figure("One batch",
                      plot_random_batch(train_dataloader, BATCH_SIZE))
    # Train model
    # model = train_model(SmilingClassifier(), train_dataloader, test_dataloader,
    #                     300, dev, writer)
    net = models.resnet50(progress=True,
                          weights=models.ResNet50_Weights.DEFAULT)

    net = nn.Sequential(net, nn.Linear(1000, 1), nn.Sigmoid())

    save_dir = os.path.join(os.path.dirname(__file__), "runs",
                            "resnet50-face-smile")

    train_model(net,
                train_dataloader,
                test_dataloader,
                int(args.epochs),
                dev,
                writer,
                save_dir=save_dir,
                nosave=args.nosave)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="pipeline",
                                     description="Train the model")
    parser.add_argument("dataset_location")
    parser.add_argument("-e", "--epochs", default="50")
    parser.add_argument("--no-cuda",
                        action="store_true",
                        dest="nocuda",
                        help="Don't use cuda for training")
    parser.add_argument("--clean",
                        action="store_true",
                        help="Clean up files in ./runs")
    parser.add_argument("--no-save",
                        dest="nosave",
                        action="store_true",
                        help="Don't save weight file")

    args = parser.parse_args()
    main(args)
