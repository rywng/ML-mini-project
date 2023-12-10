import argparse

import torch
from torch.utils.tensorboard.writer import SummaryWriter

from logger_utils import plot_random_batch
from models import Resnet
from preprocessing import get_dataloaders

TEST_SIZE = 0.3
# Don't change this!
BATCH_SIZE = 64


def main(args):
    # Check command-line arguments

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
    net = Resnet()
    net.train_model(train_dataloader,
                    test_dataloader,
                    int(args.epochs),
                    dev,
                    writer,
                    name="resnet50")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="pipeline",
                                     description="Train the model")
    parser.add_argument("dataset_location")
    parser.add_argument("-e", "--epochs", default="300")
    parser.add_argument("--no-cuda",
                        action="store_true",
                        dest="nocuda",
                        help="Don't use cuda for training")

    args = parser.parse_args()
    main(args)
