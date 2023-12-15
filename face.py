import argparse
import os
import shutil

import torch
from torch.utils.tensorboard.writer import SummaryWriter

from logger_utils import plot_random_batch
from model_utils import SmilingClassifier, resnet50, train_model
from preprocessing import get_dataloaders

TEST_SIZE = 0.3
# Don't change this!
BATCH_SIZE = 64

config = {
    "resnet50-face-smile": [resnet50.get_resnet_smile(), False],
    "simple-face-smile": [SmilingClassifier(), False],
    "resnet50-position": [resnet50.get_resnet_pos(), True],
}


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
        args.dataset_location, dev, photometric_only=config[args.model][1])

    writer = SummaryWriter('runs/face-smile')

    writer.add_figure("One batch",
                      plot_random_batch(train_dataloader, BATCH_SIZE))

    net = config[args.model][0]

    save_dir = os.path.join(os.path.dirname(__file__), "runs", args.model)

    train_model(net, (train_dataloader, test_dataloader),
                int(args.epochs),
                dev,
                writer,
                save_dir=save_dir,
                pose=config[args.model][1],
                nosave=args.nosave)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="pipeline",
                                     description="Train the model")
    parser.add_argument("dataset_location")
    parser.add_argument("-m",
                        "--model",
                        default="resnet50-face-smile",
                        choices=config.keys())
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
