import argparse

from sklearn import os
import torch

from face import get_dev
from logger_utils import plot_roc_graph
from model_utils import model_config
from preprocessing import get_dataloaders


def main(args):
    _, test_dataloader = get_dataloaders(args.dataset_location, "cpu",
                                         photometric_only=model_config.config[args.model][1])
    net = model_config.config[args.model][0]

    try:
        weight_path = os.path.join("runs", args.model)
        weight_path = os.path.join(weight_path, "best.pt")
        net.load_state_dict(torch.load(weight_path))
    except FileNotFoundError:
        print("The file is not present")
        exit(1)
    plot_roc_graph(net, test_dataloader, print=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the model with user-specified input")
    parser.add_argument("dataset_location")
    parser.add_argument("-m",
                        "--model",
                        default="resnet50-face-smile",
                        choices=model_config.config.keys())
    args = parser.parse_args()
    main(args)
