import argparse

from sklearn import os
import torch

from face import get_dev
from logger_utils import (
    plot_head_pose,
    plot_pr_graph,
    plot_roc_graph,
    print_confusion_matrix,
)
from model_utils import model_config
from preprocessing import get_dataloaders


def main(args):
    _, test_dataloader = get_dataloaders(
        args.dataset_location,
        "cpu",
        photometric_only=model_config.config[args.model][1])
    net = model_config.config[args.model][0]

    if args.savefile:
        weight_path = args.savefile
    else:
        weight_path = os.path.join("runs", args.model, "best.pt")

    try:
        net.load_state_dict(torch.load(weight_path, get_dev(args)))
    except FileNotFoundError:
        print("The file is not present")
        print(f"Path: {weight_path}")
        exit(1)
    if not model_config.config[args.model][1]:
        plot_roc_graph(net, test_dataloader)
        plot_pr_graph(net, test_dataloader)
        print_confusion_matrix(net, test_dataloader)
    else:
        plot_head_pose(test_dataloader, model=net)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the model with user-specified input")
    parser.add_argument("dataset_location")
    parser.add_argument("--no-cuda", action="store_true", dest="nocuda")
    parser.add_argument("-m",
                        "--model",
                        default="resnet50-face-smile",
                        choices=model_config.config.keys())
    parser.add_argument("-s", "--savefile", required=False)
    args = parser.parse_args()
    main(args)
