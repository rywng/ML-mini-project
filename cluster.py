# Unsupervised learning, with image clustering

import argparse
import torch
from torch import zeros
from sklearn.decomposition import PCA

from torch.utils.data import DataLoader, TensorDataset
from model_utils import resnet50
from cuda_utils import get_least_used_gpu

from preprocessing import load_data


def main(arg: argparse.Namespace):
    print(arg)
    dev = get_least_used_gpu()

    images, _ = load_data(arg.dataset_dir)

    # Convert from list
    images = torch.stack(images).float().to(dev)

    # Extract embeddings
    image_loader = DataLoader(TensorDataset(images, zeros(len(images))))
    model = resnet50.get_resnet_feature(pretrained=True).to(dev)
    embeddings = []
    for image, _ in image_loader:
        out = model(image)
        embeddings += out
    embeddings = torch.stack(embeddings).cpu().detach().numpy()

    # Dimensionality reduction


    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="cluster.py")
    parser.add_argument(
        "dataset_dir",
        help=
        "Directory of dataset, images within ${dataset_dir}/files will be used"
    )
    args = parser.parse_args()
    main(args)
