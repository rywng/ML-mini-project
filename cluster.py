# Unsupervised learning, with image clustering

import argparse
import torch
from torch import zeros
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader, TensorDataset
from model_utils import resnet50
from cuda_utils import get_least_used_gpu

from preprocessing import load_data

PCA_REDUCED = 50
TSNE_REDUCED = 2


def get_pca(embeddings: np.ndarray, show=True) -> np.ndarray:
    # Dimensionality reduction
    pca = PCA(n_components=PCA_REDUCED)
    fitted = pca.fit_transform(embeddings)
    print(f"PCA reduced: {fitted.shape}")

    if show:
        # plot pca
        fig = plt.figure(1)
        ax = fig.add_subplot(projection="3d")
        ax.scatter(
            fitted[:, 0],
            fitted[:, 1],
            fitted[:, 2],
        )
        ax.set_title("First three PCA dimensions")
        ax.set_xlabel("1st Eigenvector")
        ax.set_ylabel("2nd Eigenvector")
        ax.set_zlabel("3rd Eigenvector")
        plt.show()

    return fitted


def get_tsne(embeddings: np.ndarray, show=True) -> np.ndarray:
    # TSNE for more accurate reduction and visualization
    tsne = TSNE(n_components=TSNE_REDUCED)
    fitted = tsne.fit_transform(embeddings)

    if show:
        fig = plt.figure(2)
        ax = fig.add_subplot()
        ax.scatter(fitted[:, 0], fitted[:, 1])
        ax.set_title("TSNE visualization")
        ax.set_xlabel("1st Eigenvector")
        ax.set_ylabel("2st Eigenvector")
        plt.show()

    return fitted


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
    print(f"embeddings: {embeddings.shape}")

    # plot pca
    pca = get_pca(embeddings)

    # plot TSNE
    tsne = get_tsne(pca)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="cluster.py")
    parser.add_argument(
        "dataset_dir",
        help=
        "Directory of dataset, images within ${dataset_dir}/files will be used"
    )
    args = parser.parse_args()
    main(args)
