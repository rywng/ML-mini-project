# Unsupervised learning, with image clustering

import argparse

import hdbscan
from hdbscan import HDBSCAN
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
from torch import zeros
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from cuda_utils import get_least_used_gpu
from logger_utils import matplotlib_imshow
from model_utils import resnet50
from preprocessing import load_data

PCA_REDUCED = 80
TSNE_REDUCED = 2
BATCH_SIZE = 64
TEST_RATIO = 0.01


def plot_random_batch(images_indexes: list, images):
    fig = plt.figure()
    batch_size = len(images_indexes)
    for i in images_indexes:
        ax = fig.add_subplot(8, batch_size // 8, i + 1, xticks=[], yticks=[])
        matplotlib_imshow(images[i])
    plt.show()


def plot_cluster(X,
                 labels,
                 probabilities=None,
                 parameters=None,
                 ground_truth=False,
                 ax=None):
    if ax is None:
        _, ax = plt.subplots()
    labels = labels if labels is not None else np.ones(X.shape[0])
    probabilities = probabilities if probabilities is not None else np.ones(
        X.shape[0])
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [
        sns.color_palette("Spectral", as_cmap=True)(each)
        for each in np.linspace(0, 1, len(unique_labels))
    ]
    # The probability of a point belonging to its labeled cluster determines
    # the size of its marker
    proba_map = {idx: probabilities[idx] for idx in range(len(labels))}
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_index = np.where(labels == k)[0]
        for ci in class_index:
            ax.plot(X[ci, 0],
                    X[ci, 1],
                    "x" if k == -1 else "o",
                    markerfacecolor=tuple(col),
                    markeredgecolor="k",
                    markersize=4 if k == -1 else 1 + 5 * proba_map[ci],
                    alpha=0.3 if k == -1 else 0.7)
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    preamble = "True" if ground_truth else "Estimated"
    title = f"{preamble} number of clusters: {n_clusters_}"
    if parameters is not None:
        parameters_str = ", ".join(f"{k}={v}" for k, v in parameters.items())
        title += f" | {parameters_str}"
    ax.set_title(title)
    plt.tight_layout()


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


def get_tsne(embeddings: np.ndarray,
             show=True,
             tsne_reduced=TSNE_REDUCED) -> np.ndarray:
    # TSNE for more accurate reduction and visualization
    tsne = TSNE(n_components=tsne_reduced)
    fitted = tsne.fit_transform(embeddings)

    if show:
        # Plot TSNE
        fig = plt.figure(2)
        if tsne_reduced == 2:
            ax = fig.add_subplot()
            ax.scatter(fitted[:, 0], fitted[:, 1])
            ax.set_title("TSNE 2d visualization")
            ax.set_xlabel("1st Eigenvector")
            ax.set_ylabel("2st Eigenvector")
        elif tsne_reduced == 3:
            ax = fig.add_subplot(projection="3d")
            ax.scatter(fitted[:, 0], fitted[:, 1], fitted[:, 2])
            ax.set_title("TSNE 3d visualization")
            ax.set_xlabel("1st Eigenvector")
            ax.set_ylabel("2st Eigenvector")
            ax.set_zlabel("3st Eigenvector")
        plt.show()

    return fitted


def get_hdbscan(embeddings: np.ndarray, show=True):
    hdb = HDBSCAN(min_cluster_size=25)
    fitted = hdb.fit(embeddings)
    print(f"Labels: {fitted.labels_}, len: {len(fitted.labels_)}")
    if show:
        plot_cluster(embeddings, fitted.labels_, fitted.probabilities_)
        plt.show()

        # Show hierarchy
        hdb.condensed_tree_.plot()
        plt.show()

        # Outlier
        sns.histplot(hdb.outlier_scores_[np.isfinite(hdb.outlier_scores_)],
                     kde=True,
                     stat="density").set(title="Outlier detection")
        plt.show()

        threshold = pd.Series(hdb.outlier_scores_).quantile(0.9)
        outliers = np.where(hdb.outlier_scores_ > threshold)[0]
        plt.scatter(*embeddings.T, s=50, linewidth=0, c='gray', alpha=0.25)
        plt.scatter(*embeddings[outliers].T,
                    s=50,
                    linewidth=0,
                    c='red',
                    alpha=0.5)
        plt.show()

        # scale robustness
        fig, axes = plt.subplots(3, 1)
        for idx, scale in enumerate([1, 1.25, 1.5]):
            hdb.fit(embeddings * scale)
            plot_cluster(embeddings * scale,
                         hdb.labels_,
                         parameters={"Scale": scale},
                         ax=axes[idx])
        plt.show()

        # Multi-level hierarchy
        hdb = HDBSCAN()
        PARAM = (
            {
                "cut_distance": 0.5
            },
            {
                "cut_distance": 1.0
            },
            {
                "cut_distance": 1.5
            },
            {
                "cut_distance": 2.0
            },
            {
                "cut_distance": 2.5
            },
        )
        hdb.fit(embeddings)
        # Hierarchical clustering
        fig, axes = plt.subplots(len(PARAM), 1, figsize=(10, 12))
        for i, param in enumerate(PARAM):
            labels = hdb.dbscan_clustering(**param)
            plot_cluster(embeddings,
                         labels,
                         hdb.probabilities_,
                         param,
                         ax=axes[i])
        plt.show()


def dbscan(embeddings: np.ndarray, show=True, eps=None):
    clusters = []
    noise = []
    if eps is None:
        eps_range = np.arange(0.5, 5, 0.05)
    else:
        eps_range = [eps]

    for eps in eps_range:
        # For every eps, cluster them
        db = DBSCAN(eps=eps)
        fitted = db.fit(embeddings)
        labels = fitted.labels_

        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        clusters.append(n_clusters_)
        noise.append(n_noise_)

    if show:
        if len(eps_range) > 1:
            fig = plt.figure(3)

            ax = fig.add_subplot(1, 2, 1)
            ax.set_title("Estimated number of clusters vs. eps")
            ax.plot(eps_range, clusters, label="Estimated number of clusters")

            ax = fig.add_subplot(1, 2, 2)
            ax.set_title("Estimated number of noise vs. eps")
            ax.plot(eps_range, noise, label="Estimated number of noise")

            plt.show()
        else:
            # When eps is the same, we show dbscan is in fact scale-variant
            # This shows that care should be taken in selecting eps
            fig, axes = plt.subplots(3, 1)
            dbs = DBSCAN(eps=eps_range[0])
            for idx, scale in enumerate([1, 1.25, 1.5]):
                dbs.fit(embeddings * scale)
                plot_cluster(embeddings * scale,
                             dbs.labels_,
                             parameters={
                                 "Scale": scale,
                                 "eps": eps_range[0]
                             },
                             ax=axes[idx])
            plt.show()


def main(arg: argparse.Namespace):
    show = False if args.noshow else True
    try:
        # Load embeddings from cache to avoid re-computing the same values
        embeddings = np.load("cache.npy")
        print("Loaded embeddings from cache")
    except Exception:
        print(arg)
        dev = get_least_used_gpu()
        print(f"Using cuda device: {dev}")

        images, _ = load_data(arg.dataset_dir)

        # Convert from list
        images = torch.stack(images).float().to(dev)

        # Extract embeddings
        image_loader = DataLoader(TensorDataset(images, zeros(len(images))),
                                  batch_size=BATCH_SIZE)

        # Model construction
        model = resnet50.get_resnet_feature(pretrained=True).to(dev)

        embeddings = []

        with tqdm(total=len(image_loader)) as pbar:
            for image, _ in image_loader:
                pbar.update()
                out = model(image).detach().cpu()
                embeddings += out
        embeddings = torch.stack(embeddings).cpu().detach().numpy()
        print(f"embeddings: {embeddings.shape}")

        np.save("cache", embeddings)

    # plot pca
    pca = get_pca(embeddings, show=show)

    # plot TSNE
    _ = get_tsne(pca, show=show, tsne_reduced=3)
    tsne = get_tsne(pca, show=show)

    print(f"After TSNE: {tsne.shape}")

    # Show different DBSCAN visualization
    dbscan(tsne, show=show)
    dbscan(tsne, eps=2, show=show)
    dbscan(tsne, eps=3, show=show)

    # Show HDBSCAN visualization
    get_hdbscan(tsne, show=show)

    # Fit test data on the trained model
    train = tsne[round(len(tsne) * TEST_RATIO):]
    test = tsne[:round(len(tsne) * TEST_RATIO)]

    hdb = HDBSCAN(min_cluster_size=25).fit(train)

    _, axes = plt.subplots(2, 1)

    axes[0].set_title(
        f"The result of training, {len(set(hdb.labels_))} labels")
    pal = sns.color_palette('deep', len(set(hdb.labels_)))
    colors = [
        sns.desaturate(pal[col], sat)
        for col, sat in zip(hdb.labels_, hdb.probabilities_)
    ]
    axes[0].scatter(train.T[0], train.T[1], c=colors, alpha=0.6)

    # plot test data on the training dataset
    hdb.generate_prediction_data()
    test_labels, probs = hdbscan.approximate_predict(hdb, test)
    print(f"Probabilities: {probs}")
    colors = [
        sns.desaturate(pal[col], sat)
        for col, sat in zip(hdb.labels_, hdb.probabilities_)
    ]
    test_colors = [
        pal[col] if col >= 0 else (0.1, 0.1, 0.1) for col in test_labels
    ]
    axes[1].set_title("Added test points")
    axes[1].scatter(train.T[0], train.T[1], c=colors, alpha=0.5)
    axes[1].scatter(*test.T, c=test_colors, s=80, linewidths=1, edgecolors='k')
    plt.show()

    sns.histplot(probs, kde=True,
                 stat="density").set_title("Distribution of probabilities")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="cluster.py")
    parser.add_argument(
        "dataset_dir",
        help=
        "Directory of dataset, images within ${dataset_dir}/files will be used"
    )
    parser.add_argument("--no-show",
                        dest="noshow",
                        help="Whether or not show the plot",
                        action="store_true",
                        required=False)
    args = parser.parse_args()
    main(args)
