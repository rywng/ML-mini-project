# Miniproject: Exploring Image Classification, Face Pose, and Unsupervised Learning

This document provides an overview of the mini project exploring various computer vision concepts: image classification, face pose regression, and unsupervised learning.

**Academic Integrity:**

It is crucial to note that this repository serves as a guide and inspiration, **not a substitute for independent work.** Plagiarism is strictly prohibited. While code snippets and technical details may be used as references, **direct copy-pasting of code or text is unacceptable.** Remember to always properly cite and reference any external resources used in your project.

## Project Scope:

The project delves into these three tasks:

1. **Image Classification:** Train a model to accurately categorize images.
2. **Face Pose Regression:** Predict the 3D orientation (pose) of faces within images.
3. **Unsupervised Learning:** Analyze hidden structures and relationships within an unlabeled dataset using clustering techniques.

## Prerequisites:

- Python 3.8+
- Tensorboard (Optional, for logging the training statistics)
- PyTorch (optional, required for the simple face model)
- OpenCV

## Getting Started:

1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`

## Running the Project:

**Image Classification:**

1. Train a model: `python face.py --help`
2. Classify an image: `python test.py --help`

**Face Pose Regression:**

1. Train a model: `python face.py --help`
2. Predict pose for an image: `python test.py --help`

**Unsupervised Learning:**

1. Analyze cluster structure: `python cluster.py path/to/dataset`
2. Visualize clusters: `python notebooks/<redacted>_miniproject_submission.ipynb`

## Project Structure:

- `assets`: Visualization images and model checkpoints.
- `cache.npy`: Preprocessed data for faster loading.
- `<redacted>_miniproject_*.ipynb`: Notebooks for different project tasks.
- `cluster.py`: Unsupervised clustering script for the Genki4k dataset.
- `cuda_utils.py`: Helper functions for GPU acceleration.
- `datasets`: Stores the `Genki4k` and `Genki4k-mini` datasets.
- `face.py`: Implements face detection and pose regression.
- `logger_utils.py`: Logging functionalities.
- `model_utils.py`: Utilities for building and training models.
- `preprocessing.py`: Preprocesses raw image data.
- `requirements.txt`: Lists required dependencies.
- `runs`: Stores training logs and checkpoints for different models.
- `Session.vim`: Configuration file for Vim.
- `sync.sh`: Script for synchronizing project files to a remote server, so you can iterate fast and train remotely.
- `test.py`: Script for running inference and evaluating models.

## Further Exploration:

- Feel free to explore the code and notebooks for deeper insights into each task.

**Remember:**

This repository serves as a starting point. Always ensure thorough understanding and independent implementation of the concepts involved. Plagiarism is strongly discouraged.

**Collaboration:**

If you have any questions, suggestions, or contributions, please feel free to raise an issue or submit a pull request. Happy learning!
