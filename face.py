import os
import sys

import cv2
import keras
import numpy as np
from sklearn.model_selection import train_test_split

EPOCHS = 50
IMG_WIDTH = 128
IMG_HEIGHT = 128
# NUM_CATEGORIES = 43
NUM_CATEGORIES = 2
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python face.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    # labels = keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(np.array(images),
                                                        np.array(labels),
                                                        test_size=TEST_SIZE)

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test, y_test, verbose="2")

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []
    data_path = os.path.join(data_dir, "files")
    with open(os.path.join(data_dir, "labels.txt")) as label_file:
        for image in os.listdir(data_path):
            img = cv2.imread(os.path.join(data_path, image))
            res = cv2.resize(img, dsize=(IMG_WIDTH, IMG_HEIGHT))
            # Add image
            images.append(res)
            # Add line
            line_data = int(label_file.readline().split(" ")[0])
            labels.append(line_data)

    return (images, labels)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = keras.models.Sequential()
    input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)

    model.add(
        keras.layers.Conv2D(32, (3, 3),
                            activation="relu",
                            input_shape=input_shape))
    model.add(keras.layers.MaxPool2D(2, 2))
    model.add(keras.layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(keras.layers.MaxPool2D(2, 2))
    model.add(keras.layers.Conv2D(64, (3, 3), activation="relu"))

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'),
              )  # binary classification
    # model.add(
    #     keras.layers.Dense(NUM_CATEGORIES,
    #                           activation="softmax",
    #                           name="output"))
    # DEBUG
    # model.summary()

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.build(input_shape)
    return model


if __name__ == "__main__":
    main()
