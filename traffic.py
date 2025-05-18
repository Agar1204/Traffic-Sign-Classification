import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import keras

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():
    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled convolutional neural network
    model = get_model()

    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate CNNs performance against the test set
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file (if filename is specified)
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Loads image data from data_dir.

    Assumes data_dir has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Returns tuple (images, labels), where images is a list of all 
    the images in the data directory. Each image is formatted as a numpy array with dimensions
    IMG_WIDTH x IMG_HEIGHT x 3. labels is a list of integers, representing the categories for each of the
    corresponding images.
    """
    images = []
    labels = []
    for i in range(NUM_CATEGORIES):
        path = os.path.join(os.getcwd(), data_dir, str(i)) # Path for each image directory
        image_list = os.listdir(path) # List of all images in each subdirectory
        for image_path in image_list:
            image_path = os.path.join(path, image_path)
            image = cv2.imread(image_path) # Read in the image as a numpy array
            resized_image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT)) #Resize image
            images.append(resized_image)
            labels.append(i)
  
    return (images, labels)


def get_model():
    """
    Returns a compiled CNN. Input shape of the first layer is (IMG_WIDTH, IMG_HEIGHT, 3).
    The output layer has 43 units, one for each category.
    """
    model = tf.keras.models.Sequential([
        # Convolutional layer: Learns 32 filters using 3x3 kernel. Output: 28x28x32, dropout to avoid overfitting
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(30, 30, 3)
        ),
        tf.keras.layers.Dropout(0.1),

        # Pooling layer: Pooling size 2x2, output: 14x14x32. 
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

        # 2nd Convolutional layer: Learn 64 filters using 3x3 kernel. Output: 12x12x64, dropout to avoid overfitting
        tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu", input_shape=(14, 14, 32)
        ),
        tf.keras.layers.Dropout(0.1),

        # 2nd Pooling layer: Pooling size 2x2, output: 6x6x64
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.Flatten(),

        # Dense layer with dropout to avoid overfitting 
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.35),

        # Output layer with output units for all 43 traffic signs
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
        ])
    return model


# Only run code if script is run directly, not imported. 
if __name__ == "__main__":
    main()
