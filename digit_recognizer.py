import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras import layers, models
import yaml
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
from logzero import logger


class DigitRecognizer:
    def __init__(self, config_file):
        # Load the configuration from the YAML file
        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f)

        self.train = pd.read_csv(self.config["input"]["train"])
        self.test = pd.read_csv(self.config["input"]["test"])

        # Preprocess the data
        self.y_train = self.train["label"].astype(int)
        self.y_train = tf.keras.utils.to_categorical(self.y_train, num_classes=10)

        self.x_train = self.train.drop(columns=["label"], axis=1)
        self.x_test = self.test.astype(float)

        # Reshape the features
        self.x_train = self.x_train.values.reshape(-1, 28, 28, 1)
        self.x_train = self.x_train / 255.0
        self.x_test = self.x_test.values.reshape(-1, 28, 28, 1)
        self.x_test = self.x_test / 255.0

        # Split the training data into training and validation sets
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            self.x_train, self.y_train, test_size=0.2
        )

    def build_model(self):
        self.model = models.Sequential()

        # Add convolutional layers
        self.model.add(
            layers.Conv2D(64, (3, 3), activation="relu", input_shape=(28, 28, 1))
        )
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.MaxPooling2D((2, 2)))

        self.model.add(layers.Conv2D(128, (3, 3), activation="relu"))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.MaxPooling2D((2, 2)))

        self.model.add(layers.Conv2D(256, (3, 3), activation="relu"))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.MaxPooling2D((2, 2)))

        # Add dense layers
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(256, activation="relu"))
        self.model.add(layers.Dense(128, activation="relu"))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(10, activation="softmax"))

        # Compile the model with Adam optimizer and categorical crossentropy loss function
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

    def train_model(self):
        # Use ImageDataGenerator to generate augmented images for training data
        datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest",
        )

        # Use early stopping to stop training when the validation loss stops improving
        # and LearningRateScheduler to decrease the learning rate after every 20 epochs
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=30),
            LearningRateScheduler(lambda epoch: 0.001 * (0.1 ** (epoch // 20))),
        ]

        # Train the model
        history = self.model.fit_generator(
            datagen.flow(self.x_train, self.y_train, batch_size=50),
            epochs=100,
            validation_data=(self.x_val, self.y_val),
            callbacks=callbacks,
        )

        # Printing the final validation accuracy and loss
        DEBUGVAL__ = history.history["val_accuracy"][-1]
        logger.debug(f"history.history [val_accuracy][-1]\n {DEBUGVAL__}")

        DEBUGVAL__ = history.history["val_loss"][-1]
        logger.debug(f"history.history[val_loss][-1]\n {DEBUGVAL__}")

        return self.model

    def predict(self):

        # Make predictions on the test data
        predictions = self.model.predict(self.x_test)
        # Get the class with the highest predicted probability for each sample
        result = np.argmax(predictions, axis=1)
        results = pd.Series(result, name="Label")
        submission = pd.concat(
            [pd.Series(range(1, 28001), name="ImageId"), results], axis=1
        )
        # Write the submission DataFrame to a CSV file
        submission.to_csv(self.config["output"], index=False)


if __name__ == "__main__":
    recognizer = DigitRecognizer("/data/digits_recognizer/digit_recognizer.yaml")
    recognizer.build_model()
    recognizer.train_model()
    recognizer.predict()
