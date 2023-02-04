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
        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.train = pd.read_csv(self.config["input"]["train"])
        self.test= pd.read_csv(self.config["input"]["test"])

        self.y_train=self.train["label"].astype(int)
        self.y_train=tf.keras.utils.to_categorical(self.y_train, num_classes=10)

        self.x_train=self.train.drop(columns=["label"],axis=1)
        self.x_test=self.test.astype(float) 

        self.x_train=self.x_train.values.reshape(-1,28,28,1)
        self.x_train = self.x_train/255.0
        self.x_test=self.x_test.values.reshape(-1,28,28,1)
        self.x_test = self.x_test/255.0
        
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.x_train, self.y_train, test_size=0.2)

    def build_model(self):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.MaxPooling2D((2, 2)))

        self.model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.MaxPooling2D((2, 2)))

        self.model.add(layers.Conv2D(256, (3, 3), activation='relu'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.MaxPooling2D((2, 2)))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(256, activation='relu'))
        self.model.add(layers.Dense(128, activation='relu'))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(10, activation='softmax'))

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                        loss='categorical_crossentropy',  
                        metrics=['accuracy'])



    def train_model(self):
        datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30),
            LearningRateScheduler(lambda epoch: 0.001 * (0.1 ** (epoch // 20)))
        ]

        history = self.model.fit_generator(datagen.flow(self.x_train, self.y_train, batch_size=50),
                                    epochs=100,
                                    validation_data=(self.x_val, self.y_val),
                                    callbacks=callbacks)


        DEBUGVAL__ = history.history["val_accuracy"][-1]
        logger.debug(f"history.history [val_accuracy][-1]\n {DEBUGVAL__}")

        DEBUGVAL__ = history.history["val_loss"][-1]
        logger.debug(f"history.history[val_loss][-1]\n {DEBUGVAL__}")

        return self.model

    def predict(self):
        predictions = self.model.predict(self.x_test)
        result=np.argmax(predictions,axis=1)
        results=pd.Series(result,name="Label")
        submission= pd.concat([pd.Series(range(1,28001),name="ImageId"),results],axis=1)
        submission.to_csv(self.config["output"],index=False)

if __name__ == "__main__":
    recognizer = DigitRecognizer("/data/digits_recognizer/digit_recognizer.yaml")
    recognizer.build_model()
    recognizer.train_model()
    recognizer.predict()












