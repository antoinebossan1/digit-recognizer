# Digit_Recognizer

Objective:

This code is a python script that uses TensorFlow, Keras and other libraries to build and train a digit recognition model on the MNIST dataset. The MNIST dataset consists of images of handwritten digits and their corresponding labels (0-9). The script uses the image data to train the model and then makes predictions on the test data.

The model architecture is a convolutional neural network (CNN) with several convolutional and pooling layers, followed by a few dense layers for classification. The model is trained using an ImageDataGenerator for data augmentation and the Adam optimizer. The accuracy and loss of the model is monitored during training with EarlyStopping and a LearningRateScheduler to reduce the learning rate if the validation loss does not improve over a number of epochs. Predictions are made using the predict method on the trained model.



Dependencies:

To run this script, you need to install the following libraries:
Numpy
TensorFlow
Pandas
TensorFlow Keras
PyYAML
Scikit-learn
Logzero
It is also necessary to have access to a set of training and test data of numbers. The paths to this data must be specified in a YAML configuration file.
