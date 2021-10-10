from __future__ import print_function

import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

from scratchDL import base as dl
from scratchDL.base import layers as lyr
from scratchDL.base import optm
from scratchDL.base import loss
from scratchDL.base import activation as act
from scratchDL.base import NeuralNetwork

from scratchDL.utils import train_test_split, to_categorical
from scratchDL.utils import Plot


def main():
    data = datasets.load_digits()
    X = data.data
    y = data.target

    # Convert to one-hot encoding
    y = to_categorical(y.astype('int'))

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.4,
                                                        seed=1)

    # Reshape X to (n_samples, channels, height, width)
    X_train = X_train.reshape((-1, 1, 8, 8))
    X_test = X_test.reshape((-1, 1, 8, 8))

    clf = dl.NeuralNetwork(
                optimizer=optm.Adam(),  # default lr is 0.001
                loss=loss.CrossEntropy,
                validation_data=(X_test, y_test))
    
    clf.add(lyr.Conv2D(
                  n_filters=16,
                  filter_shape=(3, 3),
                  stride=1,
                  input_shape=(1, 8, 8),
                  padding='same'))
    clf.add(lyr.Activation(act.ReLU))
    clf.add(lyr.Dropout(0.25))
    clf.add(lyr.BatchNorm())
    clf.add(lyr.Conv2D(
                  n_filters=32, 
                  filter_shape=(3, 3), 
                  stride=1,
                  padding='same'))
    clf.add(lyr.Activation(act.ReLU))
    clf.add(lyr.Dropout(0.25))
    clf.add(lyr.BatchNorm())
    clf.add(lyr.Flatten())
    clf.add(lyr.Dense(256))
    clf.add(lyr.Activation(act.ReLU))
    clf.add(lyr.Dropout(0.4))
    clf.add(lyr.BatchNorm())
    clf.add(lyr.Dense(10))
    clf.add(lyr.Activation(act.Softmax))

    print()
    clf.summary(name='ConvNet')

    train_err, val_err = clf.fit(X_train, y_train, n_epochs=10, batch_size=256)

    # Training and validation error plot
    n = len(train_err)
    training, = plt.plot(range(n), train_err, label='Training Error')
    validation, = plt.plot(range(n), val_err, label='Validation Error')
    plt.legend(handles=[training, validation])
    plt.title('Error Plot')
    plt.ylabel('Error')
    plt.xlabel('Iterations')
    plt.show()

    _, accuracy = clf.test_on_batch(X_test, y_test)
    print('Accuracy:', accuracy)

    y_pred = np.argmax(clf.predict(X_test), axis=1)
    X_test = X_test.reshape(-1, 8 * 8)
    # Reduce dimension to 2D using PCA and plot the results
    Plot().plot_in_2d(X_test,
                      y_pred,
                      title='Convolutional Neural Network',
                      accuracy=accuracy,
                      legend_labels=range(10))


if __name__ == "__main__":
    main()
