from __future__ import print_function

import math
import numpy as np
from tqdm import tqdm
from sklearn import datasets

from scratchDL.utils import train_test_split, to_categorical, normalize
from scratchDL.base.activation import Sigmoid
from scratchDL.base.loss import CrossEntropy, SquareLoss
from scratchDL.utils import Plot


class Perceptron():
    '''The Perceptron. One layer neural network classifier.
    '''
    def __init__(self,
                 n_iterations=20000,
                 activation=Sigmoid,
                 loss=SquareLoss,
                 learning_rate=0.01):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.loss = loss()
        self.activation = activation()

    def fit(self, X, y):
        n_samples, n_features = np.shape(X)
        _, n_outputs = np.shape(y)

        # Initialize weights between [-1/sqrt(N), 1/sqrt(N)]
        limit = 1 / math.sqrt(n_features)
        self.W = np.random.uniform(-limit, limit, (n_features, n_outputs))
        self.w0 = np.zeros((1, n_outputs))

        for i in tqdm(range(self.n_iterations)):
            # Calculate outputs
            linear_output = X.dot(self.W) + self.w0
            y_pred = self.activation(linear_output)

            # Calculate the loss gradient w.r.t the input of the activation function
            error_gradient = self.loss.grad(
                y, y_pred) * self.activation.grad(linear_output)

            # Calculate the gradient of the loss with respect to each weight
            grad_wrt_w = X.T.dot(error_gradient)
            grad_wrt_w0 = np.sum(error_gradient, axis=0, keepdims=True)

            # Update weights
            self.W -= self.learning_rate * grad_wrt_w
            self.w0 -= self.learning_rate * grad_wrt_w0

    # Use the trained model to predict labels of X
    def predict(self, X):
        y_pred = self.activation(X.dot(self.W) + self.w0)
        return y_pred



def main():
    data = datasets.load_digits()
    X = normalize(data.data)
    y = data.target

    # One-hot encoding of nominal y-values
    y = to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.4,
                                                        seed=1)

    # Perceptron
    clf = Perceptron(n_iterations=5000,
                     learning_rate=0.001,
                     loss=CrossEntropy,
                     activation=Sigmoid)
    clf.fit(X_train, y_train)

    y_pred = np.argmax(clf.predict(X_test), axis=1)
    y_test = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_test, y_pred)

    print('Accuracy:', accuracy)

    # Reduce dimension to two using PCA and plot the results
    Plot().plot_in_2d(X_test,
                      y_pred,
                      title='Perceptron',
                      accuracy=accuracy,
                      legend_labels=np.unique(y))


if __name__ == '__main__':
    main()
