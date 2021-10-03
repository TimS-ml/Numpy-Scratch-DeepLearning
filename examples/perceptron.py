from __future__ import print_function
from sklearn import datasets
import math
import numpy as np

# Import helper functions
from scratchDL.utils import train_test_split, to_categorical, normalize, accuracy_score
from scratchDL.base.activation import Sigmoid
from scratchDL.base.loss import CrossEntropy, SquareLoss
from scratchDL.utils import Plot
from scratchDL.utils.misc import bar_widgets
import progressbar


class Perceptron():
    """The Perceptron. One layer neural network classifier.

    Parameters:
    -----------
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    activation_function: class
        The activation that shall be used for each neuron.
        Possible choices: Sigmoid, ExpLU, ReLU, LeakyReLU, SoftPlus, TanH
    loss: class
        The loss function used to assess the model's performance.
        Possible choices: SquareLoss, CrossEntropy
    learning_rate: float
        The step length that will be used when updating the weights.
    """
    def __init__(self,
                 n_iterations=20000,
                 activation_function=Sigmoid,
                 loss=SquareLoss,
                 learning_rate=0.01):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.loss = loss()
        self.activation_func = activation_function()
        self.progressbar = progressbar.ProgressBar(widgets=bar_widgets)

    def fit(self, X, y):
        n_samples, n_features = np.shape(X)
        _, n_outputs = np.shape(y)

        # Initialize weights between [-1/sqrt(N), 1/sqrt(N)]
        limit = 1 / math.sqrt(n_features)
        self.W = np.random.uniform(-limit, limit, (n_features, n_outputs))
        self.w0 = np.zeros((1, n_outputs))

        for i in self.progressbar(range(self.n_iterations)):
            # Calculate outputs
            linear_output = X.dot(self.W) + self.w0
            y_pred = self.activation_func(linear_output)
            # Calculate the loss gradient w.r.t the input of the activation function
            error_gradient = self.loss.gradient(
                y, y_pred) * self.activation_func.gradient(linear_output)
            # Calculate the gradient of the loss with respect to each weight
            grad_wrt_w = X.T.dot(error_gradient)
            grad_wrt_w0 = np.sum(error_gradient, axis=0, keepdims=True)
            # Update weights
            self.W -= self.learning_rate * grad_wrt_w
            self.w0 -= self.learning_rate * grad_wrt_w0

    # Use the trained model to predict labels of X
    def predict(self, X):
        y_pred = self.activation_func(X.dot(self.W) + self.w0)
        return y_pred



def main():
    data = datasets.load_digits()
    X = normalize(data.data)
    y = data.target

    # One-hot encoding of nominal y-values
    y = to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.33,
                                                        seed=1)

    # Perceptron
    clf = Perceptron(n_iterations=5000,
                     learning_rate=0.001,
                     loss=CrossEntropy,
                     activation_function=Sigmoid)
    clf.fit(X_train, y_train)

    y_pred = np.argmax(clf.predict(X_test), axis=1)
    y_test = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)

    # Reduce dimension to two using PCA and plot the results
    Plot().plot_in_2d(X_test,
                      y_pred,
                      title="Perceptron",
                      accuracy=accuracy,
                      legend_labels=np.unique(y))


if __name__ == "__main__":
    main()
