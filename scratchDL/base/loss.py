# check `nn.train_on_batch`

from __future__ import division

import numpy as np
from scratchDL.utils import accuracy_score


class Loss(object):
    def loss(self, y_true, y_pred):
        return NotImplementedError()

    def grad(self, y, y_pred):
        raise NotImplementedError()

    # you don't need this in regression
    def acc(self, y, y_pred):
        return 0


class SquareLoss(Loss):
    def __init__(self):
        pass

    def loss(self, y, y_pred):
        return 0.5 * np.power((y - y_pred), 2)

    def grad(self, y, y_pred):
        return - (y - y_pred)


class CrossEntropy(Loss):
    def __init__(self):
        self.eps = 1e-15

    def loss(self, y, y_pred):
        # Avoid division by zero
        y_pred = np.clip(y_pred, self.eps, 1 - self.eps)
        return - y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred)

    def acc(self, y, y_pred):
        return accuracy_score(np.argmax(y, axis=1), np.argmax(y_pred, axis=1))

    def grad(self, y, y_pred):
        # Avoid division by zero
        y_pred = np.clip(y_pred, self.eps, 1 - self.eps)
        return - (y / y_pred) + (1 - y) / (1 - y_pred)
