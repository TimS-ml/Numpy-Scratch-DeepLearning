# https://en.wikipedia.org/wiki/Activation_function

# from abc import ABC, abstractmethod
import numpy as np


class ActivationBase(object):
    def __call__(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return self.fn(x)

    def fn(self, x):
        raise NotImplementedError()

    def grad(self, x):
        raise NotImplementedError()


class Sigmoid(ActivationBase):
    def fn(self, x):
        return 1 / (1 + np.exp(-x))

    def grad(self, x):
        p = self.fn(x)
        return p * (1 - p)


class Softmax(ActivationBase):
    def fn(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def grad(self, x):
        p = self.fn(x)
        return p * (1 - p)


class TanH(ActivationBase):
    def fn(self, x):
        return 2 / (1 + np.exp(-2 * x)) - 1

    def grad(self, x):
        p = self.fn(x)
        return 1 - np.power(p, 2)


class ReLU(ActivationBase):
    def fn(self, x):
        return np.where(x >= 0, x, 0)

    def grad(self, x):
        return np.where(x >= 0, 1, 0)


class LeakyReLU(ActivationBase):
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def fn(self, x):
        return np.where(x >= 0, x, self.alpha * x)

    def grad(self, x):
        return np.where(x >= 0, 1, self.alpha)


class ELU(ActivationBase):
    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def fn(self, x):
        return np.where(x >= 0.0, x, self.alpha * (np.exp(x) - 1))

    def grad(self, x):
        return np.where(x >= 0.0, 1, self.fn(x) + self.alpha)


class SELU(ActivationBase):
    # Reference : https://arxiv.org/abs/1706.02515
    # https://github.com/bioinf-jku/SNNs/blob/master/SelfNormalizingNetworks_MLP_MNIST.ipynb
    def __init__(self):
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946

    def fn(self, x):
        return self.scale * np.where(x >= 0.0, x, self.alpha * (np.exp(x) - 1))

    def grad(self, x):
        return self.scale * np.where(x >= 0.0, 1, self.alpha * np.exp(x))


class SoftPlus(ActivationBase):
    def fn(self, x):
        return np.log(1 + np.exp(x))

    def grad(self, x):
        return 1 / (1 + np.exp(-x))
