from __future__ import print_function

from sys import path as syspath
import os

packagePath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

print(packagePath)
syspath.append(packagePath)

import gzip
import requests
import io

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

from cnn import main


url = 'https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/datasets/data/digits.csv.gz?raw=true'

f = requests.get(url).content
data = np.loadtxt(gzip.open(io.BytesIO(f), 'rt'),
                  delimiter=',',
                  dtype=np.float32)

X = data[:, :-1]
y = data[:, -1]

# Convert to one-hot encoding
y = to_categorical(y.astype('int'))

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.4,
                                                    seed=1)

# Reshape X to (n_samples, channels, height, width)
X_train = X_train.reshape((-1, 1, 8, 8))
X_test = X_test.reshape((-1, 1, 8, 8))

main(X_train, X_test, y_train, y_test)
