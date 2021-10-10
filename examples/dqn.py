from __future__ import print_function

import numpy as np

from scratchDL import base as dl
from scratchDL.base import layers as lyr
from scratchDL.base import optm
from scratchDL.base import loss
from scratchDL.base import NeuralNetwork
from scratchDL.models import DeepQNetwork

from scratchDL.utils import train_test_split, to_categorical
from scratchDL.utils import Plot


def main():
    dqn = DeepQNetwork(env_name='CartPole-v1',
                       epsilon=0.9,
                       gamma=0.8,
                       decay_rate=0.005,
                       min_epsilon=0.1)

    # Model builder
    def model(n_inputs, n_outputs):
        clf = NeuralNetwork(
                optimizer=optm.Adam(), 
                loss=loss.SquareLoss)
        clf.add(lyr.Dense(64, input_shape=(n_inputs, )))
        clf.add(lyr.Activation('relu'))
        clf.add(lyr.Dense(n_outputs))
        return clf

    dqn.set_model(model)

    print()
    dqn.model.summary(name="Deep Q-Network")

    dqn.train(n_epochs=500)
    dqn.play(n_epochs=100)


if __name__ == "__main__":
    main()
