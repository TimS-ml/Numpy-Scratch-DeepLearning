from __future__ import print_function
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

from scratchDL.supervised_learning import LDA
from scratchDL.utils import calculate_covariance_matrix, accuracy_score
from scratchDL.utils import normalize, standardize, train_test_split, Plot
from scratchDL.unsupervised_learning import PCA


def main():
    # Load the dataset
    data = datasets.load_iris()
    X = data.data
    y = data.target

    # Three -> two classes
    X = X[y != 2]
    y = y[y != 2]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # Fit and predict using LDA
    lda = LDA()
    lda.fit(X_train, y_train)
    y_pred = lda.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)

    Plot().plot_in_2d(X_test, y_pred, title="LDA", accuracy=accuracy)


if __name__ == "__main__":
    main()
