# utils.py
# provides utilities for learning a neural network model


import time
import numpy as np

from sklearn.utils import shuffle
from sklearn.cross_validation import StratifiedKFold


def get_current_time():
    return time.strftime('%Y-%m-%d_%H:%M:%S')


# take the data and label arrays, split them preserving
# the class representations, and optionally normalize them
def train_test_split(X, y, eval_size, normalize=True):
    kf = StratifiedKFold(y, round(1. / eval_size))

    train_indices, valid_indices = next(iter(kf))
    X_train, y_train = X[train_indices], y[train_indices]
    X_valid, y_valid = X[valid_indices], y[valid_indices]

    if normalize:
        mean = np.mean(X_train, axis=0)
        print(mean)
        X_train -= mean
        X_valid -= mean

        std = np.std(X_train, axis=0)
        print(std)
        X_train /= std
        X_valid /= std
    raw_input()

    return X_train, y_train, X_valid, y_valid


# load the data and label arrays from disk,
# and shuffles both
# expects data to be in a numpy.ndarray of the form
# [[x00, x01, ..., x0N]
#  [x10, x11, ..., x1N]
#  ...               ]]
#  where each row is a 1D-array
#  representing all the channels from a single
#  image flattened and stacked.  This is necessary
#  for pre-processing.
def load(data_file, labels_file=None, random_state=None):
    # Load X matrix (data)
    data = np.load(data_file)
    # Load y vector (labels)
    labels = None
    if labels_file is not None:
        labels = np.load(labels_file)
    # Randomly shuffle data
    if labels is None:
        data = shuffle(data, random_state=random_state)
    else:
        data, labels = shuffle(data, labels, random_state=random_state)
    # Return data
    return data, labels
