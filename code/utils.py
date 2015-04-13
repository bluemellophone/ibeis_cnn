# utils.py
# provides utilities for learning a neural network model


import time
import numpy as np
import functools
import operator

import theano
import theano.tensor as T
import lasagne
from lasagne import objectives

from lasagne import layers
from lasagne import nonlinearities  # NOQA
from sklearn.cross_validation import StratifiedKFold
from sklearn.utils import shuffle
import cv2
import cPickle as pickle


class ANSI:
    RED     = '\033[91m'
    GREEN   = '\033[92m'
    BLUE    = '\033[94m'
    CYAN    = '\033[96m'
    WHITE   = '\033[97m'
    YELLOW  = '\033[93m'
    MAGENTA = '\033[95m'
    GREY    = '\033[90m'
    BLACK   = '\033[90m'
    # DEFAULT = '\033[99m'
    RESET   = '\033[0m'


RANDOM_SEED = None
# RANDOM_SEED = 42


def get_current_time():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def train_test_split(X, y, eval_size):
    # take the data and label arrays, split them preserving the class distributions
    kf = StratifiedKFold(y, round(1. / eval_size))

    train_indices, valid_indices = next(iter(kf))
    X_train, y_train = X[train_indices], y[train_indices]
    X_valid, y_valid = X[valid_indices], y[valid_indices]

    return X_train, y_train, X_valid, y_valid


def load(data_file, labels_file=None, random_state=None):
    # Load X matrix (data)
    data = np.load(data_file, mmap_mode='r')
    # Load y vector (labels)
    labels = None
    if labels_file is not None:
        labels = np.load(labels_file, mmap_mode='r')
    # Return data
    return data, labels


def print_header_columns():
    print('''
[info]   Epoch |  Train Loss  |  Valid Loss  |  Train / Val  |  Valid Acc  |  Dur
[info] --------|--------------|--------------|---------------|-------------|------\
''')


def print_layer_info(output_layer):
    nn_layers = layers.get_all_layers(output_layer)[::-1]
    print('\n[info] Network Structure:')
    for layer in nn_layers:
        output_shape = layer.get_output_shape()
        print('[info]     {:<18}\t{:<20}\tproduces {:>7,} outputs'.format(
            layer.__class__.__name__,
            str(output_shape),
            int(str(functools.reduce(operator.mul, output_shape[1:]))),
        ))
    print('[info] ...this model has {:,} learnable parameters\n'.format(
        layers.count_params(output_layer)
    ))


def print_epoch_info(valid_loss, best_valid_loss, valid_accuracy,
                     best_valid_accuracy, train_loss, best_train_loss, epoch,
                     duration):
    best_train      = train_loss == best_train_loss
    best_valid      = valid_loss == best_valid_loss
    best_accuracy   = valid_accuracy == best_valid_accuracy
    ratio           = train_loss / valid_loss
    unhealthy_ratio = ratio <= 0.5 or 2.0 <= ratio
    print('[info]  {:>5}  |  {}{:>10.6f}{}  |  {}{:>10.6f}{}  '
          '|  {}{:>11.6f}{}  |  {}{:>9}{}  |  {:>3.1f}s'.format(
              epoch,
              ANSI.BLUE if best_train else '',
              train_loss,
              ANSI.RESET if best_train else '',
              ANSI.GREEN if best_valid else '',
              valid_loss,
              ANSI.RESET if best_valid else '',
              ANSI.RED if unhealthy_ratio else '',
              ratio,
              ANSI.RESET if unhealthy_ratio else '',
              ANSI.MAGENTA if best_accuracy else '',
              '{:.2f}%'.format(valid_accuracy * 100),
              ANSI.RESET if best_accuracy else '',
              duration,
          ))


def float32(k):
    return np.cast['float32'](k)


def batch_iterator(X, y, bs, mean=None, std=None, rand=False, augment=None):
    # divides X and y into batches of size bs for sending to the GPU
    # Randomly shuffle data
    if rand:
        if y is None:
            X = shuffle(X, random_state=RANDOM_SEED)
        else:
            X, y = shuffle(X, y, random_state=RANDOM_SEED)
    N = X.shape[0]
    for i in range((N + bs - 1) // bs):
        sl = slice(i * bs, (i + 1) * bs)
        Xb = X[sl]
        if y is not None:
            yb = y[sl]
        else:
            yb = None
        # Get corret dtype
        Xb_ = Xb.astype(np.float32)
        yb_ = yb.astype(np.int32)
        # Whiten)
        if mean is not None:
            Xb_ -= mean
        if std is not None and std != 0.0:
            Xb_ /= std
        # Augment
        if augment is not None:
            Xb_, yb_ = augment(Xb_, yb_)
        yield Xb_, yb_


def multinomial_nll(x, t):
    return T.nnet.categorical_crossentropy(x, t)


def create_iter_funcs(learning_rate, momentum, output_layer, input_type=T.tensor4,
                      output_type=T.ivector, regularization=None):
    # build the Theano functions that will be used in the optimization
    # refer to this link for info on tensor types:
    # http://deeplearning.net/software/theano/library/tensor/basic.html
    X = input_type('x')
    y = output_type('y')
    X_batch = input_type('x_batch')
    y_batch = output_type('y_batch')

    # we are minimizing the multi-class negative log-likelihood
    objective = objectives.Objective(output_layer, loss_function=multinomial_nll)

    loss_train = objective.get_loss(X_batch, target=y_batch)
    if regularization is not None:
        L2 = lasagne.regularization.l2(output_layer)
        loss_train += L2 * regularization
    loss_eval = objective.get_loss(X_batch, target=y_batch, deterministic=True)
    predict_proba = output_layer.get_output(X_batch, deterministic=True)
    pred = T.argmax(predict_proba, axis=1)

    accuracy = T.mean(T.eq(pred, y_batch))

    all_params = layers.get_all_params(output_layer)
    updates = lasagne.updates.nesterov_momentum(
        loss_train, all_params, learning_rate, momentum)

    train_iter = theano.function(
        inputs=[theano.Param(X_batch), theano.Param(y_batch)],
        outputs=[loss_train],
        updates=updates,
        givens={
            X: X_batch,
            y: y_batch,
        },
    )

    valid_iter = theano.function(
        inputs=[theano.Param(X_batch), theano.Param(y_batch)],
        outputs=[loss_eval, accuracy],
        givens={
            X: X_batch,
            y: y_batch,
        },
    )

    predict_iter = theano.function(
        inputs=[theano.Param(X_batch)],
        outputs=predict_proba,
        givens={
            X: X_batch,
        },
    )

    return train_iter, valid_iter, predict_iter


def add_channels(data):
    add = 5 + 3 + 3 + 2
    points, channels, height, width = data.shape
    dtype = data.dtype
    data_channels = np.empty((points, channels + add, height, width), dtype=dtype)
    data_channels[:, :channels, :, :] = data
    for index in range(points):
        image     = cv2.merge(data[index])
        hsv       = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab       = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobelx    = cv2.Sobel(grayscale, -1, 1, 0)
        sobely    = cv2.Sobel(grayscale, -1, 0, 1)
        sobelxx   = cv2.Sobel(sobelx, -1, 1, 0)
        sobelyy   = cv2.Sobel(sobely, -1, 0, 1)
        sobelxy   = cv2.Sobel(sobelx, -1, 0, 1)
        data_channels[index, 3:6, :, :] = np.array(cv2.split(hsv))
        data_channels[index, 6:9, :, :] = np.array(cv2.split(lab))
        data_channels[index, 6:9, :, :] = np.array(cv2.split(lab))
        data_channels[index, 9, :, :]   = grayscale
        data_channels[index, 10, :, :]  = 255.0 - grayscale
        data_channels[index, 11, :, :]  = sobelx
        data_channels[index, 12, :, :]  = sobely
        data_channels[index, 13, :, :]  = sobelxx
        data_channels[index, 14, :, :]  = sobelyy
        data_channels[index, 15, :, :]  = sobelxy
    return data_channels


def show_image_from_data(data):
    def add_to_template(template, x, y, image_):
        template[y * h : (y + 1) * h, x * h : (x + 1) * w] = image_

    def replicate(channel, index=None):
        temp = np.zeros((3, h, w), dtype=np.uint8)
        if index is None:
            temp[0] = channel
            temp[1] = channel
            temp[2] = channel
        else:
            temp[index] = channel
        return cv2.merge(temp)

    template_h, template_w = (5, 5)
    image = data[0]
    c, h, w = image.shape

    # Create temporary copies for displaying
    bgr_   = cv2.merge(image[0:3])
    hsv_   = cv2.merge(image[3:6])
    lab_   = cv2.merge(image[6:9])

    template = np.zeros((template_h * h, template_w * w, 3), dtype=np.uint8)
    add_to_template(template, 0, 0, replicate(image[0]))
    add_to_template(template, 1, 0, replicate(image[1]))
    add_to_template(template, 2, 0, replicate(image[2]))
    add_to_template(template, 3, 0, bgr_)

    add_to_template(template, 0, 1, replicate(image[3]))
    add_to_template(template, 1, 1, replicate(image[4]))
    add_to_template(template, 2, 1, replicate(image[5]))
    add_to_template(template, 3, 1, hsv_)

    add_to_template(template, 0, 2, replicate(image[6]))
    add_to_template(template, 1, 2, replicate(image[7]))
    add_to_template(template, 2, 2, replicate(image[8]))
    add_to_template(template, 3, 2, lab_)

    add_to_template(template, 0, 3, replicate(image[9]))
    add_to_template(template, 1, 3, replicate(image[10]))

    add_to_template(template, 0, 4, replicate(image[11]))
    add_to_template(template, 1, 4, replicate(image[12]))
    add_to_template(template, 2, 4, replicate(image[13]))
    add_to_template(template, 3, 4, replicate(image[14]))
    add_to_template(template, 4, 4, replicate(image[15]))

    cv2.imshow('template', template)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_best_model(best_weights, best_valid_accuracy, weights_file):
    acc = 100.0 * best_valid_accuracy
    print('[model] saving best network with accuracy: %02.2f%%' % (acc, ))
    print('[model] saving best weights to %s' % (weights_file))
    with open(weights_file, 'wb') as pfile:
        pickle.dump(best_weights, pfile, protocol=pickle.HIGHEST_PROTOCOL)
    print('[model] ...saved\n')


def shock_network(output_layer, voltage=0.10):
    print('[model] shocking the network with voltage: %0.2f%%' % (voltage, ))
    current_weights = layers.get_all_param_values(output_layer)
    for index in range(len(current_weights)):
        temp = current_weights[index] * voltage
        shock = np.random.uniform(low=-1.0, high=1.0, size=temp.shape)
        temp *= shock
        # Apply shock
        current_weights[index] += temp
    layers.set_all_param_values(output_layer, current_weights)
    print('[model] ...shocked')


def set_learning_rate(learning_rate, update):
    new_learning_rate = update(learning_rate.get_value())
    learning_rate.set_value(float32(new_learning_rate))
    print('\n[train] setting learning rate to %.9f' % (new_learning_rate))
    print_header_columns()
