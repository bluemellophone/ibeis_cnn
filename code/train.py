#!/usr/bin/env python

# train.py
# constructs the Theano optimization and trains a learning model,
# optionally by initializing the network with pre-trained weights.

# our own imports
import info
import utils
import model

# module imports
import time
import theano   # NOQA
import theano.tensor as T
import itertools
import numpy as np
import cPickle as pickle
import lasagne

from lasagne import layers
from lasagne import objectives
from lasagne import nonlinearities  # NOQA

from os.path import join, abspath
import cv2
import random


# divides X and y into batches of size bs for sending to the GPU
def batch_iterator(X, y, bs, norm=None, mean=None, std=None, augment=None):
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
        # Whiten
        if mean is not None:
            Xb -= mean
        if std is not None:
            Xb /= std
        if norm is not None and norm > 0.0:
            Xb_ /= norm
        # Augment
        if augment is not None:
            Xb, yb = augment(Xb, yb)
        yield Xb_, yb_


def multinomial_nll(x, t):
    return T.nnet.categorical_crossentropy(x, t)


# build the Theano functions that will be used in the optimization
# refer to this link for info on tensor types:
# http://deeplearning.net/software/theano/library/tensor/basic.html
def create_iter_funcs(learning_rate, momentum, output_layer, input_type=T.tensor4, output_type=T.ivector):
    X = input_type('x')
    y = output_type('y')
    X_batch = input_type('x_batch')
    y_batch = output_type('y_batch')

    # we are minimizing the multi-class negative log-likelihood
    objective = objectives.Objective(output_layer, loss_function=multinomial_nll)

    loss_train = objective.get_loss(X_batch, target=y_batch)
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
    additional = 5 + 3 + 3 + 2
    points, channels, height, width = data.shape
    dtype = data.dtype
    data_channels = np.empty((points, channels + additional, height, width), dtype=dtype)
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
    add_to_template(template, 0, 0, replicate(image[0], 0))
    add_to_template(template, 1, 0, replicate(image[1], 1))
    add_to_template(template, 2, 0, replicate(image[2], 2))
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


def augmentation(Xb, yb):
    # label_map = {
    #     0:  4,
    #     1:  5,
    #     2:  6,
    #     3:  7,
    #     8:  12,
    #     9:  13,
    #     10: 14,
    #     11: 15,
    # }
    label_map = { x: x + 4 for x in range(0, 4) + range(8, 12) }
    # Apply inverse
    for key in label_map.keys():
        label = label_map[key]
        label_map[label] = key
    # Map
    points, channels, height, width = Xb.shape
    for index in range(points):
        if random.uniform(0.0, 1.0) <= 0.5:
            Xb[index] = Xb[index, :, ::-1]
            yb[index] = label_map[yb[index]]
    return Xb, yb


def train(data_file, labels_file, trained_weights_file=None, pretrained_weights_file=None):
    current_time = utils.get_current_time()
    if trained_weights_file is None:
        trained_weights_file = '%s.pickle' % (current_time)

    learning_rate_schedule = {
        0: 0.03,
        25: 0.003,
        50: 0.0003,
    }

    learning_rate = theano.shared(np.cast['float32'](learning_rate_schedule[0]))
    max_epochs = 75
    momentum   = 0.9
    batch_size = 128
    whiten     = True
    normalizer = 255.0
    output_dim = 16    # the number of outputs from the softmax layer (# classes)

    print('loading data...')
    data, labels = utils.load(data_file, labels_file)
    print('adding channels...')
    data = add_channels(data)
    print('  X.shape = %r' % (data.shape,))
    print('  X.dtype = %r' % (data.dtype,))
    print('  y.shape = %r' % (labels.shape,))
    print('  y.dtype = %r' % (labels.dtype,))

    show_image_from_data(data)

    print('building model...')
    input_cases, input_channels, input_height, input_width = data.shape
    output_layer = model.build_model(batch_size, input_width, input_height, input_channels, output_dim)
    info.print_layer_info(layers.get_all_layers(output_layer)[::-1])
    print('this model has %d learnable parameters' % (layers.count_params(output_layer)))

    if pretrained_weights_file is not None:
        print('loading pretrained weights from %s' % (pretrained_weights_file))
        with open(pretrained_weights_file, 'rb') as pfile:
            pretrained_weights = pickle.load(pfile)
            layers.set_all_param_values(output_layer, pretrained_weights)

    train_iter, valid_iter, predict_iter = create_iter_funcs(learning_rate, momentum, output_layer)

    print('creating train, validation datasaets...')
    X_train, y_train, X_valid, y_valid = utils.train_test_split(data, labels, eval_size=0.2)

    print('calculating whitening...')
    if whiten:
        whiten_mean = np.mean(X_train, axis=0)
        whiten_std  = np.std(X_train, axis=0)
    else:
        whiten_mean = None  # 0.0
        whiten_std  = None  # 1.0

    best_weights = None
    best_train_loss, best_valid_loss = np.inf, np.inf
    print('starting training at %s...' % (current_time))
    info.print_header_columns()

    try:
        for epoch in itertools.count(1):
            train_losses, valid_losses, valid_accuracies = [], [], []

            t0 = time.time()
            # compute the loss over all training batches
            for Xb, yb in batch_iterator(X_train, y_train, batch_size, normalizer, whiten_mean, whiten_std, augment=augmentation):
                batch_train_loss = train_iter(Xb, yb)
                train_losses.append(batch_train_loss)

            # compute the loss over all validation batches
            for Xb, yb in batch_iterator(X_valid, y_valid, batch_size, normalizer, whiten_mean, whiten_std):
                batch_valid_loss, batch_accuracy = valid_iter(Xb, yb)
                valid_losses.append(batch_valid_loss)
                valid_accuracies.append(batch_accuracy)

            # estimate the loss over all batches
            avg_train_loss = np.mean(train_losses)
            avg_valid_loss = np.mean(valid_losses)
            avg_valid_accuracy = np.mean(valid_accuracies)

            if np.isnan(avg_train_loss):
                print('training diverged')
                break

            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss
                best_weights = layers.get_all_param_values(output_layer)

            info.print_epoch_info(avg_valid_loss, best_valid_loss, avg_valid_accuracy, avg_train_loss, best_train_loss, epoch, time.time() - t0)

            # check if it's time to update the LR
            if epoch in learning_rate_schedule:
                new_learning_rate = learning_rate_schedule[epoch]
                learning_rate.set_value(new_learning_rate)
                print('\nsetting learning rate to %.6f\n' % (new_learning_rate))

            if epoch > max_epochs:
                print('\nmaximum number of epochs exceeded')
                print('saving best weights to %s' % (weights_file))
                with open(weights_file, 'wb') as pfile:
                    pickle.dump(best_weights, pfile, protocol=pickle.HIGHEST_PROTOCOL)
                break
    except KeyboardInterrupt:
        print('saving best weights to %s' % (weights_file))
        with open(weights_file, 'wb') as pfile:
            pickle.dump(best_weights, pfile, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    project_name = 'viewpoint'
    root = abspath(join('..', 'data'))
    train_data_file = join(root, 'numpy', project_name, 'X.npy')
    train_labels_file = join(root, 'numpy', project_name, 'y.npy')
    weights_file = join(root, 'nets', 'ibeis_cnn_weights.pickle')
    pretrained_weights_file = join(root, 'nets', 'pretrained_weights.pickle')
    train(train_data_file, train_labels_file, weights_file)
    #train(train_data_file, train_labels_file, weights_file, pretrained_weights_file)
