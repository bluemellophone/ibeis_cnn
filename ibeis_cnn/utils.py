# utils.py
# provides utilities for learning a neural network model
from __future__ import absolute_import, division, print_function
import time
import numpy as np
import functools
import operator

import theano
import theano.tensor as T
import lasagne
from lasagne import objectives

from lasagne import layers
from sklearn.cross_validation import StratifiedKFold
from sklearn.utils import shuffle
import cv2
import cPickle as pickle
import matplotlib.pyplot as plt
from os.path import join
import utool as ut
#from six.moves import range, zip


# NOTE: can use pygments instead
class ANSI(object):
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


def _update(kwargs, key, value):
    #if key not in kwargs.keys():
    if key not in kwargs:
        kwargs[key] = value


def testdata_imglist():
    import vtool as vt
    x = 32
    width = 32
    height = 32
    channels = 3
    img0 = np.arange(x ** 2 * 3, dtype=np.uint8).reshape(x, x, 3)
    img1 = vt.imread(ut.grab_test_imgpath('jeff.png'))
    img2 = vt.imread(ut.grab_test_imgpath('carl.jpg'))
    img3 = vt.imread(ut.grab_test_imgpath('lena.png'))
    img_list = [vt.padded_resize(img, (width, height)) for img in [img0, img1, img2, img3]]
    return img_list, width, height, channels


def convert_imagelist_to_data(img_list):
    """
    Converts a list of cv2-style images into a single numpy array of nonflat
    theano-style images.

    h=height, w=width, b=batchid, c=channel

    Args:
        img_list (list of ndarrays): a list of numpy arrays with shape [h, w, c]

    Returns:
        data: in the shape [b, (c x h x w)]

    CommandLine:
        python -m ibeis_cnn.utils --test-convert_imagelist_to_data

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_cnn.utils import *  # NOQA
        >>> import vtool as vt
        >>> # build test data
        >>> # execute function
        >>> img_list, width, height, channels = testdata_imglist()
        >>> data = convert_imagelist_to_data(img_list)
        >>> data[0].reshape(3, 32, 32)[:, 0:2, 0:2]
        >>> subset = (data[0].reshape(3, 32, 32)[:, 0:2, 0:2])
        >>> #result = str(np.transpose(subset, (1, 2, 0)))
        >>> result = str(subset).replace('\n', '')
        >>> print(result)
        [[[  0   3]  [ 96  99]] [[  1   4]  [ 97 100]] [[  2   5]  [ 98 101]]]
    """
    #[img.shape for img in img_list]
    # format to [b, c, h, w]
    shape_list = [img.shape for img in img_list]
    assert ut.list_allsame(shape_list)
    theano_style_imgs = [np.transpose(img, (2, 0, 1))[None, :] for img in img_list]
    data = np.vstack(theano_style_imgs)
    #data = np.vstack([img[None, :] for img in img_list])
    return data


def convert_data_to_imglist(data, width, height, channels):
    r"""
    Args:
        data (ndarray): in the shape [b, (c x h x w)]
        width (?):
        height (?):
        channels (?):

    Returns:
        img_list (list of ndarrays): a list of numpy arrays with shape [h, w, c]

    CommandLine:
        python -m ibeis_cnn.utils --test-convert_data_to_imglist

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_cnn.utils import *  # NOQA
        >>> # build test data
        >>> img_list, width, height, channels = testdata_imglist()
        >>> data = convert_imagelist_to_data(img_list)
        >>> img_list2 = convert_data_to_imglist(data, width, height, channels)
        >>> assert np.all(img_list == img_list2)
    """
    #num_imgs = data.shape[0]
    #newshape = (num_imgs, channels, width, height)
    #data_ = data.reshape(newshape)
    img_list = np.transpose(data, (0, 2, 3, 1))
    return img_list


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
[info]   Epoch |  Train Loss  |  Valid Loss  |  Train / Val  |  Valid Acc  |  Test Acc   |  Dur
[info] --------|--------------|--------------|---------------|-------------|-------------|------\
''')


def print_layer_info(output_layer):
    nn_layers = layers.get_all_layers(output_layer)[::-1]
    print('\n[info] Network Structure:')
    for layer in nn_layers[::-1]:
        output_shape = layer.get_output_shape()
        print('[info]     {:<18}\t{:<20}\tproduces {:>7,} outputs'.format(
            layer.__class__.__name__,
            str(output_shape),
            int(str(functools.reduce(operator.mul, output_shape[1:]))),
        ))
    print('[info] ...this model has {:,} learnable parameters\n'.format(
        layers.count_params(output_layer)
    ))


def print_epoch_info(train_loss, valid_loss, valid_accuracy, test_accuracy,
                     epoch, duration, best_train_loss, best_valid_loss,
                     best_valid_accuracy, best_test_accuracy, **kwargs):
    best_train      = train_loss == best_train_loss
    best_valid      = valid_loss == best_valid_loss
    best_valid_accuracy = valid_accuracy == best_valid_accuracy
    best_train_accuracy = test_accuracy == best_test_accuracy
    ratio           = train_loss / valid_loss
    unhealthy_ratio = ratio <= 0.5 or 2.0 <= ratio
    print('[info]  {:>5}  |  {}{:>10.6f}{}  |  {}{:>10.6f}{}  '
          '|  {}{:>11.6f}{}  |  {}{:>9}{}  |  {}{:>9}{}  |  {:>3.1f}s'.format(
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
              ANSI.MAGENTA if best_valid_accuracy else '',
              '{:.2f}%'.format(valid_accuracy * 100),
              ANSI.RESET if best_valid_accuracy else '',
              ANSI.CYAN if best_train_accuracy else '',
              '{:.2f}%'.format(test_accuracy * 100) if test_accuracy is not None else '',
              ANSI.RESET if best_train_accuracy else '',
              duration,
          ))


def float32(k):
    return np.cast['float32'](k)


def batch_iterator(X, y, batch_size, encoder=None, rand=False, augment=None,
                   center_mean=None, center_std=None, **kwargs):
    # divides X and y into batches of size bs for sending to the GPU
    # Randomly shuffle data
    if rand:
        if y is None:
            X = shuffle(X, random_state=RANDOM_SEED)
        else:
            X, y = shuffle(X, y, random_state=RANDOM_SEED)
    N = X.shape[0]
    for i in range((N + batch_size - 1) // batch_size):
        sl = slice(i * batch_size, (i + 1) * batch_size)
        Xb = X[sl]
        if y is not None:
            yb = y[sl]
        else:
            yb = None
        # Get corret dtype for X
        Xb = Xb.astype(np.float32)
        # Whiten)
        if center_mean is not None:
            Xb -= center_mean
        if center_std is not None and center_std != 0.0:
            Xb /= center_std
        # Augment
        if augment is not None:
            Xb, yb = augment(Xb, yb)
        # Encode
        if encoder is not None:
            yb = encoder.transform(yb)
        # Get corret dtype for y (after encoding)
        yb = yb.astype(np.int32)
        yield Xb, yb


def multinomial_nll(x, t):
    return T.nnet.categorical_crossentropy(x, t)


def create_iter_funcs(learning_rate_theano, output_layer, momentum=0.9,
                      input_type=T.tensor4, output_type=T.ivector,
                      regularization=None, **kwargs):
    """
    build the Theano functions that will be used in the optimization
    refer to this link for info on tensor types:

    References:
        http://deeplearning.net/software/theano/library/tensor/basic.html
    """
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
        loss_train, all_params, learning_rate_theano, momentum)

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

    test_iter = theano.function(
        inputs=[theano.Param(X_batch), theano.Param(y_batch)],
        outputs=[predict_proba, pred, accuracy],
        givens={
            X: X_batch,
            y: y_batch,
        },
    )

    return train_iter, valid_iter, test_iter


def forward_train(X_train, y_train, train_iter, rand=False, augment=None, **kwargs):
    """ compute the loss over all training batches """
    train_losses = []
    for Xb, yb in batch_iterator(X_train, y_train, rand=rand, augment=augment, **kwargs):
        batch_train_loss = train_iter(Xb, yb)
        train_losses.append(batch_train_loss)
    avg_train_loss = np.mean(train_losses)
    return avg_train_loss


def forward_valid(X_valid, y_valid, valid_iter, rand=False, augment=None, **kwargs):
    """ compute the loss over all validation batches """
    valid_losses = []
    valid_accuracies = []
    for Xb, yb in batch_iterator(X_valid, y_valid, rand=rand, augment=augment, **kwargs):
        batch_valid_loss, batch_accuracy = valid_iter(Xb, yb)
        valid_losses.append(batch_valid_loss)
        valid_accuracies.append(batch_accuracy)
    avg_valid_loss = np.mean(valid_losses)
    avg_valid_accuracy = np.mean(valid_accuracies)
    return avg_valid_loss, avg_valid_accuracy


def forward_test(X_test, y_test, test_iter, show=False, confusion=True, **kwargs):
    """ compute the loss over all test batches """
    all_correct = []
    all_predict = []
    test_accuracies = []
    for Xb, yb in batch_iterator(X_test, y_test, **kwargs):
        batch_predict_proba, batch_pred, batch_accuracy = test_iter(Xb, yb)
        test_accuracies.append(batch_accuracy)
        all_correct.append(yb)
        all_predict.append(batch_pred)
        if show:
            print('Predect: ', batch_pred)
            print('Correct: ', yb)
            print('--------------')
            show = False
    avg_test_accuracy = np.mean(test_accuracies)
    if confusion:
        all_correct = np.hstack(all_correct)
        all_predict = np.hstack(all_predict)
        labels = list(range(kwargs.get('output_dims')))
        encoder = kwargs.get('encoder', None)
        if encoder is not None:
            labels = encoder.inverse_transform(labels)
        show_confusion_matrix(all_correct, all_predict, labels)
    return avg_test_accuracy


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


def save_model(kwargs, weights_file):
    print('[model] saving best weights to %s' % (weights_file))
    with open(weights_file, 'wb') as pfile:
        pickle.dump(kwargs, pfile, protocol=pickle.HIGHEST_PROTOCOL)
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


def set_learning_rate(learning_rate_theano, update):
    new_learning_rate_theano = update(learning_rate_theano.get_value())
    learning_rate_theano.set_value(float32(new_learning_rate_theano))
    print('\n[train] setting learning rate to %.9f' % (new_learning_rate_theano))
    print_header_columns()


def show_confusion_matrix(correct_y, expert_y, category_list):
    """
    Given the correct and expert labels, show the confusion matrix

    Args:
        correct_y (list of int): the list of correct labels
        expert_y (list of int): the list of expert assigned labels
        category_list (list of str): the category list of all categories

    Displays:
        matplotlib: graph of the confusion matrix

    Returns:
        None
    """
    size = len(category_list)
    confidences = np.zeros((size, size))
    for correct, expert, in zip(correct_y, expert_y):
        confidences[correct][expert] += 1

    row_sums = np.sum(confidences, axis=1)
    norm_conf = (confidences.T / row_sums).T

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet,
                    interpolation='nearest')

    for x in range(size):
        for y in range(size):
            ax.annotate(str(int(confidences[x][y])), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')

    cb = fig.colorbar(res)  # NOQA
    plt.xticks(np.arange(size), category_list[0:size], rotation=90)
    plt.yticks(np.arange(size), category_list[0:size])
    plt.xlabel('Predicted')
    plt.ylabel('Correct')
    plt.savefig(join('..', 'confusion.png'))


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis_cnn.utils
        python -m ibeis_cnn.utils --allexamples
        python -m ibeis_cnn.utils --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
