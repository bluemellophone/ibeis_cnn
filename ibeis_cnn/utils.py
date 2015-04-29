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
#from sklearn.utils import shuffle
import cv2
import cPickle as pickle
import matplotlib.pyplot as plt
from operator import itemgetter
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.cm as cm
from os.path import join, exists
import utool as ut
import six
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


def convert_cv2_images_to_theano_images(img_list):
    """
    Converts a list of cv2-style images into a single numpy array of nonflat
    theano-style images.

    h=height, w=width, b=batchid, c=channel

    Args:
        img_list (list of ndarrays): a list of numpy arrays with shape [h, w, c]

    Returns:
        data: in the shape [b, (c x h x w)]

    CommandLine:
        python -m ibeis_cnn.utils --test-convert_cv2_images_to_theano_images

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_cnn.utils import *  # NOQA
        >>> import vtool as vt
        >>> # build test data
        >>> # execute function
        >>> img_list, width, height, channels = testdata_imglist()
        >>> data = convert_cv2_images_to_theano_images(img_list)
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


def convert_theano_images_to_cv2_images(data, width, height, channels):
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


def testdata_xy():
    img_list, width, height, channels = testdata_imglist()
    data = np.array((img_list * 20))
    labels = np.random.rand(len(data) / 2) > .5
    data_per_label = 2
    return data, labels, data_per_label


def train_test_split(X, y, eval_size, data_per_label=1):
    r"""
    Args:
        X (ndarray):
        y (ndarray):
        eval_size (?):

    Returns:
        tuple: (X_train, y_train, X_valid, y_valid)

    CommandLine:
        python -m ibeis_cnn.utils --test-train_test_split

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.utils import *  # NOQA
        >>> # build test data
        >>> data, labels, data_per_label = testdata_xy()
        >>> X = data
        >>> y = labels
        >>> eval_size = .2
        >>> # execute function
        >>> (X_train, y_train, X_valid, y_valid) = train_test_split(X, y, eval_size, data_per_label)
        >>> # verify results
        >>> result = str((X_train, y_train, X_valid, y_valid))
        >>> print(result)
    """
    # take the data and label arrays, split them preserving the class distributions
    assert len(X) == len(y) * data_per_label
    nfolds = round(1. / eval_size)
    kf = StratifiedKFold(y, nfolds)
    train_indices, valid_indices = six.next(iter(kf))
    data_train_indicies = expand_data_indicies(train_indices, data_per_label)
    data_valid_indicies = expand_data_indicies(valid_indices, data_per_label)
    X_train = X.take(data_train_indicies, axis=0)
    X_valid = X.take(data_valid_indicies, axis=0)
    y_train = y.take(train_indices, axis=0)
    y_valid = y.take(valid_indices, axis=0)
    return X_train, y_train, X_valid, y_valid


def write_data_and_labels(data, labels, data_fpath, labels_fpath):
    print('[write_data_and_labels] np.shape(data) = %r' % (np.shape(data),))
    print('[write_data_and_labels] np.shape(labels) = %r' % (np.shape(labels),))
    # to resize the images back to their 2D-structure:
    # X = images_array.reshape(-1, 3, 48, 48)
    print('[write_data_and_labels] writing training data to %s...' % (data_fpath))
    with open(data_fpath, 'wb') as ofile:
        np.save(ofile, data)

    print('[write_data_and_labels] writing training labels to %s...' % (labels_fpath))
    with open(labels_fpath, 'wb') as ofile:
        np.save(ofile, labels)


def load(data_fpath, labels_fpath=None, random_state=None):
    # Load X matrix (data)
    data = np.load(data_fpath, mmap_mode='r')
    # Load y vector (labels)
    labels = None
    if labels_fpath is not None:
        labels = np.load(labels_fpath, mmap_mode='r')
    # Return data
    return data, labels


def print_header_columns():
    print('''
[info]   Epoch |  Train Loss (determ)  |  Valid Loss  |  Train / Val (determ)  |  Valid Acc  |  Test Acc   |  Dur
[info] --------|-----------------------|--------------|------------------------|-------------|-------------|------\
''')


def print_layer_info(output_layer):
    nn_layers = layers.get_all_layers(output_layer)
    print('\n[info] Network Structure:')
    for index, layer in enumerate(nn_layers):
        output_shape = layer.get_output_shape()
        print('[info]  {:>3}  {:<18}\t{:<20}\tproduces {:>7,} outputs'.format(
            index,
            layer.__class__.__name__,
            str(output_shape),
            int(str(functools.reduce(operator.mul, output_shape[1:]))),
        ))
    print('[info] ...this model has {:,} learnable parameters\n'.format(
        layers.count_params(output_layer)
    ))


def print_epoch_info(train_loss, train_determ_loss, valid_loss, valid_accuracy,
                     test_accuracy, epoch, duration, best_train_loss, best_valid_loss,
                     best_valid_accuracy, best_test_accuracy, **kwargs):
    best_train      = train_loss == best_train_loss
    best_valid      = valid_loss == best_valid_loss
    best_valid_accuracy = valid_accuracy == best_valid_accuracy
    best_train_accuracy = test_accuracy == best_test_accuracy
    ratio           = train_loss / valid_loss
    ratio_determ    = None if train_determ_loss is None else train_determ_loss / valid_loss
    unhealthy_ratio = ratio <= 0.5 or 2.0 <= ratio
    print('[info]  {:>5}  |  {}{:<19}{}  |  {}{:>10.6f}{}  '
          '|  {}{:<20}{}  |  {}{:>9}{}  |  {}{:>9}{}  |  {:>3.1f}s'.format(
              epoch,
              ANSI.BLUE if best_train else '',
              '%0.6f' % (train_loss, ) if train_determ_loss is None else '%0.6f (%0.6f)' % (train_loss, train_determ_loss),
              ANSI.RESET if best_train else '',
              ANSI.GREEN if best_valid else '',
              valid_loss,
              ANSI.RESET if best_valid else '',
              ANSI.RED if unhealthy_ratio else '',
              '%0.6f' % (ratio, ) if ratio_determ is None else '%0.6f (%0.6f)' % (ratio, ratio_determ),
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


def expand_data_indicies(indices, data_per_label=1):
    expanded_indicies = [indices * data_per_label + count for count in range(data_per_label)]
    data_indices = np.vstack(expanded_indicies).T.flatten()
    return data_indices


def make_random_indicies(num):
    randstate = np.random.RandomState(RANDOM_SEED)
    random_indicies = np.arange(num)
    randstate.shuffle(random_indicies)
    return random_indicies


def data_label_shuffle(X, y, data_per_label=1):
    r"""
    CommandLine:
        python -m ibeis_cnn.utils --test-data_label_shuffle

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.utils import *  # NOQA
        >>> # build test data
        >>> X, y, data_per_label = testdata_xy()
        >>> # execute function
        >>> result = data_label_shuffle(X, y)
        >>> # verify results
        >>> print(result)
    """
    random_indicies = make_random_indicies(len(X) // data_per_label)
    data_random_indicies = expand_data_indicies(random_indicies, data_per_label)
    X = np.ascontiguousarray(X.take(data_random_indicies, axis=0))
    if y is not None:
        y =  np.ascontiguousarray(y.take(random_indicies, axis=0))
    return X, y


def batch_iterator(X, y, batch_size, encoder=None, rand=False, augment=None,
                   center_mean=None, center_std=None, model=None, **kwargs):
    r"""
    Args:
        X (ndarray):
        y (ndarray):
        batch_size (int):
        encoder (None):
        rand (bool):
        augment (None):
        center_mean (None):
        center_std (None):

    CommandLine:
        python -m ibeis_cnn.utils --test-batch_iterator

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.utils import *  # NOQA
        >>> # build test data
        >>> X = np.random.rand(64, 3, 5, 4)
        >>> y = (np.random.rand(64) * 4).astype(np.int32)
        >>> batch_size = 16
        >>> encoder = None
        >>> rand = True
        >>> augment = None
        >>> center_mean = None
        >>> center_std = None
        >>> data_per_label = 2
        >>> model = None
        >>> # execute function
        >>> result = batch_iterator(X, y, batch_size, encoder, rand, augment, center_mean, center_std)
        >>> # verify results
        >>> print(next(result))
    """
    verbose = kwargs.get('verbose', ut.VERYVERBOSE)
    data_per_label = getattr(model, 'data_per_label', 1) if model is not None else 1
    # divides X and y into batches of size bs for sending to the GPU
    if rand:
        # Randomly shuffle data
        X, y = data_label_shuffle(X, y, data_per_label)
    if verbose:
        print('[batchiter] BEGIN')
        #print('[batchiter] X.flags \n%r' % (X.flags, ))
        print('[batchiter] X.shape %r' % (X.shape, ))
        if y is not None:
            print('[batchiter] y.shape %r' % (y.shape, ))
    if y is not None:
        assert X.shape[0] == (y.shape[0] * data_per_label), 'bad data / label alignment'
    #N = (X).shape[0] // data_per_label
    N = (X).shape[0]
    num_batches = (N + batch_size - 1) // batch_size
    #num_batches -= 2
    if verbose:
        print('[batchiter] num_batches = %r' % (num_batches,))
    for i in range(num_batches):
        start_x = i * batch_size
        end_x = (i + 1) * batch_size
        #x_sl = slice(start_y * data_per_label, end_y * data_per_label)
        #y_sl = slice(start_y, end_y)
        # Take full batch of images and take the fraction of labels if data_per_label > 1
        x_sl = slice(start_x, end_x)
        y_sl = slice(start_x // data_per_label, end_x // data_per_label)
        Xb = X[x_sl]
        if y is not None:
            yb = y[y_sl]
        else:
            yb = None
        # Get corret dtype for X
        Xb = Xb.astype(np.float32)
        # Whiten
        if center_mean is not None:
            Xb -= center_mean
        if center_std is not None and center_std != 0.0:
            Xb /= center_std
        # Augment
        if augment is not None:
            Xb_ = np.copy(Xb)
            yb_ = None if yb is None else np.copy(yb)
            Xb, yb = augment(Xb_, yb_)
        # Encode
        if encoder is not None and yb is not None:
            yb = encoder.transform(yb)
        # Get corret dtype for y (after encoding)
        if yb is not None:
            if data_per_label > 1:
                # TODO: FIX data_per_label ISSUES
                yb_buffer = -np.ones(len(yb) * (data_per_label - 1), np.int32)
                yb = np.vstack((yb, yb_buffer)).T.flatten()
            yb = yb.astype(np.int32)
        # Convert cv2 format to Lasagne format for batching
        Xb = Xb.transpose((0, 3, 1, 2))
        if verbose:
            print('[batchiter] Yielding batch:')
            print('[batchiter]   * x_sl = %r' % (x_sl,))
            print('[batchiter]   * y_sl = %r' % (y_sl,))
            print('[batchiter]   * Xb.shape = %r' % (Xb.shape,))
            print('[batchiter]   * yb.shape = %r' % (yb.shape,))
        # Ugg, we can't have data and labels of different lengths
        yield Xb, yb
    if verbose:
        print('[batchiter] END')


def multinomial_nll(x, t):
    #coding_dist=x, true_dist=t
    return T.nnet.categorical_crossentropy(x, t)


def create_theano_funcs(learning_rate_theano, output_layer, model, momentum=0.9,
                          input_type=T.tensor4, output_type=T.ivector,
                          regularization=None, **kwargs):
    """
    build the Theano functions (symbolic expressions) that will be used in the
    optimization refer to this link for info on tensor types:

    References:
        http://deeplearning.net/software/theano/library/tensor/basic.html
    """
    X = input_type('x')
    y = output_type('y')
    X_batch = input_type('x_batch')
    y_batch = output_type('y_batch')

    # Defaults that are overwritable by a model
    loss_function = multinomial_nll
    if model is not None and hasattr(model, 'loss_function'):
        loss_function = model.loss_function

    # we are minimizing the multi-class negative log-likelihood
    objective = objectives.Objective(output_layer, loss_function=loss_function)
    loss = objective.get_loss(X_batch, target=y_batch)
    loss_determ = objective.get_loss(X_batch, target=y_batch, deterministic=True)

    # Regularize
    if regularization is not None:
        L2 = lasagne.regularization.l2(output_layer)
        loss += L2 * regularization

    # Run inference and get performance
    probabilities = output_layer.get_output(X_batch, deterministic=True)
    predictions = T.argmax(probabilities, axis=1)
    confidences = probabilities.max(axis=1)
    # accuracy = T.mean(T.eq(predictions, y_batch))
    performance = [probabilities, predictions, confidences]  # , accuracy]

    # Define how to update network parameters based on the training loss
    parameters = layers.get_all_params(output_layer)
    updates = lasagne.updates.nesterov_momentum(loss, parameters, learning_rate_theano, momentum)

    theano_forward = theano.function(
        inputs=[theano.Param(X_batch), theano.Param(y_batch)],
        outputs=[loss_determ] + performance ,
        givens={
            X: X_batch,
            y: y_batch,
        },
    )

    theano_backprop = theano.function(
        inputs=[theano.Param(X_batch), theano.Param(y_batch)],
        outputs=[loss] + performance,
        updates=updates,
        givens={
            X: X_batch,
            y: y_batch,
        },
    )

    return theano_forward, theano_backprop


def process_batch(X_train, y_train, theano_fn, **kwargs):
    """
        compute the loss over all training batches

        Jon, if you get to this before I do, please fix. -J
    """
    albl_list = []  # [a]ugmented [l]a[b]e[l] list
    loss_list = []
    prob_list = []
    pred_list = []
    conf_list = []
    accu_list = []
    show = True
    for Xb, yb in batch_iterator(X_train, y_train, **kwargs):
        # Runs a batch through the network and updates the weights. Just returns what it did
        loss, prob, pred, conf = theano_fn(Xb, yb)
        if yb is not None:
            accu = np.mean(np.equal(pred, yb))
        else:
            accu = [np.nan] * len(Xb)
        albl_list.append(yb)
        loss_list.append(loss)
        prob_list.append(loss)
        pred_list.append(pred)
        conf_list.append(conf)
        accu_list.append(accu)
        if show:
            # Print the network output for the first batch
            print('--------------')
            print('Predect: ', pred)
            print('Correct: ', yb)
            print('Loss:    ', loss)
            print('Prob:    ', prob)
            print('Conf:    ', conf)
            print('Accu:    ', accu)
            print('--------------')
            show = False
    # Convert to numpy array
    albl_list = np.hstack(albl_list)
    loss_list = np.hstack(loss_list)
    prob_list = np.vstack(prob_list)
    pred_list = np.hstack(pred_list)
    conf_list = np.hstack(conf_list)
    accu_list = np.hstack(accu_list)
    print(albl_list.shape)
    print(loss_list.shape)
    print(prob_list.shape)
    print(pred_list.shape)
    print(conf_list.shape)
    print(accu_list.shape)
    # Return
    return albl_list, loss_list, prob_list, pred_list, conf_list, accu_list


def process_train(X_train, y_train, theano_fn, **kwargs):
    """ compute the loss over all training batches """
    results = process_batch(X_train, y_train, theano_fn, **kwargs)
    albl_list, loss_list, prob_list, pred_list, conf_list, accu_list = results
    # Find whatever metrics we want
    avg_train_loss = np.mean(loss_list)
    return avg_train_loss


def process_valid(X_valid, y_valid, theano_fn, **kwargs):
    """ compute the loss over all validation batches """
    results = process_batch(X_valid, y_valid, theano_fn, **kwargs)
    albl_list, loss_list, prob_list, pred_list, conf_list, accu_list = results
    # Find whatever metrics we want
    avg_valid_loss = np.mean(loss_list)
    avg_valid_accu = np.mean(accu_list)
    return avg_valid_loss, avg_valid_accu


def process_test(X_test, y_test, theano_fn, results_path=None, **kwargs):
    """ compute the loss over all test batches """
    results = process_batch(X_test, y_test, theano_fn, **kwargs)
    albl_list, loss_list, prob_list, pred_list, conf_list, accu_list = results
    # Find whatever metrics we want
    avg_test_accu = np.mean(accu_list)
    # Output confusion matrix
    if results_path is not None:
        # Grab model
        model = kwargs.get('model', None)
        mapping_fn = None
        if model is not None:
            mapping_fn = getattr(model, 'label_order_mapping', None)
        # TODO: THIS NEEDS TO BE FIXED
        label_list = list(range(kwargs.get('output_dims')))
        # Encode labels if avaialble
        encoder = kwargs.get('encoder', None)
        if encoder is not None:
            label_list = encoder.inverse_transform(label_list)
        # Make confusion matrix (pass X to write out failed cases)
        show_confusion_matrix(albl_list, pred_list, label_list, results_path,
                              mapping_fn, X_test)
    return avg_test_accu


def process_predictions(X_test, theano_fn, **kwargs):
    """ compute the loss over all test batches """
    results = process_batch(X_test, None, theano_fn, **kwargs)
    albl_list, loss_list, prob_list, pred_list, conf_list, accu_list = results
    # Find whatever metrics we want
    encoder = kwargs.get('encoder', None)
    if encoder is not None:
        label_list = encoder.inverse_transform(pred_list)
    else:
        label_list = [None] * len(pred_list)
    return pred_list, label_list


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


def show_confusion_matrix(correct_y, predict_y, category_list, results_path,
                          mapping_fn=None, data_x=None):
    """
    Given the correct and predict labels, show the confusion matrix

    Args:
        correct_y (list of int): the list of correct labels
        predict_y (list of int): the list of predict assigned labels
        category_list (list of str): the category list of all categories

    Displays:
        matplotlib: graph of the confusion matrix

    Returns:
        None
    """
    confused_examples = join(results_path, 'confused')
    if data_x is not None:
        if exists(confused_examples):
            ut.remove_dirs(confused_examples, quiet=True)
        ut.ensuredir(confused_examples)
    size = len(category_list)

    if mapping_fn is None:
        # Identity
        category_mapping = { key: index for index, key in enumerate(category_list) }
        category_list_ = category_list
    else:
        category_mapping = mapping_fn(category_list)
        assert all([ category in category_mapping.keys() for category in category_list ]), 'Not all categories are mapped'
        values = list(category_mapping.values())
        assert len(list(set(values))) == len(values), 'Mapped categories have a duplicate assignment'
        assert 0 in values, 'Mapped categories must have a 0 index'
        temp = list(category_mapping.iteritems())
        temp = sorted(temp, key=itemgetter(1))
        category_list_ = [ t[0] for t in temp ]

    confidences = np.zeros((size, size))
    counters = {}
    for index, (correct, predict) in enumerate(zip(correct_y, predict_y)):
        # Ensure type
        correct = int(correct)
        predict = int(predict)
        # Get the "text" label
        example_correct_label = category_list[correct]
        example_predict_label = category_list[predict]
        # Perform any mapping that needs to be done
        correct_ = category_mapping[example_correct_label]
        predict_ = category_mapping[example_predict_label]
        # Add to the confidence matrix
        confidences[correct_][predict_] += 1
        if data_x is not None and correct_ != predict_:
            example = data_x[index]
            example_name = '%s^SEEN_INCORRECTLY_AS^%s' % (example_correct_label, example_predict_label, )
            if example_name not in counters.keys():
                counters[example_name] = 0
            counter = counters[example_name]
            counters[example_name] += 1
            example_name = '%s^%d.png' % (example_name, counter)
            example_path = join(confused_examples, example_name)
            cv2.imwrite(example_path, example)

    row_sums = np.sum(confidences, axis=1)
    norm_conf = (confidences.T / row_sums).T

    fig = plt.figure(1)
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
    plt.xticks(np.arange(size), category_list_[0:size], rotation=90)
    plt.yticks(np.arange(size), category_list_[0:size])
    margin_small = 0.1
    margin_large = 0.9
    plt.subplots_adjust(left=margin_small, right=margin_large, bottom=margin_small, top=margin_large)
    plt.xlabel('Predicted')
    plt.ylabel('Correct')
    plt.savefig(join(results_path, 'confusion.png'))


def show_convolutional_layers(output_layer, results_path, color=False, limit=150, target=None, epoch=None):
    nn_layers = layers.get_all_layers(output_layer)
    cnn_layers = []
    for layer in nn_layers:
        layer_type = str(type(layer))
        # Only print convolutional layers
        if 'Conv2DCCLayer' not in layer_type:
            continue
        cnn_layers.append(layer)

    weights_list = [layer.W.get_value() for layer in cnn_layers]
    show_convolutional_features(weights_list, results_path, color=color, limit=limit, target=target, epoch=epoch)


def show_convolutional_features(weights_list, results_path, color=False, limit=150, target=None, epoch=None):
    for index, all_weights in enumerate(weights_list):
        if target is not None and target != index:
            continue
        # re-use the same figure to save memory
        fig = plt.figure(1)
        ax1 = plt.axes(frameon=False)
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        # Get shape of weights
        num, channels, height, width = all_weights.shape
        # non-color features need to be flattened
        if not color:
            all_weights = all_weights.reshape(num * channels, height, width)
            num, height, width = all_weights.shape
        # Limit all_weights
        if limit is not None and num > limit:
            all_weights = all_weights[:limit]
            if color:
                num, channels, height, width = all_weights.shape
            else:
                num, height, width = all_weights.shape
        # Find how many features and build grid
        dim = int(np.ceil(np.sqrt(num)))
        grid = ImageGrid(fig, 111, nrows_ncols=(dim, dim))

        # Build grid
        for f, feature in enumerate(all_weights):
            # get all the weights and scale them to dimensions that can be shown
            if color:
                feature = feature[::-1]  # Rotate BGR to RGB
                feature = cv2.merge(feature)
            fmin, fmax = np.min(feature), np.max(feature)
            domain = fmax - fmin
            feature = (feature - fmin) * (255. / domain)
            feature = feature.astype(np.uint8)
            if color:
                grid[f].imshow(feature, interpolation='nearest')
            else:
                grid[f].imshow(feature, cmap=cm.Greys_r, interpolation='nearest')

        for j in range(dim * dim):
            grid[j].get_xaxis().set_visible(False)
            grid[j].get_yaxis().set_visible(False)

        color_str = 'color' if color else 'gray'
        if epoch is None:
            epoch = 'X'
        output_file = 'features_conv%d_epoch_%s_%s.png' % (index, epoch, color_str)
        output_path = join(results_path, output_file)
        plt.savefig(output_path, bbox_inches='tight')

        output_file = 'features_conv%d_%s.png' % (index, color_str)
        output_path = join(results_path, output_file)
        plt.savefig(output_path, bbox_inches='tight')


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
