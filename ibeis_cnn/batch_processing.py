from __future__ import absolute_import, division, print_function
from ibeis_cnn import utils, draw_net
import utool as ut
import numpy as np


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
        X, y = utils.data_label_shuffle(X, y, data_per_label)
    if verbose:
        print('[batchiter] BEGIN')
        print('[batchiter] X.shape %r' % (X.shape, ))
        if y is not None:
            print('[batchiter] y.shape %r' % (y.shape, ))
    if y is not None:
        assert X.shape[0] == (y.shape[0] * data_per_label), 'bad data / label alignment'
    num_batches = (X.shape[0] + batch_size - 1) // batch_size
    if verbose:
        print('[batchiter] num_batches = %r' % (num_batches,))
    for batch_index in range(num_batches):
        # Get batch slice
        Xb, yb = utils.slice_data_labels(X, y, batch_size, batch_index, data_per_label)
        # Whiten
        Xb = utils.whiten_data(Xb, center_mean, center_std)
        # Augment
        if augment is not None:
            Xb_ = np.copy(Xb)
            yb_ = None if yb is None else np.copy(yb)
            Xb, yb = augment(Xb_, yb_)
        # Encode
        if yb is not None:
            if encoder is not None:
                yb = encoder.transform(yb)
            # Get corret dtype for y (after encoding)
            if data_per_label > 1:
                # TODO: FIX data_per_label ISSUES
                yb_buffer = -np.ones(len(yb) * (data_per_label - 1), np.int32)
                yb = np.hstack((yb, yb_buffer))
            yb = yb.astype(np.int32)
        # Convert cv2 format to Lasagne format for batching
        Xb = Xb.transpose((0, 3, 1, 2))
        if verbose:
            print('[batchiter] Yielding batch:')
            print('[batchiter]   * Xb.shape = %r' % (Xb.shape,))
            print('[batchiter]   * yb.shape = %r' % (yb.shape,))
        # Ugg, we can't have data and labels of different lengths
        yield Xb, yb
    if verbose:
        print('[batchiter] END')


def process_batch(X_train, y_train, theano_fn, **kwargs):
    """
        compute the loss over all training batches

        Jon, if you get to this before I do, please fix. -J
    """
    loss_list = []
    prob_list = []
    albl_list = []  # [a]ugmented [l]a[b]e[l] list
    pred_list = []
    conf_list = []
    show = False
    for Xb, yb in batch_iterator(X_train, y_train, **kwargs):
        # Runs a batch through the network and updates the weights. Just returns what it did
        loss, prob, pred, conf = theano_fn(Xb, yb)
        loss_list.append(loss)
        prob_list.append(prob)
        albl_list.append(yb)
        pred_list.append(pred)
        conf_list.append(conf)
        if show:
            # Print the network output for the first batch
            print('--------------')
            print('Loss:    ', loss)
            print('Prob:    ', prob)
            print('Correct: ', yb)
            print('Predect: ', pred)
            print('Conf:    ', conf)
            print('--------------')
            show = False
    # Convert to numpy array
    prob_list = np.vstack(prob_list)
    albl_list = np.hstack(albl_list)
    pred_list = np.hstack(pred_list)
    conf_list = np.hstack(conf_list)

    # Calculate performance
    loss = np.mean(loss_list)
    accu = np.mean(np.equal(albl_list, pred_list))

    # Return
    return loss, accu, prob_list, albl_list, pred_list, conf_list


def predict_batch(X_train, theano_fn, **kwargs):
    """
        compute the loss over all training batches

        Jon, if you get to this before I do, please fix. -J
    """
    prob_list = []
    pred_list = []
    conf_list = []
    for Xb, _ in batch_iterator(X_train, None, **kwargs):
        # Runs a batch through the network and updates the weights. Just returns what it did
        prob, pred, conf = theano_fn(Xb)
        prob_list.append(prob)
        pred_list.append(pred)
        conf_list.append(conf)
    # Convert to numpy array
    prob_list = np.vstack(prob_list)
    pred_list = np.hstack(pred_list)
    conf_list = np.hstack(conf_list)
    # Return
    return prob_list, pred_list, conf_list


def process_train(X_train, y_train, theano_fn, **kwargs):
    """ compute the loss over all training batches """
    results = process_batch(X_train, y_train, theano_fn, **kwargs)
    loss, accu, prob_list, albl_list, pred_list, conf_list = results
    # Return whatever metrics we want
    return loss


def process_valid(X_valid, y_valid, theano_fn, **kwargs):
    """ compute the loss over all validation batches """
    results = process_batch(X_valid, y_valid, theano_fn, **kwargs)
    loss, accu, prob_list, albl_list, pred_list, conf_list = results
    # rRturn whatever metrics we want
    return loss, accu


def process_test(X_test, y_test, theano_fn, results_path=None, **kwargs):
    """ compute the loss over all test batches """
    results = process_batch(X_test, y_test, theano_fn, **kwargs)
    loss, accu, prob_list, albl_list, pred_list, conf_list = results
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
        draw_net.show_confusion_matrix(albl_list, pred_list, label_list, results_path,
                                       mapping_fn, X_test)
    return accu


def process_predictions(X_test, theano_fn, **kwargs):
    """ compute the loss over all test batches """
    results = predict_batch(X_test, theano_fn, **kwargs)
    prob_list, pred_list, conf_list = results
    # Find whatever metrics we want
    encoder = kwargs.get('encoder', None)
    if encoder is not None:
        label_list = encoder.inverse_transform(pred_list)
    else:
        label_list = [None] * len(pred_list)
    return pred_list, label_list, conf_list
