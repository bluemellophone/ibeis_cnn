#!/usr/bin/env python
"""
tests a test set of data using a specified, pre0trained model and weights
"""
from __future__ import absolute_import, division, print_function
from ibeis_cnn import utils
from ibeis_cnn import models

import cPickle as pickle
from lasagne import layers

import utool as ut
import six  # NOQA
from os.path import join, abspath


def test(data_fpath, model, weights_fpath, results_dpath, **kwargs):
    """
    Driver function

    Args:
        data_fpath (?):
        labels_fpath (?):
        model (?):
        weights_fpath (?):
    """

    ######################################################################################

    # Load the data
    print('\n[data] loading data...')
    print('data_fpath = %r' % (data_fpath,))
    X_test, _ = utils.load(data_fpath, None)

    # Load the pretrained model if specified
    print('[model] loading pretrained weights from %s' % (weights_fpath))
    pretrained_weights = None
    with open(weights_fpath, 'rb') as pfile:
        kwargs = pickle.load(pfile)
        pretrained_weights = kwargs.get('best_weights', None)

    print('test kwargs = \n' + (ut.dict_str(kwargs)))

    # Build and print the model
    print('\n[model] building model...')
    input_cases, input_height, input_width, input_channels = kwargs.get('model_shape', None)  # SHOULD ERROR IF NOT SET
    output_layer = model.build_model(
        kwargs.get('batch_size'), input_width, input_height,
        input_channels, kwargs.get('output_dims'))
    utils.print_layer_info(output_layer)

    # Create the Theano primitives
    print('[model] creating Theano primitives...')
    test_iter = utils.create_testing_func(output_layer, **kwargs)

    # Set weights to model
    layers.set_all_param_values(output_layer, pretrained_weights)

    # Begin testing with the neural network
    vals = (utils.get_current_time(), kwargs.get('batch_size'), )
    print('\n[test] starting testing with batch size %0.1f' % vals)

    all_predict, labels = utils.forward_test_predictions(X_test, test_iter, results_dpath, **kwargs)
    print(all_predict)
    print(labels)


def test_pz():
    r"""
    CommandLine:
        python -m ibeis_cnn.test --test-test_pz

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.test import *  # NOQA
        >>> test_pz()
    """
    project_name            = 'plains'
    model                   = models.PZ_GIRM_Model()

    root                    = abspath(join('..', 'data'))
    test_data_fpath        = join(root, 'numpy', project_name, 'X.npy')
    results_dpath           = join(root, 'results', project_name)
    weights_fpath           = join(root, 'nets', project_name, 'ibeis_cnn_weights.pickle')
    config = {}

    test(test_data_fpath, model, weights_fpath, results_dpath, **config)


def test_pz_large():
    r"""
    CommandLine:
        python -m ibeis_cnn.test --test-test_pz_large

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.test import *  # NOQA
        >>> test_pz_large()
    """
    project_name            = 'plains_large'
    model                   = models.PZ_GIRM_LARGE_Model()

    root                    = abspath(join('..', 'data'))
    test_data_fpath        = join(root, 'numpy', project_name, 'X.npy')
    results_dpath           = join(root, 'results', project_name)
    weights_fpath           = join(root, 'nets', project_name, 'ibeis_cnn_weights.pickle')
    config = {}

    test(test_data_fpath, model, weights_fpath, results_dpath, **config)


def test_pz_girm():
    r"""
    CommandLine:
        python -m ibeis_cnn.test --test-test_pz_girm

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.test import *  # NOQA
        >>> test_pz_girm()
    """
    project_name            = 'viewpoint'
    model                   = models.PZ_GIRM_Model()

    root                    = abspath(join('..', 'data'))
    test_data_fpath        = join(root, 'numpy', project_name, 'X.npy')
    results_dpath           = join(root, 'results', project_name)
    weights_fpath           = join(root, 'nets', project_name, 'ibeis_cnn_weights.pickle')
    config = {}

    test(test_data_fpath, model, weights_fpath, results_dpath, **config)


def test_pz_girm_large():
    r"""
    CommandLine:
        python -m ibeis_cnn.test --test-test_pz_girm_large

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.test import *  # NOQA
        >>> test_pz_girm_large()
    """
    project_name            = 'viewpoint_large'
    model                   = models.PZ_GIRM_LARGE_Model()

    root                    = abspath(join('..', 'data'))
    test_data_fpath        = join(root, 'numpy', project_name, 'X.npy')
    results_dpath           = join(root, 'results', project_name)
    weights_fpath           = join(root, 'nets', project_name, 'ibeis_cnn_weights.pickle')
    config = {}

    test(test_data_fpath, model, weights_fpath, results_dpath, **config)


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis_cnn.test
        python -m ibeis_cnn.test --allexamples
        python -m ibeis_cnn.test --allexamples --noface --nosrc

    CommandLine:
        cd %CODE_DIR%/ibies_cnn/code
        cd $CODE_DIR/ibies_cnn/code
        code
        cd ibeis_cnn/code
        python test.py

    PythonPrereqs:
        pip install theano
        pip install git+https://github.com/Lasagne/Lasagne.git
        pip install git+git://github.com/lisa-lab/pylearn2.git
        #pip install lasagne
        #pip install pylearn2
        git clone git://github.com/lisa-lab/pylearn2.git
        git clone https://github.com/Lasagne/Lasagne.git
        cd pylearn2
        python setup.py develop
        cd ..
        cd Lesagne
        git checkout 8758ac1434175159e5c1f30123041799c2b6098a
        python setup.py develop
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
