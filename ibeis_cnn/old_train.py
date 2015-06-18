# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from ibeis_cnn import utils
from ibeis_cnn import models
from ibeis_cnn import ingest_ibeis
from ibeis_cnn import ingest_data
from ibeis_cnn import harness
import utool as ut
from os.path import join, abspath

def train_viewpoint_pz():
    r"""
    CommandLine:
        python -m ibeis_cnn.train --test-train_viewpoint_pz

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.train import *  # NOQA
        >>> train_viewpoint_pz()
    """
    project_name             = 'viewpoint_pz'
    model                    = models.ViewpointModel()
    root                     = abspath(join('..', 'data'))

    train_data_fpath         = join(root, 'numpy',   project_name, 'X.npy')
    train_labels_fpath       = join(root, 'numpy',   project_name, 'y.npy')
    results_dpath            = join(root, 'results', project_name)
    weights_fpath            = join(root, 'nets',    project_name, 'ibeis_cnn_weights.pickle')
    pretrained_weights_fpath = join(root, 'nets',    project_name, 'ibeis_cnn_weights.pickle')  # NOQA
    config                   = {
        'patience': 10,
        'regularization': 0.0001,
        'pretrained_weights_fpath': pretrained_weights_fpath,
    }
    harness.train(model, train_data_fpath, train_labels_fpath, weights_fpath, results_dpath, **config)


def train_quality_pz():
    r"""
    CommandLine:
        python -m ibeis_cnn.train --test-train_quality_pz

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.train import *  # NOQA
        >>> train_quality_pz()
    """
    project_name             = 'quality_pz'
    model                    = models.QualityModel()
    root                     = abspath(join('..', 'data'))

    train_data_fpath         = join(root, 'numpy',   project_name, 'X.npy')
    train_labels_fpath       = join(root, 'numpy',   project_name, 'y.npy')
    results_dpath            = join(root, 'results', project_name)
    weights_fpath            = join(root, 'nets',    project_name, 'ibeis_cnn_weights.pickle')
    pretrained_weights_fpath = join(root, 'nets',    project_name, 'ibeis_cnn_weights.pickle')  # NOQA
    config                   = {
        'patience': 10,
        'regularization': 0.0001,
        'pretrained_weights_fpath': pretrained_weights_fpath,
    }
    harness.train(model, train_data_fpath, train_labels_fpath, weights_fpath, results_dpath, **config)


def train_viewpoint():
    r"""
    CommandLine:
        python -m ibeis_cnn.train --test-train_viewpoint

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.train import *  # NOQA
        >>> train_viewpoint()
    """
    project_name             = 'viewpoint'
    model                    = models.ViewpointModel()
    root                     = abspath(join('..', 'data'))

    train_data_fpath         = join(root, 'numpy',   project_name, 'X.npy')
    train_labels_fpath       = join(root, 'numpy',   project_name, 'y.npy')
    results_dpath            = join(root, 'results', project_name)
    weights_fpath            = join(root, 'nets',    project_name, 'ibeis_cnn_weights.pickle')
    pretrained_weights_fpath = join(root, 'nets',    project_name, 'ibeis_cnn_weights.pickle')  # NOQA
    config                   = {
        'patience': 10,
        'regularization': 0.0001,
        'pretrained_weights_fpath': pretrained_weights_fpath,
    }
    harness.train(model, train_data_fpath, train_labels_fpath, weights_fpath, results_dpath, **config)


#def train_patchmatch_liberty():
#    r"""
#    CommandLine:
#        python -m ibeis_cnn.train --test-train_patchmatch_liberty --show
#        python -m ibeis_cnn.train --test-train_patchmatch_liberty --vtd
#        python -m ibeis_cnn.train --test-train_patchmatch_liberty --show --test
#        python -m ibeis_cnn.train --test-train_patchmatch_liberty --test
#        python -m ibeis_cnn.train --test-train_patchmatch_liberty --train

#    Example:
#        >>> # ENABLE_DOCTEST
#        >>> from ibeis_cnn.train import *  # NOQA
#        >>> result = train_patchmatch_liberty()
#        >>> ut.show_if_requested()
#        >>> print(result)
#    """
#    # TODO; integrate into data tagging system
#    train_params = ut.argparse_dict(
#        {
#            #'learning_rate': .01,
#            #'weight_decay': 0.0005,
#            'batch_size': 128,
#            'learning_rate': .007,
#            'momentum': .9,
#            'weight_decay': 0.0008,
#        }
#    )
#    #pairs = 500
#    pairs = 250000
#    trainset = ingest_data.grab_cached_liberty_data(pairs)
#    #model = models.SiameseModel()
#    model = models.SiameseCenterSurroundModel(data_shape=trainset.data_shape, training_dpath=trainset.training_dpath, **train_params)

#    model.initialize_architecture()

#    if ut.get_argflag('--test'):
#        # Use external state
#        extern_training_dpath = ingest_data.get_extern_training_dpath('NNP_Master3;dict(max_examples=None, num_top=3,)')
#        model.load_extern_weights(dpath=extern_training_dpath)
#    else:
#        if model.has_saved_state():
#            model.load_model_state()
#        else:
#            model.reinit_weights()
#    print(model.get_state_str())

#    if ut.get_argflag('--train'):
#        config = dict(
#            patience=100,
#        )
#        X_train, y_train = trainset.load_subset('train')
#        X_valid, y_valid = trainset.load_subset('valid')
#        #X_test, y_test = utils.load_from_fpath_dicts(data_fpath_dict, label_fpath_dict, 'test')
#        harness.train(model, X_train, y_train, X_valid, y_valid, trainset, config)
#    elif ut.get_argflag('--test'):

#        X_test, y_test = trainset.load_subset('test')
#        data, labels = X_test, y_test
#        data, labels = utils.random_xy_sample(X_test, y_test, 1000, model.data_per_label)
#        dataname = trainset.alias_key
#        experiments.test_siamese_performance(model, data, labels, dataname)

#    else:
#        raise NotImplementedError('nothing here. need to train or test')
