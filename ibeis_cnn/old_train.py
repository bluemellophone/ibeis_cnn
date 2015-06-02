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
