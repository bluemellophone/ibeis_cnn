#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests a test set of data using a specified, pre0trained model and weights
"""
from __future__ import absolute_import, division, print_function
from ibeis_cnn import utils
from ibeis_cnn import models
from ibeis_cnn import net_strs
from ibeis_cnn import batch_processing as batch

from six.moves import cPickle as pickle
from lasagne import layers

import theano
import time
import numpy as np
import utool as ut
import cv2
import six  # NOQA
from os.path import join, abspath
print, rrr, profile = ut.inject2(__name__, '[ibeis_cnn.test]')


def test(data_fpath, model, weights_fpath, results_dpath=None, labels_fpath=None, **kwargs):
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
    X_test, y_test = utils.load(data_fpath, labels_fpath)
    if len(X_test.shape) == 3:
        # add channel dimension for implicit grayscale
        X_test.shape = X_test.shape + (1,)

    return test_data(X_test, y_test, model, weights_fpath, results_dpath, **kwargs)


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
