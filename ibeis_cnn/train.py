#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines the models and the data we will send to the harness

CommandLine:
    python -m ibeis_cnn.train --test-train_patchmatch_pz
    python -m ibeis_cnn.train --test-train_patchmatch_pz --vtd
    python -m ibeis_cnn.train --test-train_patchmatch_pz --vtd --max-examples=5 --batch_size=128 --learning_rate .0000001
    python -m ibeis_cnn.train --test-train_patchmatch_pz --vtd --max-examples=5 --batch_size=128 --learning_rate .0000001 --verbcnn
    python -m ibeis_cnn.train --test-train_patchmatch_pz --db NNP_Master3
    python -m ibeis_cnn.train --test-train_patchmatch_pz --db NNP_Master3 --num-top=5
    python -m ibeis_cnn.train --test-train_patchmatch_pz --db NNP_Master3 --vtd
    python -m ibeis_cnn.train --test-train_patchmatch_pz --db NNP_Master3 --max-examples=1000
    python -m ibeis_cnn.train --test-train_patchmatch_pz --db NNP_Master3 --max-examples=1000 --num-top=5

TestTraining:
    python -m ibeis_cnn.train --test-train_patchmatch_pz --vtd --max-examples=5 --batch_size=128 --learning_rate .0000001
    python -m ibeis_cnn.train --test-train_patchmatch_mnist --vtd
    utprof.py -m ibeis_cnn.train --test-train_patchmatch_pz --vtd --max-examples=5 --batch_size=128 --learning_rate .0000001
    utprof.py -m ibeis_cnn.train --test-train_patchmatch_mnist --vtd

NightlyTraining:
    python -m ibeis_cnn.train --test-train_patchmatch_pz --db NNP_Master3 --max-examples=1000


TODO:
    A model architecture should have a data-agnostic directory
    Allow multiple different datasets to train the same model
      * The adjustments that that model makes should be saved in a data-specific directory
    Show bad training examples
"""
from __future__ import absolute_import, division, print_function
from ibeis_cnn import utils
from ibeis_cnn import models
from ibeis_cnn import ingest_data
from ibeis_cnn import harness
import utool as ut
from os.path import join
print, rrr, profile = ut.inject2(__name__, '[ibeis_cnn.train]')


def train_patchmatch_pz():
    r"""

    CommandLine:
        python -m ibeis_cnn.train --test-train_patchmatch_pz --show
        python -m ibeis_cnn.train --test-train_patchmatch_pz --vtd
        python -m ibeis_cnn.train --test-train_patchmatch_pz --vtd --max-examples=5 --learning_rate .0000001

        python -m ibeis_cnn.train --test-train_patchmatch_pz --db NNP_Master3 --nocache-train
        python -m ibeis_cnn.train --test-train_patchmatch_pz --db NNP_Master3 --num-top=20
        python -m ibeis_cnn.train --test-train_patchmatch_pz --db NNP_Master3 --vtd

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.train import *  # NOQA
        >>> train_patchmatch_pz()

    Ignore:
        weights_fpath = '/media/raid/work/NNP_Master3/_ibsdb/_ibeis_cache/nets/train_patchmetric((2462)&ju%uw7bta19cunw)/ibeis_cnn_weights.pickle'

        weights_fpath = '/media/raid/work/NNP_Master3/_ibsdb/_ibeis_cache/nets/train_patchmetric((2462)&ju%uw7bta19cunw)/arch_d788de3571330d42/training_state.cPkl'

    """
    datakw = ut.parse_dict_from_argv(
        {
            #'db': 'PZ_MTEST',
            'max_examples': None,
            'num-top': 3,
        }
    )
    print('[train] train_patchmatch_pz')

    with ut.Indenter('[LOAD IBEIS DB]'):
        import ibeis
        ibs = ibeis.opendb(defaultdb='PZ_MTEST')

    # Nets dir is the root dir for all training
    training_dpath = ibs.get_neuralnet_dir()
    ut.ensuredir(training_dpath)
    log_dir = join(training_dpath, 'logs')
    ut.start_logging(log_dir=log_dir)
    #ut.embed()

    with ut.Indenter('[CHECKDATA]'):
        data_fpath, labels_fpath, training_dpath, data_shape = ingest_data.get_patchmetric_training_fpaths(ibs, **datakw)

    #model = models.SiameseModel()
    batch_size = ut.get_argval('--batch_size', type_=int, default=128)
    model = models.SiameseCenterSurroundModel(data_shape=data_shape, batch_size=batch_size)
    model.initialize_architecture()

    if not model.has_saved_state():
        # initialize with liberty
        extern_training_dpath = ut.ensure_app_resource_dir('ibeis_cnn', 'training', 'liberty')
        model.load_extern_weights(dpath=extern_training_dpath)
        #model.draw_convolutional_layers()
        #ut.show_if_requested()

    learning_state = dict(
        momentum=.9,
        regularization=0.0005,
        learning_rate=ut.get_argval('--learning_rate', type_=float, default=.001),
    )

    config = dict(
        patience=100,
        equal_batch_sizes=True,
        batch_size=batch_size,
        show_confusion=False,
        requested_headers=['epoch', 'train_loss', 'valid_loss', 'trainval_rat', 'duration'],
        run_test=None,
        show_features=False,
        print_timing=False,
    )
    weights_fpath = join(training_dpath, 'ibeis_cnn_weights.pickle')

    #ut.embed()
    if ut.get_argflag('--test'):
        harness.test(data_fpath, model, weights_fpath, labels_fpath, **config)
    else:
        harness.train(model, data_fpath, labels_fpath, weights_fpath, training_dpath, **config)


def train_patchmatch_liberty():
    r"""
    CommandLine:
        python -m ibeis_cnn.train --test-train_patchmatch_liberty --show
        python -m ibeis_cnn.train --test-train_patchmatch_liberty --vtd
        python -m ibeis_cnn.train --test-train_patchmatch_liberty --show --test

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_cnn.train import *  # NOQA
        >>> result = train_patchmatch_liberty()
        >>> ut.show_if_requested()
        >>> print(result)
    """
    # TODO: move to standard path
    training_dpath = ut.ensure_app_resource_dir('ibeis_cnn', 'training', 'liberty')
    if ut.get_argflag('--vtd'):
        ut.vd(training_dpath)
    ut.ensuredir(training_dpath)

    pairs = 500
    data_fpath, labels_fpath, data_shape = ingest_data.grab_cached_liberty_data(training_dpath, pairs)
    #model = models.SiameseModel()
    model = models.SiameseCenterSurroundModel(data_shape=data_shape, training_dpath=training_dpath)
    model.initialize_architecture()
    # DO CONVERSION
    if False:
        old_weights_fpath = ut.truepath('~/Dropbox/ibeis_cnn_weights_liberty.pickle')
        if ut.checkpath(old_weights_fpath, verbose=True):
            self = model
            self.load_old_weights_kw(old_weights_fpath)
        self.save_model_state()
        #self.save_state()
    weights_fpath = join(training_dpath, 'ibeis_cnn_weights.pickle')

    config = dict(
        #patience=100,
        #show_confusion=False,
        #run_test=None,
        #show_features=False,
        #print_timing=False,
        #requested_headers=['epoch', 'train_loss', 'valid_loss', 'trainval_rat', 'duration'],
        #equal_batch_sizes=True,
        learning_rate=ut.get_argval('--learning_rate', type_=float, default=.001),
        momentum=.9,
        regularization=0.0005,
    )

    #ut.embed()
    if ut.get_argflag('--test'):
        from ibeis_cnn import test
        data, labels = utils.load(data_fpath, labels_fpath)
        train_split = .2
        # Grab the validataion set
        _, _, X_test_, y_test_ = utils.train_test_split(
            data, labels, eval_size=train_split,
            data_per_label=model.data_per_label)
        # Grab a sample of that
        _, _, X_test, y_test = utils.train_test_split(
            X_test_, y_test_, eval_size=.1,
            data_per_label=model.data_per_label)
        ### Compare to SIFT descriptors
        import pyhesaff
        X_sift = pyhesaff.extract_desc_from_patches(X_test)
        import numpy as np
        sqrddist = ((X_sift[::2].astype(np.float32) - X_sift[1::2].astype(np.float32)) ** 2).sum(axis=1)
        test.test_siamese_thresholds(sqrddist[None, :].T, y_test)
        # add channel dimension for implicit grayscale
        if len(X_test.shape) == 3:
            X_test.shape = X_test.shape + (1,)
        #
        (pred_list, label_list, conf_list, prob_list) = test.test_data2(
            X_test, y_test, model, weights_fpath, **config)
        harness.test_siamese_thresholds(prob_list, y_test)
    else:
        harness.train(model, data_fpath, labels_fpath, weights_fpath, training_dpath, **config)
    pass


def train_mnist():
    r"""
    CommandLine:
        python -m ibeis_cnn.train --test-train_mnist

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.train import *  # NOQA
        >>> result = train_mnist()
        >>> print(result)
    """
    training_dpath = ut.get_app_resource_dir('ibeis_cnn', 'training', 'mnist')
    #training_dpath = ut.truepath('mnist')
    ut.ensuredir(training_dpath)
    data_fpath, labels_fpath = ingest_data.grab_cached_mist_data(training_dpath)

    model                    = models.MNISTModel()
    #model                    = models.ViewpointModel()
    train_data_fpath         = data_fpath
    train_labels_fpath       = labels_fpath
    results_dpath            = training_dpath
    weights_fpath            = join(training_dpath, 'ibeis_cnn_weights.pickle')
    pretrained_weights_fpath = join(training_dpath, 'ibeis_cnn_weights.pickle')  # NOQA
    config                   = {
        'patience': 10,
        'regularization': 0.0001,
        'learning_rate': 0.0001,
        'pretrained_weights_fpath': pretrained_weights_fpath,
    }
    harness.train(model, train_data_fpath, train_labels_fpath, weights_fpath, results_dpath, **config)
    #images, labels = open_mnist(train_imgs_fpath, train_lbls_fpath)


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis_cnn.train
        python -m ibeis_cnn.train --allexamples
        python -m ibeis_cnn.train --allexamples --noface --nosrc
    """
    #train_pz()
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    #import warnings
    #with warnings.catch_warnings():
    #    # Cause all warnings to always be triggered.
    #    warnings.filterwarnings("error", ".*get_all_non_bias_params.*")
    ut.doctest_funcs()
