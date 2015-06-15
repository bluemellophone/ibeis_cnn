#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines the models and the data we will send to the harness

CommandLine:
    python -m ibeis_cnn.train --test-train_patchmatch_pz
    python -m ibeis_cnn.train --test-train_patchmatch_pz --vtd
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

    Model Weight Tags
    Model Architecture Tags
    Dataset Tags
    Checkpoint Tag?

    Data Metadata - Along with ability to go back and check the context of fail cases
    - need to use original SIFT descriptors from ibeis db if available




Ideas:
    Neural Network Vocabulary?
    Input a patch
    Output a word
    Training: unsupervised sparse autoencoder

"""
from __future__ import absolute_import, division, print_function
from ibeis_cnn import utils  # NOQA
from ibeis_cnn import models
from ibeis_cnn import ingest_data
from ibeis_cnn import harness
from ibeis_cnn import experiments
import utool as ut
print, rrr, profile = ut.inject2(__name__, '[ibeis_cnn.train]')


# second level of alias indirection
# This is more of a dataset tag
weights_tag_alias2 = {
    'nnp_master': 'NNP_Master3;dict(max_examples=None, num_top=3,)',
    'pzmtest': 'PZ_MTEST;dict(max_examples=None, num_top=3,)',
    'liberty': 'liberty;dict(detector=\'dog\', pairs=250000,)',
}


# This is more of a history tag
checkpoint_tag_alias = {
    '1': 'hist_eras1_epochs1_luhacgyiftsezrzi',
    '11': 'hist_eras1_epochs11_anivdezohtrouieo',
    '12': 'hist_eras1_epochs12_hmkamjjumeifwufs',
    '21': 'hist_eras1_epochs21_cnsszjkathjbluos',

    'lib30': 'hist_eras3_epochs30_zqwhqylxyihnknxc'
}


def get_patchmatch_transet():
    dbname = ut.get_argval('--db')
    if dbname == 'liberty':
        pairs = 250000
        trainset = ingest_data.grab_cached_liberty_data(pairs)
        pass
    else:
        trainset = ingest_data.get_patchmetric_training_fpaths()
    return trainset


def train_patchmatch_pz():
    r"""
    TrainingCommandLine:
        python -m ibeis_cnn.train --test-train_patchmatch_pz --train
        python -m ibeis_cnn.train --test-train_patchmatch_pz --vtd --max_examples=3 --learning_rate .0000001 --train
        python -m ibeis_cnn.train --test-train_patchmatch_pz --db NNP_Master3 --nocache-train

        python -m ibeis_cnn.train --test-train_patchmatch_pz --db NNP_Master3 --train
        python -m ibeis_cnn.train --test-train_patchmatch_pz --db PZ_Master0 --train

    TestingCommandLine:
        python -m ibeis_cnn.train --test-train_patchmatch_pz --test='this'
        python -m ibeis_cnn.train --test-train_patchmatch_pz --vtd

        python -m ibeis_cnn.train --test-train_patchmatch_pz --db NNP_Master3 --num-top=20
        python -m ibeis_cnn.train --test-train_patchmatch_pz --db NNP_Master3 --vtd

        python -m ibeis_cnn.train --test-train_patchmatch_pz --db NNP_Master3 --test --weights=this
        python -m ibeis_cnn.train --test-train_patchmatch_pz --db NNP_Master3 --test --weights=nnp_master
        python -m ibeis_cnn.train --test-train_patchmatch_pz --db PZ_MTEST --test --weights=nnp_master
        python -m ibeis_cnn.train --test-train_patchmatch_pz --db PZ_MTEST --test --weights=nnp_master --checkpoint=11
        python -m ibeis_cnn.train --test-train_patchmatch_pz --db NNP_Master3 --test --weights=nnp_master --checkpoint=12
        python -m ibeis_cnn.train --test-train_patchmatch_pz --db PZ_MTEST --test --weights=new
        python -m ibeis_cnn.train --test-train_patchmatch_pz --db NNP_Master3 --test --weights=new

        python -m ibeis_cnn.train --test-train_patchmatch_pz --db NNP_Master3 --test --weights=liberty
        python -m ibeis_cnn.train --test-train_patchmatch_pz --db PZ_MTEST --test --weights=liberty --checkpoint=lib30

        python -m ibeis_cnn.train --test-train_patchmatch_pz --db liberty --test
        python -m ibeis_cnn.train --test-train_patchmatch_pz --db liberty --test --checkpoint=hist_eras1_epochs14_mzdgzqtjprzddqie


    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.train import *  # NOQA
        >>> from ibeis_cnn.harness import *  # NOQA
        >>> from ibeis_cnn.batch_processing import *  # NOQA
        >>> train_patchmatch_pz()
        >>> ut.show_if_requested()

    Ignore:
        weights_fpath = '/media/raid/work/NNP_Master3/_ibsdb/_ibeis_cache/nets/train_patchmetric((2462)&ju%uw7bta19cunw)/ibeis_cnn_weights.pickle'
        weights_fpath = '/media/raid/work/NNP_Master3/_ibsdb/_ibeis_cache/nets/train_patchmetric((2462)&ju%uw7bta19cunw)/arch_d788de3571330d42/training_state.cPkl'
    """
    print('[train] train_patchmatch_pz')
    # Choose data
    trainset = get_patchmatch_transet()
    #trainset = ingest_data.get_patchmetric_training_fpaths()

    # Choose model
    # TODO: data will need to return info about number of labels in viewpoint models
    train_params = ut.argparse_dict(
        {
            'batch_size': 128,
            'learning_rate': .007,
            'momentum': .9,
            'weight_decay': 0.0008,
        }
    )
    model = models.SiameseCenterSurroundModel(data_shape=trainset.data_shape,
                                              training_dpath=trainset.training_dpath,
                                              **train_params)

    model.initialize_architecture()

    weights_tag = 'this'
    weights_tag = ut.get_argval('--weights', type_=str, default=weights_tag)
    weights_tag = weights_tag_alias2.get(weights_tag, weights_tag)

    checkpoint_tag = ut.get_argval('--checkpoint', type_=str, default=None)
    checkpoint_tag = checkpoint_tag_alias.get(checkpoint_tag, checkpoint_tag)

    if weights_tag == 'this':
        if model.has_saved_state(checkpoint_tag=checkpoint_tag):
            model.load_model_state(checkpoint_tag=checkpoint_tag)
        else:
            model.reinit_weights()
    elif weights_tag == 'new':
        model.reinit_weights()
    else:
        extern_training_dpath = ingest_data.get_extern_training_dpath(weights_tag)
        model.load_extern_weights(dpath=extern_training_dpath, checkpoint_tag=checkpoint_tag)

    print('MODEL STATE:')
    print(model.get_state_str())

    if ut.get_argflag('--train'):
        config = dict(
            patience=100,
        )
        X_train, y_train = trainset.load_subset('train')
        X_valid, y_valid = trainset.load_subset('valid')
        #X_test, y_test = utils.load_from_fpath_dicts(data_fpath_dict, label_fpath_dict, 'test')
        harness.train(model, X_train, y_train, X_valid, y_valid, trainset, config)
    elif ut.get_argflag('--test'):
        #assert model.best_results['epoch'] is not None
        X_test, y_test = trainset.load_subset('test')
        data, labels = X_test, y_test
        #data, labels = utils.random_test_train_sample(X_test, y_test, 1000, model.data_per_label)
        dataname = trainset.alias_key
        experiments.test_siamese_performance(model, data, labels, dataname)
    else:
        raise NotImplementedError('nothing here. need to train or test')


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
#        data, labels = utils.random_test_train_sample(X_test, y_test, 1000, model.data_per_label)
#        dataname = trainset.alias_key
#        experiments.test_siamese_performance(model, data, labels, dataname)

#    else:
#        raise NotImplementedError('nothing here. need to train or test')


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
    train_params = ut.argparse_dict(
        {
            'batch_size': 128,
            'learning_rate': .001,
            'momentum': .9,
            'weight_decay': 0.0005,
        }
    )
    trainset = ingest_data.grab_cached_mist_data()
    data_shape = trainset.data_shape
    input_shape = (None, data_shape[2], data_shape[0], data_shape[1])

    # Choose model
    model = models.MNISTModel(
        input_shape=input_shape, output_dims=trainset.output_dims,
        training_dpath=trainset.training_dpath, **train_params)

    # Initialize architecture
    model.initialize_architecture()

    # Load previously learned weights or initialize new weights
    if model.has_saved_state():
        model.load_model_state()
    else:
        model.reinit_weights()

    config = dict(
        patience=100,
        show_confusion=False,
        run_test=None,
        show_features=False,
        print_timing=False,
    )

    X_train, y_train = trainset.load_subset('train')
    X_valid, y_valid = trainset.load_subset('valid')
    #X_test, y_test = utils.load_from_fpath_dicts(data_fpath_dict, label_fpath_dict, 'test')
    harness.train(model, X_train, y_train, X_valid, y_valid, trainset, config)


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
