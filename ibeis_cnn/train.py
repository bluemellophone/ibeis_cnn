#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines the models and the data we will send to the harness

CommandLineHelp:
    ./cnn.py

    --dataset, --ds = <dstag>:<subtag>
        dstag is the main dataset name (eg PZ_MTEST), subtag are parameters to modify (max_examples=3)

    --weights, -w = |new|<checkpoint_tag>|<dstag>:<checkpoint_tag> (default: <checkpoint_tag>)
        new will initialize clean weights.
        a checkpoint tag will try to to match a saved model state in the history.
        can load weights from an external dataset.
        <checkpoint_tag> defaults to current

    --arch, -a = <archtag>
        model architecture tag (eg siaml2, siam2stream, viewpoint)

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


# This is more of a history tag
checkpoint_tag_alias = {
    'current': None,
    '': None,
}

# second level of alias indirection
# This is more of a dataset tag
ds_tag_alias2 = {
    'nnp'       : 'NNP_Master3;dict(controlled=True, max_examples=None, num_top=3,)',
    'nnp3-2'    : 'NNP_Master3;dict(controlled=True, max_examples=None, num_top=None,)',
    'pzmaster' : 'PZ_Master0;dict(controlled=True, max_examples=None, num_top=3,)',
    'pzmtest'   : 'PZ_MTEST;dict(controlled=True, max_examples=None, num_top=3,)',
    'liberty'   : 'liberty;dict(detector=\'dog\', pairs=250000,)',
}


def train_patchmatch_pz():
    r"""
    RENAME:
        patchmatch?

    CommandLine:
        # Build Aliased Datasets

        # Train NNP_Master
        #python -m ibeis_cnn.train --test-train_patchmatch_pz --ds nnp3-2 --weights=nnp3-2:epochs0011 --arch=siaml2 --diagnose --test
        python -m ibeis_cnn.train --test-train_patchmatch_pz --ds nnp3-2 --weights=nnp3-2:epochs0011 --arch=siaml2 --train

        # Test NNP_Master on in sample
        python -m ibeis_cnn.train --test-train_patchmatch_pz --ds nnp3-2 --weights=current --arch=siaml2 --test
        python -m ibeis_cnn.train --test-train_patchmatch_pz --ds nnp3-2 --weights=nnp3-2 --arch=siaml2 --test

        # Test NNP_Master3 weights on out of sample data
        python -m ibeis_cnn.train --test-train_patchmatch_pz --ds pzmtest --weights=nnp3-2:epochs0021 --arch=siaml2 --test
        python -m ibeis_cnn.train --test-train_patchmatch_pz --ds pzmtest --weights=nnp3-2:epochs0011 --arch=siaml2 --test

        # Build PZ_Mater0
        python -m ibeis_cnn.train --test-train_patchmatch_pz --db PZ_Master0 --weights=nnp3-2:epochs0021 --arch=siaml2 --test --num_top=None
        # Now can use the alias
        python -m ibeis_cnn.train --test-train_patchmatch_pz --ds pzmtest --weights=nnp3-2:epochs0021 --arch=siaml2 --test
        python -m ibeis_cnn.train --test-train_patchmatch_pz --ds pzmaster --weights=nnp3-2:epochs0021 --arch=siaml2 --test


        # Hyperparameter settings
        python -m ibeis_cnn.train --test-train_patchmatch_pz --ds pzmtest --weights=new --arch=siaml2 --train --monitor

        --test --checkpoint master21


        # THIS DID WELL VERY QUICKLY
        python -m ibeis_cnn.train --test-train_patchmatch_pz --ds pzmtest --weights=new --arch=siaml2 --train --monitor --learning_rate=.1 --weight_decay=0.0005

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
    print('[train] train_siam')

    # Parse commandline args
    ds_tag      = ut.get_argval(('--dataset', '--ds'), type_=str, default=None)
    arch_tag    = ut.get_argval(('--arch', '-a'), default='siam2stream')
    weights_tag = ut.get_argval(('--weights', '+w'), type_=str, default=None)

    # breakup weights tag into extern_ds and checkpoint
    if weights_tag is not None and ':' in weights_tag:
        extern_ds_tag, checkpoint_tag = weights_tag.split(':')
    else:
        extern_ds_tag = None
        checkpoint_tag = weights_tag

    hyperparams = ut.argparse_dict(
        {
            'batch_size': 128,
            #'learning_rate': .0005,
            'learning_rate': .1,
            'momentum': .9,
            'weight_decay': 0.0005,
        }
    )

    # resolve aliases
    ds_tag = ds_tag_alias2.get(ds_tag, ds_tag)
    extern_ds_tag = ds_tag_alias2.get(extern_ds_tag, extern_ds_tag)
    checkpoint_tag = checkpoint_tag_alias.get(checkpoint_tag, checkpoint_tag)

    # ----------------------------
    # Choose the main dataset
    trainset = ingest_data.grab_siam_trainset(ds_tag)
    if extern_ds_tag is not None:
        extern_dpath = ingest_data.get_extern_training_dpath(extern_ds_tag)
    else:
        extern_dpath = None

    # ----------------------------
    # Choose model architecture
    # TODO: data will need to return info about number of labels in viewpoint models
    # Specify model archichitecture
    if arch_tag == 'siam2stream':
        model = models.SiameseCenterSurroundModel(
            data_shape=trainset.data_shape,
            training_dpath=trainset.training_dpath, **hyperparams)
    elif arch_tag == 'siaml2':
        model = models.SiameseL2(
            data_shape=trainset.data_shape,
            training_dpath=trainset.training_dpath, **hyperparams)
    else:
        raise ValueError('Unknown arch_tag=%r' % (arch_tag,))
    model.initialize_architecture()

    # ----------------------------
    # Choose weight initialization
    if checkpoint_tag == 'new':
        model.reinit_weights()
    else:
        checkpoint_tag = model.resolve_fuzzy_checkpoint_pattern(checkpoint_tag, extern_dpath)
        if extern_dpath is not None:
            model.load_extern_weights(dpath=extern_dpath, checkpoint_tag=checkpoint_tag)
        elif model.has_saved_state(checkpoint_tag=checkpoint_tag):
            model.load_model_state(checkpoint_tag=checkpoint_tag)
        else:
            raise ValueError('Unresolved weight init: checkpoint_tag=%r, extern_ds_tag=%r' % (checkpoint_tag, extern_ds_tag,))
    #print('Model State:')
    #print(model.get_state_str())

    # ----------------------------
    # Run Actions
    if ut.get_argflag('--train'):
        config = dict(
            learning_rate_schedule=10,
            max_epochs=100,
        )
        X_train, y_train = trainset.load_subset('train')
        X_valid, y_valid = trainset.load_subset('valid')
        #X_test, y_test = utils.load_from_fpath_dicts(data_fpath_dict, label_fpath_dict, 'test')
        harness.train(model, X_train, y_train, X_valid, y_valid, trainset, config)
    elif ut.get_argflag('--test'):
        #assert model.best_results['epoch'] is not None
        X_test, y_test = trainset.load_subset('all')
        #X_test, y_test = trainset.load_subset('test')
        data, labels = X_test, y_test
        #data, labels = utils.random_xy_sample(X_test, y_test, 1000, model.data_per_label_input)
        dataname = trainset.alias_key
        experiments.test_siamese_performance(model, data, labels, dataname)
    else:
        raise ValueError('nothing here. need to train or test')


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
    hyperparams = ut.argparse_dict(
        {
            'batch_size': 128,
            'learning_rate': .001,
            'momentum': .9,
            'weight_decay': 0.0005,
        }
    )
    trainset = ingest_data.grab_mnist_category_trainset()
    data_shape = trainset.data_shape
    input_shape = (None, data_shape[2], data_shape[0], data_shape[1])

    # Choose model
    model = models.MNISTModel(
        input_shape=input_shape, output_dims=trainset.output_dims,
        training_dpath=trainset.training_dpath, **hyperparams)

    # Initialize architecture
    model.initialize_architecture()

    # Load previously learned weights or initialize new weights
    if model.has_saved_state():
        model.load_model_state()
    else:
        model.reinit_weights()

    config = dict(
        learning_rate_schedule=10,
        max_epochs=100,
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
