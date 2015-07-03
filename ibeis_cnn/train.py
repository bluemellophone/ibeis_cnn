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


    * Training Babysitting:
    * Graph of weight magnitude updates - per layer as well
    * validation loss variance - per example -
    * determenistic loss ratios
    * loss ratios without weight decay

    * Baysian Hyperparamater optimization

    * Combine datasets

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
    'nnp3-2-bgr'   : "NNP_Master3;dict(colorspace='bgr', controlled=True, max_examples=None, num_top=None,)",
    'pzmaster-bgr' : "PZ_Master0;dict(colorspace='bgr', controlled=True, max_examples=None, num_top=None,)",
    'pzmtest-bgr'  : "PZ_MTEST;dict(colorspace='bgr', controlled=True, max_examples=None, num_top=None,)",

    'nnp'         : "NNP_Master3;dict(colorspace='gray', controlled=True, max_examples=None, num_top=None,)",
    'pzmtest'     : "PZ_MTEST;dict(colorspace='gray', controlled=True, max_examples=None, num_top=None,)",
    'gz-gray'     : "GZ_ALL;dict(colorspace='gray', controlled=False, max_examples=None, num_top=None,)",
    'girm'        : "NNP_MasterGIRM_core;dict(colorspace='gray', controlled=True, max_examples=None, num_top=None,)",

    'pzmaster'    : 'PZ_Master0;dict(controlled=True, max_examples=None, num_top=None,)',
    'liberty'     : "liberty;dict(detector='dog', pairs=250000,)",

    'combo': 'combo_vdsujffw',
}


def merge_ds_tags(ds_alias_list):
    r"""
    CommandLine:
        python -m ibeis_cnn.train --test-merge_ds_tags --alias-list gz-gray girm pzmtest nnp

    TODO:
        http://stackoverflow.com/questions/18492273/combining-hdf5-files

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_cnn.train import *  # NOQA
        >>> ds_alias_list = ut.get_argval('--alias-list', type_=list, default=[])
        >>> result = merge_ds_tags(ds_alias_list)
        >>> print(result)
    """
    ds_tag_list = [ds_tag_alias2.get(ds_tag, ds_tag) for ds_tag in ds_alias_list]
    dataset_list = [ingest_data.grab_siam_dataset(ds_tag) for ds_tag in ds_tag_list]
    merged_dataset = ingest_data.merge_datasets(dataset_list)
    print(merged_dataset.alias_key)
    return merged_dataset


def train_patchmatch_pz():
    r"""
    RENAME:
        patchmatch?

    CommandLine:

        # --- DATASET BUILDING ---

        # Build Dataset Aliases
        python -m ibeis_cnn.train --test-train_patchmatch_pz --db PZ_Master0 --colorspace='gray' --num-top=None --controlled=True --aliasexit
        python -m ibeis_cnn.train --test-train_patchmatch_pz --db liberty --colorspace='gray' --aliasexit
        python -m ibeis_cnn.train --test-train_patchmatch_pz --db PZ_MTEST --colorspace='gray' --num-top=None --controlled=True --aliasexit
        python -m ibeis_cnn.train --test-train_patchmatch_pz --db NNP_Master3 --colorspace='gray' --num-top=None --controlled=True --aliasexit
        python -m ibeis_cnn.train --test-train_patchmatch_pz --db GZ_ALL --colorspace='gray' --num-top=None --controlled=True --aliasexit
        python -m ibeis_cnn.train --test-train_patchmatch_pz --db NNP_MasterGIRM_core --colorspace='gray' --num-top=None --controlled=True --aliasexit

        # --- TRAINING ---

        # Train NNP_Master
        python -m ibeis_cnn.train --test-train_patchmatch_pz --ds nnp3-2 --weights=nnp3-2:epochs0011 --arch=siaml2 --train --monitor
        python -m ibeis_cnn.train --test-train_patchmatch_pz --ds nnp3-2 --weights=new --arch=siaml2 --train --weight_decay=0.0001 --monitor
        python -m ibeis_cnn.train --test-train_patchmatch_pz --ds nnp3-2 --weights=new --arch=siaml2 --train --weight_decay=0.0001 --monitor

        python -m ibeis_cnn.train --test-train_patchmatch_pz --ds pzmtest --weights=new --arch=siaml2_128 --train --monitor
        python -m ibeis_cnn.train --test-train_patchmatch_pz --db liberty --weights=new --arch=siaml2_128 --train --monitor

        # Train COMBO
        python -m ibeis_cnn.train --test-train_patchmatch_pz --ds combo --weights=new --arch=siaml2_128 --train --monitor
        python -m ibeis_cnn.train --test-train_patchmatch_pz --ds combo --weights=hist_eras007_epochs0098_gvmylbm --arch=siaml2_128 --train --monitor --max-epochs=1000 --learning_rate=.02 --learning_rate_adjust=.9 --learning_rate_schedule=20

        # --- MONITOR TRAINING ---

        # Hyperparameter settings
        python -m ibeis_cnn.train --test-train_patchmatch_pz --ds pzmtest --weights=new --arch=siaml2 --train --monitor

        # Grevys
        python -m ibeis_cnn.train --test-train_patchmatch_pz --ds gz-gray --weights=new --arch=siaml2 --train --weight_decay=0.0001 --monitor

        # THIS DID WELL VERY QUICKLY
        python -m ibeis_cnn.train --test-train_patchmatch_pz --ds pzmtest --weights=new --arch=siaml2 --train --monitor --learning_rate=.1 --weight_decay=0.0005
        python -m ibeis_cnn.train --test-train_patchmatch_pz --ds pzmtest --weights=new --arch=siaml2 --train --monitor --DEBUG_AUGMENTATION

        python -m ibeis_cnn.train --test-train_patchmatch_pz --ds pzmtest --weights=new --arch=siaml2 --train --monitor

        python -m ibeis_cnn.train --test-train_patchmatch_pz --ds nnp3-2 --weights=new --arch=siaml2 --train --monitor
        python -m ibeis_cnn.train --test-train_patchmatch_pz --ds nnp3-2 --weights=new --arch=siam2streaml2 --train --monitor

        python -m ibeis_cnn.train --test-train_patchmatch_pz --db NNP_Master3 --weights=new --arch=siaml2 --train --monitor --colorspace='bgr' --num_top=None

        python -m ibeis_cnn.train --test-train_patchmatch_pz --ds pzmtest-bgr --weights=nnp3-2-bgr:epochs0023_rttjuahuhhraphyb --arch=siaml2 --test
        python -m ibeis_cnn.train --test-train_patchmatch_pz --ds pzmaster-bgr --weights=nnp3-2-bgr:epochs0023_rttjuahuhhraphyb --arch=siaml2 --test
        --monitor --colorspace='bgr' --num_top=None

        # --- INITIALIZED-TRAINING ---
        python -m ibeis_cnn.train --test-train_patchmatch_pz --ds pzmtest --arch=siaml2 --weights=gz-gray:current --train --monitor

        # --- TESTING ---
        python -m ibeis_cnn.train --test-train_patchmatch_pz --db liberty --weights=liberty:current --arch=siaml2_128 --test

        # test combo
        python -m ibeis_cnn.train --test-train_patchmatch_pz --db PZ_Master0 --weights=combo:hist_eras007_epochs0098_gvmylbm --arch=siaml2_128 --testall
        python -m ibeis_cnn.train --test-train_patchmatch_pz --db liberty --weights=liberty:current --arch=siaml2_128 --test

        python -m ibeis_cnn.train --test-train_patchmatch_pz --ds gz-gray --arch=siaml2 --weights=gz-gray:current --test
        python -m ibeis_cnn.train --test-train_patchmatch_pz --ds nnp3-2 --arch=siaml2 --weights=gz-gray:current --test
        python -m ibeis_cnn.train --test-train_patchmatch_pz --ds pzmtest --arch=siaml2 --weights=gz-gray:current --test
        python -m ibeis_cnn.train --test-train_patchmatch_pz --ds pzmaster --arch=siaml2 --weights=gz-gray:current --test

        # Test NNP_Master on in sample
        python -m ibeis_cnn.train --test-train_patchmatch_pz --ds nnp3-2 --weights=current --arch=siaml2 --test
        python -m ibeis_cnn.train --test-train_patchmatch_pz --ds nnp3-2 --weights=nnp3-2 --arch=siaml2 --test

        # Test NNP_Master3 weights on out of sample data
        python -m ibeis_cnn.train --test-train_patchmatch_pz --ds pzmtest --weights=nnp3-2:epochs0021 --arch=siaml2 --test
        python -m ibeis_cnn.train --test-train_patchmatch_pz --ds pzmtest --weights=nnp3-2:epochs0011 --arch=siaml2 --test

        # Now can use the alias
        python -m ibeis_cnn.train --test-train_patchmatch_pz --ds pzmtest --weights=nnp3-2:epochs0021 --arch=siaml2 --test
        python -m ibeis_cnn.train --test-train_patchmatch_pz --ds pzmaster --weights=nnp3-2:epochs0021 --arch=siaml2 --test

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
            #'batch_size': 128,
            'batch_size': 256,
            #'learning_rate': .0005,
            'learning_rate': .1,
            'momentum': .9,
            #'weight_decay': 0.0005,
            'weight_decay': 0.0001,
        }
    )

    # resolve aliases
    ds_tag = ds_tag_alias2.get(ds_tag, ds_tag)
    extern_ds_tag = ds_tag_alias2.get(extern_ds_tag, extern_ds_tag)
    checkpoint_tag = checkpoint_tag_alias.get(checkpoint_tag, checkpoint_tag)

    # ----------------------------
    # Choose the main dataset
    dataset = ingest_data.grab_siam_dataset(ds_tag)
    if extern_ds_tag is not None:
        extern_dpath = ingest_data.get_extern_training_dpath(extern_ds_tag)
    else:
        extern_dpath = None

    if ut.get_argflag('--aliasexit'):
        print(repr(dataset.alias_key))
        import sys
        sys.exit(1)

    # ----------------------------
    # Choose model architecture
    # TODO: data will need to return info about number of labels in viewpoint models
    # Specify model archichitecture
    if arch_tag == 'siam2stream':
        model = models.SiameseCenterSurroundModel(
            data_shape=dataset.data_shape,
            training_dpath=dataset.training_dpath, **hyperparams)
    elif arch_tag in ['siaml2', 'siaml2_128']:
        model = models.SiameseL2(
            data_shape=dataset.data_shape,
            arch_tag=arch_tag,
            training_dpath=dataset.training_dpath, **hyperparams)
    elif arch_tag == 'siam2streaml2':
        model = models.SiameseL2(
            data_shape=dataset.data_shape,
            arch_tag=arch_tag,
            training_dpath=dataset.training_dpath,
            **hyperparams)
    elif arch_tag == 'mnist-category':
        model = models.MNISTModel(
            data_shape=dataset.data_shape, output_dims=dataset.output_dims,
            training_dpath=dataset.training_dpath, **hyperparams)
        pass
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
        config = ut.argparse_dict(dict(
            learning_rate_schedule=15,
            max_epochs=120,
            learning_rate_adjust=.8,
        ))
        X_train, y_train = dataset.load_subset('train')
        X_valid, y_valid = dataset.load_subset('valid')
        #X_test, y_test = utils.load_from_fpath_dicts(data_fpath_dict, label_fpath_dict, 'test')
        harness.train(model, X_train, y_train, X_valid, y_valid, dataset, config)
    elif ut.get_argflag('--test') or ut.get_argflag('--testall'):
        #assert model.best_results['epoch'] is not None
        if ut.get_argflag('--testall'):
            X_test, y_test = dataset.load_subset('all')
        X_test, y_test = dataset.load_subset('test')
        data, labels = X_test, y_test
        #data, labels = utils.random_xy_sample(X_test, y_test, 1000, model.data_per_label_input)
        dataname = dataset.alias_key
        experiments.test_siamese_performance(model, data, labels, dataname)
    else:
        raise ValueError('nothing here. need to train or test')


#def train_mnist():
#    r"""
#    CommandLine:
#        python -m ibeis_cnn.train --test-train_mnist

#    Example:
#        >>> # DISABLE_DOCTEST
#        >>> from ibeis_cnn.train import *  # NOQA
#        >>> result = train_mnist()
#        >>> print(result)
#    """
#    hyperparams = ut.argparse_dict(
#        {
#            'batch_size': 128,
#            'learning_rate': .001,
#            'momentum': .9,
#            'weight_decay': 0.0005,
#        }
#    )
#    dataset = ingest_data.grab_mnist_category_dataset()
#    data_shape = dataset.data_shape
#    input_shape = (None, data_shape[2], data_shape[0], data_shape[1])

#    # Choose model
#    model = models.MNISTModel(
#        input_shape=input_shape, output_dims=dataset.output_dims,
#        training_dpath=dataset.training_dpath, **hyperparams)

#    # Initialize architecture
#    model.initialize_architecture()

#    # Load previously learned weights or initialize new weights
#    if model.has_saved_state():
#        model.load_model_state()
#    else:
#        model.reinit_weights()

#    config = dict(
#        learning_rate_schedule=15,
#        max_epochs=120,
#        show_confusion=False,
#        run_test=None,
#        show_features=False,
#        print_timing=False,
#    )

#    X_train, y_train = dataset.load_subset('train')
#    X_valid, y_valid = dataset.load_subset('valid')
#    #X_test, y_test = utils.load_from_fpath_dicts(data_fpath_dict, label_fpath_dict, 'test')
#    harness.train(model, X_train, y_train, X_valid, y_valid, dataset, config)


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
