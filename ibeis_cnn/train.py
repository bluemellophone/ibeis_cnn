#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TODO:
    rename this file to not be train. maybe dev. I'm not sure


Defines the models and the data we will send to the harness

CommandLineHelp:
    ./cnn.py

    --dataset, --ds = <dstag>:<subtag>
        dstag is the main dataset name (eg PZ_MTEST), subtag are parameters to
        modify (max_examples=3)

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
CHECKPOINT_TAG_ALIAS = {
    'current': None,
    '': None,
}

# second level of alias indirection
# This is more of a dataset tag
DS_TAG_ALIAS2 = {
    'nnp3-2-bgr'   : "NNP_Master3;dict(colorspace='bgr', controlled=True, max_examples=None, num_top=None)",  # NOQA
    'pzmaster-bgr' : "PZ_Master0;dict(colorspace='bgr', controlled=True, max_examples=None, num_top=None)",  # NOQA
    'pzmtest-bgr'  : "PZ_MTEST;dict(colorspace='bgr', controlled=True, max_examples=None, num_top=None)",  # NOQA

    'nnp'          : "NNP_Master3;dict(colorspace='gray', controlled=True, max_examples=None, num_top=None)",  # NOQA
    'pzmtest'      : "PZ_MTEST;dict(colorspace='gray', controlled=True, max_examples=None, num_top=None)",  # NOQA
    'gz-gray'      : "GZ_ALL;dict(colorspace='gray', controlled=False, max_examples=None, num_top=None)",  # NOQA
    'girm'         : "NNP_MasterGIRM_core;dict(colorspace='gray', controlled=True, max_examples=None, num_top=None)",  # NOQA

    'pzmaster'     : 'PZ_Master0;dict(controlled=True, max_examples=None, num_top=None)',
    'liberty'      : "liberty;dict(detector='dog', pairs=250000)",

    'combo'        : 'combo_vdsujffw',
}


def pz_patchmatch():
    r"""
    RENAME:
        patchmatch?

    CommandLine:
        THEANO_FLAGS='device=gpu1'

        # --- UTILITY
        python -m ibeis_cnn --tf get_juction_dpath --show

        # --- LIBERTY EXAMPLES ---

        # Build / Ensure Liberty dataset
        python -m ibeis_cnn --tf pz_patchmatch --ds liberty --ensuredata --colorspace='gray'

        # Train on liberty dataset
        python -m ibeis_cnn --tf pz_patchmatch --ds liberty --train --weights=new --arch=siaml2_128 --monitor

        # Continue liberty training using previous learned weights
        python -m ibeis_cnn --tf pz_patchmatch --ds liberty --train --weights=current --arch=siaml2_128 --monitor --learning-rate=.03

        # Test liberty accuracy
        python -m ibeis_cnn --tf pz_patchmatch --ds liberty --test --weights=liberty:current --arch=siaml2_128 --test

        # Initialize a second database using IBEIS
        python -m ibeis_cnn --tf pz_patchmatch --db PZ_MTEST --colorspace='gray' --num-top=None --controlled=True --ensuredata

        # Test accuracy of another dataset using weights from liberty
        python -m ibeis_cnn --tf pz_patchmatch --ds pzmtest --test --weights=liberty:current --arch=siaml2_128 --testall  # NOQA


        # --- DATASET BUILDING ---

        # Build Dataset Aliases
        python -m ibeis_cnn --tf pz_patchmatch --db PZ_Master0 --colorspace='gray' --num-top=None --controlled=True --ensuredata
        python -m ibeis_cnn --tf pz_patchmatch --db PZ_MTEST --colorspace='gray' --num-top=None --controlled=True --ensuredata
        python -m ibeis_cnn --tf pz_patchmatch --db NNP_Master3 --colorspace='gray' --num-top=None --controlled=True --ensuredata
        python -m ibeis_cnn --tf pz_patchmatch --db GZ_ALL --colorspace='gray' --num-top=None --controlled=True --ensuredata
        python -m ibeis_cnn --tf pz_patchmatch --db NNP_MasterGIRM_core --colorspace='gray' --num-top=None --controlled=True --ensuredata

        # --- TRAINING ---
        # Train NNP_Master
        python -m ibeis_cnn --tf pz_patchmatch --ds nnp3-2 --weights=nnp3-2:epochs0011 --arch=siaml2 --train --monitor
        python -m ibeis_cnn --tf pz_patchmatch --ds nnp3-2 --weights=new --arch=siaml2 --train --weight_decay=0.0001 --monitor
        python -m ibeis_cnn --tf pz_patchmatch --ds nnp3-2 --weights=new --arch=siaml2 --train --weight_decay=0.0001 --monitor

        python -m ibeis_cnn --tf pz_patchmatch --ds pzmtest --weights=new --arch=siaml2_128 --train --monitor

        # Train COMBO
        python -m ibeis_cnn --tf pz_patchmatch --ds combo --weights=new --arch=siaml2_128 --train --monitor


        # --- MONITOR TRAINING ---

        # Hyperparameter settings
        python -m ibeis_cnn --tf pz_patchmatch --ds pzmtest --weights=new --arch=siaml2 --train --monitor

        # Grevys
        python -m ibeis_cnn --tf pz_patchmatch --ds gz-gray --weights=new --arch=siaml2 --train --weight_decay=0.0001 --monitor

        # THIS DID WELL VERY QUICKLY
        python -m ibeis_cnn --tf pz_patchmatch --ds pzmtest --weights=new --arch=siaml2 --train --monitor --learning_rate=.1 --weight_decay=0.0005
        python -m ibeis_cnn --tf pz_patchmatch --ds pzmtest --weights=new --arch=siaml2 --train --monitor --DEBUG_AUGMENTATION

        python -m ibeis_cnn --tf pz_patchmatch --ds pzmtest --weights=new --arch=siaml2 --train --monitor

        python -m ibeis_cnn --tf pz_patchmatch --ds nnp3-2 --weights=new --arch=siaml2 --train --monitor
        python -m ibeis_cnn --tf pz_patchmatch --ds nnp3-2 --weights=new --arch=siam2streaml2 --train --monitor

        python -m ibeis_cnn --tf pz_patchmatch --db NNP_Master3 --weights=new --arch=siaml2 --train --monitor --colorspace='bgr' --num_top=None

        python -m ibeis_cnn --tf pz_patchmatch --ds pzmtest-bgr --weights=nnp3-2-bgr:epochs0023_rttjuahuhhraphyb --arch=siaml2 --test
        python -m ibeis_cnn --tf pz_patchmatch --ds pzmaster-bgr --weights=nnp3-2-bgr:epochs0023_rttjuahuhhraphyb --arch=siaml2 --test
        --monitor --colorspace='bgr' --num_top=None

        # --- INITIALIZED-TRAINING ---
        python -m ibeis_cnn --tf pz_patchmatch --ds pzmtest --arch=siaml2 --weights=gz-gray:current --train --monitor

        # --- TESTING ---
        python -m ibeis_cnn --tf pz_patchmatch --db liberty --weights=liberty:current --arch=siaml2_128 --test

        # test combo
        python -m ibeis_cnn --tf pz_patchmatch --db PZ_Master0 --weights=combo:hist_eras007_epochs0098_gvmylbm --arch=siaml2_128 --testall
        python -m ibeis_cnn --tf pz_patchmatch --db PZ_Master0 --weights=combo:current --arch=siaml2_128 --testall

        python -m ibeis_cnn --tf pz_patchmatch --ds gz-gray --arch=siaml2 --weights=gz-gray:current --test
        python -m ibeis_cnn --tf pz_patchmatch --ds nnp3-2 --arch=siaml2 --weights=gz-gray:current --test
        python -m ibeis_cnn --tf pz_patchmatch --ds pzmtest --arch=siaml2 --weights=gz-gray:current --test
        python -m ibeis_cnn --tf pz_patchmatch --ds pzmaster --arch=siaml2 --weights=gz-gray:current --test

        # Test NNP_Master on in sample
        python -m ibeis_cnn --tf pz_patchmatch --ds nnp3-2 --weights=current --arch=siaml2 --test
        python -m ibeis_cnn --tf pz_patchmatch --ds nnp3-2 --weights=nnp3-2 --arch=siaml2 --test

        # Test NNP_Master3 weights on out of sample data
        python -m ibeis_cnn --tf pz_patchmatch --ds pzmtest --weights=nnp3-2:epochs0021 --arch=siaml2 --test
        python -m ibeis_cnn --tf pz_patchmatch --ds pzmtest --weights=nnp3-2:epochs0011 --arch=siaml2 --test

        # Now can use the alias
        python -m ibeis_cnn --tf pz_patchmatch --ds pzmtest --weights=nnp3-2:epochs0021 --arch=siaml2 --test
        python -m ibeis_cnn --tf pz_patchmatch --ds pzmaster --weights=nnp3-2:epochs0021 --arch=siaml2 --test

    Ignore:
        weights_fpath = '/media/raid/work/NNP_Master3/_ibsdb/_ibeis_cache/nets/train_patchmetric((2462)&ju%uw7bta19cunw)/ibeis_cnn_weights.pickle'
        weights_fpath = '/media/raid/work/NNP_Master3/_ibsdb/_ibeis_cache/nets/train_patchmetric((2462)&ju%uw7bta19cunw)/arch_d788de3571330d42/training_state.cPkl'

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.train import *  # NOQA
        >>> pz_patchmatch()
        >>> ut.show_if_requested()
    """
    ut.colorprint('[pz_patchmatch] Ensuring Dataset', 'white')

    ds_default = None
    arch_default = 'siaml2'
    weights_tag_default = None
    # Test values
    if False:
        ds_default = 'liberty'
        weights_tag_default = 'current'
        assert ut.inIPython()

    # Parse commandline args
    ds_tag      = ut.get_argval(('--dataset', '--ds'), type_=str, default=ds_default)
    arch_tag    = ut.get_argval(('--arch', '-a'), default=arch_default)
    weights_tag = ut.get_argval(('--weights', '+w'), type_=str, default=weights_tag_default)
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
    request_train = ut.get_argflag('--train')
    request_testall = ut.get_argflag('--testall')
    request_test = ut.get_argflag('--test') or request_testall
    request_publish = ut.get_argflag('--publish')

    # breakup weights tag into extern_ds and checkpoint
    if weights_tag is not None and ':' in weights_tag:
        extern_ds_tag, checkpoint_tag = weights_tag.split(':')
    else:
        extern_ds_tag = None
        checkpoint_tag = weights_tag
    # resolve aliases
    ds_tag = DS_TAG_ALIAS2.get(ds_tag, ds_tag)
    extern_ds_tag = DS_TAG_ALIAS2.get(extern_ds_tag, extern_ds_tag)
    checkpoint_tag = CHECKPOINT_TAG_ALIAS.get(checkpoint_tag, checkpoint_tag)

    ut.colorprint('[pz_patchmatch] * ds_tag=%r' % (ds_tag,), 'lightgray')
    ut.colorprint('[pz_patchmatch] * arch_tag=%r' % (arch_tag,), 'lightgray')
    ut.colorprint('[pz_patchmatch] * extern_ds_tag=%r' % (extern_ds_tag,), 'lightgray')
    ut.colorprint('[pz_patchmatch] * checkpoint_tag=%r' % (checkpoint_tag,), 'lightgray')

    # ----------------------------
    # Choose the main dataset
    dataset = ingest_data.grab_siam_dataset(ds_tag)
    if extern_ds_tag is not None:
        extern_dpath = ingest_data.get_extern_training_dpath(extern_ds_tag)
    else:
        extern_dpath = None

    if ut.get_argflag('--ensuredata'):
        # Print alias key that maps to this particular dataset
        print('Dataset Alias Key: %r' % (dataset.alias_key,))
        print('Current Dataset Tag: %r' % (
            ut.invert_dict(DS_TAG_ALIAS2).get(dataset.alias_key, None),))
        import sys
        sys.exit(1)

    # ----------------------------
    # Choose model architecture
    # TODO: data will need to return info about number of labels in viewpoint models
    # Specify model archichitecture
    ut.colorprint('[pz_patchmatch] Architecture Specification', 'white')
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

    ut.colorprint('[pz_patchmatch] Initialize archchitecture', 'white')
    model.initialize_architecture()

    # ----------------------------
    # Choose weight initialization
    ut.colorprint('[pz_patchmatch] Setting weights', 'white')
    if checkpoint_tag == 'new':
        ut.colorprint('[pz_patchmatch] * Initializing new weights', 'lightgray')
        model.reinit_weights()
    else:
        checkpoint_tag = model.resolve_fuzzy_checkpoint_pattern(checkpoint_tag, extern_dpath)
        ut.colorprint('[pz_patchmatch] * Resolving weights checkpoint_tag=%r' %
                      (checkpoint_tag,), 'lightgray')
        if extern_dpath is not None:
            model.load_extern_weights(dpath=extern_dpath, checkpoint_tag=checkpoint_tag)
        elif model.has_saved_state(checkpoint_tag=checkpoint_tag):
            model.load_model_state(checkpoint_tag=checkpoint_tag)
        else:
            raise ValueError(('Unresolved weight init: '
                              'checkpoint_tag=%r, extern_ds_tag=%r') % (
                                  checkpoint_tag, extern_ds_tag,))

    #print('Model State:')
    #print(model.get_state_str())
    # ----------------------------
    # Run Actions
    if request_train:
        ut.colorprint('[pz_patchmatch] Training Requested', 'white')
        config = ut.argparse_dict(dict(
            learning_rate_schedule=15,
            max_epochs=120,
            learning_rate_adjust=.8,
        ))
        X_train, y_train = dataset.load_subset('train')
        X_valid, y_valid = dataset.load_subset('valid')
        #X_test, y_test = utils.load_from_fpath_dicts(data_fpath_dict, label_fpath_dict, 'test')
        harness.train(model, X_train, y_train, X_valid, y_valid, dataset, config)
    elif request_test:
        #assert model.best_results['epoch'] is not None
        if request_testall:
            ut.colorprint('[pz_patchmatch]  * Testing on all data', 'lightgray')
            X_test, y_test = dataset.load_subset('all')
        else:
            ut.colorprint('[pz_patchmatch]  * Testing on test subset', 'lightgray')
            X_test, y_test = dataset.load_subset('test')
        data, labels = X_test, y_test
        #data, labels = utils.random_xy_sample(X_test, y_test, 1000, model.data_per_label_input)
        dataname = dataset.alias_key
        experiments.test_siamese_performance(model, data, labels, dataname)
    else:
        raise ValueError('nothing here. need to train or test')

    if request_publish:
        ut.colorprint('[pz_patchmatch] Publish Requested', 'white')
        publish_dpath = ut.truepath('~/Dropbox/IBEIS')
        from os.path import join
        published_model_state = join(publish_dpath, model.arch_tag + '_model_state.pkl')
        ut.copy(model.get_model_state_fpath(), published_model_state)
        ut.vd(publish_dpath)
        print('You need to get the dropbox link and register it into the appropriate file')
        # pip install dropbox
        # https://www.dropbox.com/developers/core/start/python
        # import dropbox  # need oauth
        #client.share('/myfile.txt', short_url=False)
        # https://www.dropbox.com/s/k92s6i5i1hwwy07/siaml2_128_model_state.pkl


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
    ds_tag_list = [DS_TAG_ALIAS2.get(ds_tag, ds_tag) for ds_tag in ds_alias_list]
    dataset_list = [ingest_data.grab_siam_dataset(ds_tag) for ds_tag in ds_tag_list]
    merged_dataset = ingest_data.merge_datasets(dataset_list)
    print(merged_dataset.alias_key)
    return merged_dataset


def train_background():
    r"""
    CommandLine:
        python -m ibeis_cnn.train --test-train_background

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.train import *  # NOQA
        >>> result = train_background()
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

    from os.path import join
    source_path = join('data', 'numpy', 'background_patches')
    data_fpath = join(source_path, 'X.npy')
    labels_fpath = join(source_path, 'y.npy')
    training_dpath = join('data', 'results', 'backgound_patches')
    dataset = ingest_data.get_numpy_dataset(data_fpath, labels_fpath, training_dpath)

    data_shape = dataset.data_shape
    input_shape = (None, data_shape[2], data_shape[0], data_shape[1])

    # Choose model
    model = models.BackgroundModel(
        input_shape=input_shape, output_dims=dataset.output_dims,
        training_dpath=dataset.training_dpath, **hyperparams)

    # Initialize architecture
    model.initialize_architecture()

    # Load previously learned weights or initialize new weights
    if model.has_saved_state():
        model.load_model_state()
    else:
        model.reinit_weights()

    config = dict(
        learning_rate_schedule=15,
        max_epochs=120,
        show_confusion=False,
        run_test=None,
        show_features=False,
        print_timing=False,
    )

    X_train, y_train = dataset.load_subset('train')
    X_valid, y_valid = dataset.load_subset('valid')
    #X_test, y_test = utils.load_from_fpath_dicts(data_fpath_dict, label_fpath_dict, 'test')
    harness.train(model, X_train, y_train, X_valid, y_valid, dataset, config)


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
