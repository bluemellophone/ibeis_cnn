#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines the models and the data we will send to the harness

FIXME:
    sometimes you have to chown -R user:user ~/.theano or run with sudo the
    first time after roboot, otherwise you get errors

CommandLineHelp:
    python -m ibeis_cnn --tf netrun <networkmodel>

    --dataset, --ds = <dstag>:<subtag>
        dstag is the main dataset name (eg PZ_MTEST), subtag are parameters to
        modify (max_examples=3)

    --weights, -w = |new|<checkpoint_tag>|<dstag>:<checkpoint_tag> (default: <checkpoint_tag>)
        new will initialize clean weights.
        a checkpoint tag will try to to match a saved model state in the history.
        can load weights from an external dataset.
        <checkpoint_tag> defaults to current

    --arch, -a = <archtag>
        model architecture tag (eg siaml2_128, siam2stream, viewpoint)

    --device = <processor>
       sets theano device flag to a processor like gpu0, gpu1, or cpu0
"""
from __future__ import absolute_import, division, print_function
from ibeis_cnn import models
from ibeis_cnn import ingest_data
from ibeis_cnn import harness
from ibeis_cnn import experiments
import utool as ut
import sys
print, rrr, profile = ut.inject2(__name__, '[ibeis_cnn.train]')


# This is more of a history tag
CHECKPOINT_TAG_ALIAS = {
    'current': None,
    '': None,
}

# second level of alias indirection
# This is more of a dataset tag
DS_TAG_ALIAS2 = {
    'flankhack'    : "dict(acfg_name='ctrl:pername=None,excluderef=False,contrib_contains=FlankHack', colorspace='gray', db='PZ_Master1')",
    'pzmtest-bgr'  : "PZ_MTEST;dict(colorspace='bgr', controlled=True, max_examples=None, num_top=None)",  # NOQA

    'pzmtest'      : "PZ_MTEST;dict(colorspace='gray', controlled=True, max_examples=None, num_top=None)",  # NOQA
    'gz-gray'      : "GZ_ALL;dict(colorspace='gray', controlled=False, max_examples=None, num_top=None)",  # NOQA

    'liberty'      : "liberty;dict(detector='dog', pairs=250000)",

    'combo'        : 'combo_vdsujffw',
    'timectrl_pzmaster1'    : "PZ_Master1;dict(acfg_name='timectrl', colorspace='gray', min_featweight=0.99)"  # NOQA
}


def netrun():
    r"""
    CommandLine:
        # --- UTILITY
        python -m ibeis_cnn --tf get_juction_dpath --show

        # --- DATASET BUILDING ---
        # Build Dataset Aliases
        python -m ibeis_cnn --tf netrun --db PZ_MTEST --acfg ctrl --ensuredata --show
        python -m ibeis_cnn --tf netrun --db PZ_Master1 --acfg timectrl --ensuredata
        python -m ibeis_cnn --tf netrun --db PZ_Master1 --acfg timectrl:pername=None --ensuredata
        python -m ibeis_cnn --tf netrun --db mnist --ensuredata --show
        python -m ibeis_cnn --tf netrun --db mnist --ensuredata --show --datatype=category
        python -m ibeis_cnn --tf netrun --db mnist --ensuredata --show --datatype=siam-patch

        python -m ibeis_cnn --tf netrun --db PZ_Master1 --acfg ctrl:pername=None,excluderef=False,contrib_contains=FlankHack --ensuredata --show --datatype=siam-part

        # Parts based datasets
        python -m ibeis_cnn --tf netrun --db PZ_MTEST --acfg ctrl --datatype=siam-part --ensuredata --show

        # --- TRAINING ---
        python -m ibeis_cnn --tf netrun --ds timectrl_pzmaster1 --acfg ctrl:pername=None,excluderef=False,contrib_contains=FlankHack --train --weights=new --arch=siaml2_128  --monitor  # NOQA
        python -m ibeis_cnn --tf netrun --ds pzmtest --weights=new --arch=siaml2_128 --train --monitor --DEBUG_AUGMENTATION
        python -m ibeis_cnn --tf netrun --ds pzmtest --weights=new --arch=siaml2_128 --train --monitor

        python -m ibeis_cnn --tf netrun --ds flankhack --weights=new --arch=siaml2_partmatch --train --monitor --learning_rate=.00001

        # Different ways to train mnist
        python -m ibeis_cnn --tf netrun --db mnist --weights=new --arch=mnist_siaml2 --train --monitor --datatype=siam-patch
        python -m ibeis_cnn --tf netrun --db mnist --weights=new --arch=mnist-category --train --monitor --datatype=category

        # --- INITIALIZED-TRAINING ---
        python -m ibeis_cnn --tf netrun --ds pzmtest --arch=siaml2_128 --weights=gz-gray:current --train --monitor

        # --- TESTING ---
        python -m ibeis_cnn --tf netrun --db liberty --weights=liberty:current --arch=siaml2_128 --test
        python -m ibeis_cnn --tf netrun --db PZ_Master0 --weights=combo:current --arch=siaml2_128 --testall

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.netrun import *  # NOQA
        >>> netrun()
        >>> ut.show_if_requested()
    """

    requests, hyperparams, tags = parse_args()
    ds_tag         = tags['ds_tag']
    datatype       = tags['datatype']
    extern_ds_tag  = tags['extern_ds_tag']
    arch_tag       = tags['arch_tag']
    checkpoint_tag = tags['checkpoint_tag']

    # ----------------------------
    # Choose the main dataset
    ut.colorprint('[netrun] Ensuring Dataset', 'white')
    dataset = ingest_data.grab_dataset(ds_tag, datatype)
    if extern_ds_tag is not None:
        extern_dpath = ingest_data.get_extern_training_dpath(extern_ds_tag)
    else:
        extern_dpath = None

    if requests['ensuredata']:
        # Print alias key that maps to this particular dataset
        print('Dataset Alias Key: %r' % (dataset.alias_key,))
        print('Current Dataset Tag: %r' % (
            ut.invert_dict(DS_TAG_ALIAS2).get(dataset.alias_key, None),))
        if ut.show_was_requested():
            interact_ = dataset.interact()  # NOQA
            return
        sys.exit(1)

    # ----------------------------
    # Choose model architecture
    # TODO: data will need to return info about number of labels in viewpoint models
    # Specify model archichitecture
    ut.colorprint('[netrun] Architecture Specification', 'white')
    if arch_tag == 'siam2stream':
        model = models.SiameseCenterSurroundModel(
            data_shape=dataset.data_shape,
            training_dpath=dataset.training_dpath, **hyperparams)
    elif arch_tag.startswith('siam'):
        model = models.SiameseL2(
            data_shape=dataset.data_shape,
            arch_tag=arch_tag,
            training_dpath=dataset.training_dpath, **hyperparams)
    elif arch_tag == 'mnist-category':
        model = models.MNISTModel(
            data_shape=dataset.data_shape,
            output_dims=dataset.output_dims,
            arch_tag=arch_tag,
            training_dpath=dataset.training_dpath, **hyperparams)
        pass
    else:
        raise ValueError('Unknown arch_tag=%r' % (arch_tag,))

    ut.colorprint('[netrun] Initialize archchitecture', 'white')
    model.initialize_architecture()

    # ----------------------------
    # Choose weight initialization
    ut.colorprint('[netrun] Setting weights', 'white')
    if checkpoint_tag == 'new':
        ut.colorprint('[netrun] * Initializing new weights', 'lightgray')
        model.reinit_weights()
    else:
        checkpoint_tag = model.resolve_fuzzy_checkpoint_pattern(checkpoint_tag, extern_dpath)
        ut.colorprint('[netrun] * Resolving weights checkpoint_tag=%r' %
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
    if requests['train']:
        ut.colorprint('[netrun] Training Requested', 'white')
        # parse training arguments
        config = ut.argparse_dict(dict(
            learning_rate_schedule=15,
            max_epochs=120,
            learning_rate_adjust=.8,
        ))
        X_train, y_train = dataset.load_subset('train')
        X_valid, y_valid = dataset.load_subset('valid')
        harness.train(model, X_train, y_train, X_valid, y_valid, dataset, config)
    elif requests['test']:
        #assert model.best_results['epoch'] is not None
        if requests['testall']:
            ut.colorprint('[netrun]  * Testing on all data', 'lightgray')
            X_test, y_test = dataset.load_subset('all')
        else:
            ut.colorprint('[netrun]  * Testing on test subset', 'lightgray')
            X_test, y_test = dataset.load_subset('test')
        data, labels = X_test, y_test
        dataname = dataset.alias_key
        experiments.test_siamese_performance(model, data, labels, dataname)
    else:
        raise ValueError('nothing here. need to train or test')

    if requests['publish']:
        ut.colorprint('[netrun] Publish Requested', 'white')
        publish_dpath = ut.truepath('~/Dropbox/IBEIS')
        published_model_state = ut.unixjoin(
            publish_dpath, model.arch_tag + '_model_state.pkl')
        ut.copy(model.get_model_state_fpath(), published_model_state)
        ut.view_directory(publish_dpath)
        print('You need to get the dropbox link and register it into the appropriate file')
        # pip install dropbox
        # https://www.dropbox.com/developers/core/start/python
        # import dropbox  # need oauth
        #client.share('/myfile.txt', short_url=False)
        # https://www.dropbox.com/s/k92s6i5i1hwwy07/siaml2_128_model_state.pkl


def parse_args():
    ds_default = None
    arch_default = 'siaml2_128'
    weights_tag_default = None
    # Test values
    if False:
        ds_default = 'liberty'
        weights_tag_default = 'current'
        assert ut.inIPython()

    # Parse commandline args
    ds_tag      = ut.get_argval(('--dataset', '--ds'), type_=str, default=ds_default)
    datatype    = ut.get_argval(('--datatype', '--dt'), type_=str, default='siam-patch')
    arch_tag    = ut.get_argval(('--arch', '-a'), default=arch_default)
    weights_tag = ut.get_argval(('--weights', '+w'), type_=str, default=weights_tag_default)

    # Incorporate new config stuff?
    #NEW = False
    #if NEW:
    #    from ibeis.experiments import cfghelpers
    #    default_dstag_cfg = {
    #        'ds': 'PZ_MTEST',
    #        'mode': 'patches',
    #        'arch': arch_default
    #    }
    #    named_defaults_dict = {
    #        '': default_dstag_cfg
    #    }
    #    cfghelpers.parse_argv_cfg('dstag', named_defaults_dict=named_defaults_dict)

    hyperparams = ut.argparse_dict(
        {
            #'batch_size': 128,
            'batch_size': 256,
            #'learning_rate': .0005,
            'learning_rate': .1,
            'momentum': .9,
            #'weight_decay': 0.0005,
            'weight_decay': 0.0001,
        },
        alias_dict={
            'weight_decay': ['decay'],
            'learning_rate': ['learn_rate'],
        }
    )
    requests = ut.argparse_dict(
        {
            'train': False,
            'test': False,
            'testall': False,
            'publish': False,
            'ensuredata': False,
        }
    )
    requests['test'] = requests['test'] or requests['testall']

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
    tags = {
        'ds_tag': ds_tag,
        'extern_ds_tag': extern_ds_tag,
        'checkpoint_tag': checkpoint_tag,
        'arch_tag': arch_tag,
        'datatype': datatype,
    }
    ut.colorprint('[netrun] * ds_tag=%r' % (ds_tag,), 'lightgray')
    ut.colorprint('[netrun] * arch_tag=%r' % (arch_tag,), 'lightgray')
    ut.colorprint('[netrun] * extern_ds_tag=%r' % (extern_ds_tag,), 'lightgray')
    ut.colorprint('[netrun] * checkpoint_tag=%r' % (checkpoint_tag,), 'lightgray')
    return requests, hyperparams, tags


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
    harness.train(model, X_train, y_train, X_valid, y_valid, dataset, config)


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
