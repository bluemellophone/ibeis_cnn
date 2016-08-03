# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
import functools
import six
import numpy as np
import utool as ut
from collections import namedtuple
from os.path import join, exists, dirname, basename
from six.moves import cPickle as pickle  # NOQA
import warnings
import sklearn.preprocessing
import ibeis_cnn.__THEANO__ as theano
from ibeis_cnn.__THEANO__ import tensor as T  # NOQA
import ibeis_cnn.__LASAGNE__ as lasagne
from ibeis_cnn import net_strs
from ibeis_cnn import custom_layers
from ibeis_cnn import draw_net
from ibeis_cnn import utils
#ut.noinject('ibeis_cnn.abstract_models')
print, rrr, profile = ut.inject2(__name__, '[ibeis_cnn.abstract_models]')


TheanoFuncs = namedtuple('TheanoFuncs', (
    'theano_backprop', 'theano_forward', 'theano_predict', 'updates'))


Conv2DLayer = custom_layers.Conv2DLayer
MaxPool2DLayer = custom_layers.MaxPool2DLayer


def imwrite_wrapper(show_func):
    r""" helper to convert show funcs into imwrite funcs """
    def imwrite_func(model, dpath=None, dpi=180, asdiagnostic=True,
                     ascheckpoint=None, verbose=1, **kwargs):
        import plottool as pt
        # Resolve path to save the image
        if dpath is None:
            if ascheckpoint is True:
                history_hashid = model.get_model_history_hashid()
                dpath = model._get_model_dpath(checkpoint_tag=history_hashid)
            elif asdiagnostic is True:
                dpath = model.get_epoch_diagnostic_dpath()
            else:
                dpath = model.training_dpath
        # Make the image
        fig = show_func(model, **kwargs)
        # Save the image
        output_fpath = pt.save_figure(fig=fig, dpath=dpath, dpi=dpi,
                                      verbose=verbose)
        return output_fpath
    return imwrite_func


def evaluate_layer_list(network_layers_def, verbose=None):
    r"""
    compiles a sequence of partial functions into a network
    """
    if verbose is None:
        verbose = utils.VERBOSE_CNN
    total = len(network_layers_def)
    network_layers = []
    if verbose:
        print('Evaluting List of %d Layers' % (total,))
    layer_fn_iter = iter(network_layers_def)
    #if True:
    #with warnings.catch_warnings():
    #    warnings.filterwarnings(
    #        'ignore', '.*The uniform initializer no longer uses Glorot.*')
    try:
        with ut.Indenter(' ' * 4, enabled=verbose):
            next_args = tuple()
            for count, layer_fn in enumerate(layer_fn_iter, start=1):
                if verbose:
                    print('Evaluating layer %d/%d (%s) ' %
                          (count, total, ut.get_funcname(layer_fn), ))
                with ut.Timer(verbose=False) as tt:
                    layer = layer_fn(*next_args)
                next_args = (layer,)
                network_layers.append(layer)
                if verbose:
                    print('  * took %.4fs' % (tt.toc(),))
                    print('  * layer = %r' % (layer,))
                    if hasattr(layer, 'input_shape'):
                        print('  * layer.input_shape = %r' % (
                            layer.input_shape,))
                    if hasattr(layer, 'shape'):
                        print('  * layer.shape = %r' % (
                            layer.shape,))
                    print('  * layer.output_shape = %r' % (
                        layer.output_shape,))
    except Exception as ex:
        keys = ['layer_fn', 'layer_fn.func', 'layer_fn.args',
                'layer_fn.keywords', 'layer', 'count']
        ut.printex(ex,
                   ('Error buildling layers.\n'
                    'layer.name=%r') % (layer),
                   keys=keys)
        raise
    return network_layers


def testdata_model_with_history():
    model = BaseModel()
    # make a dummy history
    X_train, y_train = [1, 2, 3], [0, 0, 1]
    rng = np.random.RandomState(0)
    def dummy_epoch_dict(num):
        epoch_info = {
            'epoch': num,
            'loss': 1 / np.exp(num / 10) + rng.rand() / 100,
            'train_loss': 1 / np.exp(num / 10) + rng.rand() / 100,
            'train_loss_regularized': (1 / np.exp(num / 10) +
                                       np.exp(rng.rand() * num) +
                                       rng.rand() / 100),
            'valid_loss': 1 / np.exp(num / 10) - rng.rand() / 100,
            'param_update_mags': {
                'C0': (rng.normal() ** 2, rng.rand()),
                'F1': (rng.normal() ** 2, rng.rand()),
            }
        }
        return epoch_info
    count = 0
    for era_length in [4, 4, 4]:
        alias_key = 'dummy_alias_key'
        model.start_new_era(X_train, y_train, X_train, y_train, alias_key)
        for count in range(count, count + era_length):
            model.record_epoch(dummy_epoch_dict(count))
    #model.record_epoch({'epoch': 1, 'valid_loss': .8, 'train_loss': .9})
    #model.record_epoch({'epoch': 2, 'valid_loss': .5, 'train_loss': .7})
    #model.record_epoch({'epoch': 3, 'valid_loss': .3, 'train_loss': .6})
    #model.record_epoch({'epoch': 4, 'valid_loss': .2, 'train_loss': .3})
    #model.record_epoch({'epoch': 5, 'valid_loss': .1, 'train_loss': .2})
    return model


@six.add_metaclass(ut.ReloadingMetaclass)
class BaseModel(object):
    """
    Abstract model providing functionality for all other models to derive from
    """
    def __init__(model, output_dims=None, input_shape=None, batch_size=None,
                 strict_batch_size=True, data_shape=None, training_dpath='.',
                 momentum=.9, weight_decay=.0005, learning_rate=.001,
                 arch_tag=None):
        """

        Guess on Shapes:
            input_shape (tuple): in Theano format (b, c, h, w)
            data_shape (tuple):  in  Numpy format (b, h, w, c)
        """

        if input_shape is None and data_shape is not None:
            if strict_batch_size is True:
                strict_batch_size = batch_size
            elif strict_batch_size is False:
                strict_batch_size = None
            else:
                raise AssertionError('strict_batch_size must be a bool')
            input_shape = (strict_batch_size,
                           data_shape[2],
                           data_shape[0],
                           data_shape[1])
        if data_shape is None and input_shape is not None:
            data_shape = (input_shape[2], input_shape[3], input_shape[1])
        model.data_shape = data_shape
        #model.network_layers = None  # We don't need to save all of these
        model.arch_tag = arch_tag
        model.output_layer = None
        model.output_dims = output_dims
        model.input_shape = input_shape
        model.batch_size = batch_size
        # bad name, says that this network will take
        # 2*N images in a batch and N labels that map to
        # two images a piece
        model.data_per_label_input  = 1  # state of network input
        model.data_per_label_output = 1  # state of network output
        model.training_dpath = training_dpath  # TODO
        # era=(group of epochs)
        model.current_era = None
        model.era_history = []
        # Training state
        model.requested_headers = ['train_loss', 'valid_loss', 'trainval_rat']
        model.preproc_kw   = None
        model.best_results = {
            'epoch': None,
            'train_loss':     np.inf,
            'valid_loss':     np.inf,
        }
        model.best_weights = None
        model.learning_state = {
            'momentum': momentum,
            'weight_decay': weight_decay,
        }
        # Theano shared state
        model.shared_state = {
            'learning_rate': None,
        }
        # TODO: do not set learning rate until theano is initialized
        model.learning_rate = learning_rate

    # --- OTHER
    @property
    def input_batchsize(model):
        return model.input_shape[0]

    @property
    def input_channels(model):
        return model.input_shape[1]

    @property
    def input_height(model):
        return model.input_shape[2]

    @property
    def input_width(model):
        return model.input_shape[3]

    # --- INITIALIZATION

    def get_epoch_diagnostic_dpath(model, epoch=None):
        import utool as ut
        #if epoch is None:
        #    # use best epoch if not specified
        #    # (WARNING: Make sure the weights in the model are
        #    model.best_weights)
        #    # they may be out of sync
        #    epoch = model.best_results['epoch']
        history_hashid = model.get_model_history_hashid()
        diagnostic_dpath = ut.unixjoin(model.training_dpath, 'diagnostics')
        ut.ensuredir(diagnostic_dpath )
        epoch_dpath = ut.unixjoin(diagnostic_dpath, history_hashid)
        ut.ensuredir(epoch_dpath)
        return epoch_dpath

    def initialize_architecture(model):
        raise NotImplementedError('reimplement')

    def assert_valid_data(model, X_train):
        # Check to make sure data agrees with input
        # FIXME: This check should not be in this fuhnction
        return
        input_layer = model.get_all_layers()[0]
        expected_item_shape = ut.take(input_layer.shape[1:], [1, 2, 0])
        expected_item_shape = tuple(expected_item_shape)
        given_item_shape = X_train.shape[1:]
        assert given_item_shape == expected_item_shape, (
            'inconsistent item shape: ' +
            ('expected_item_shape = %r, ' % (expected_item_shape,)) +
            ('given_item_shape = %r' % (given_item_shape,))
        )

    def is_train_state_initialized(model):
        # TODO: move to dataset. This is independant of the model.
        return model.preproc_kw is not None

    def ensure_training_state(model, X_train, y_train):
        # TODO: move to dataset. This is independant of the model.
        if model.preproc_kw is None:
            # TODO: move this to data preprocessing, not model preprocessing
            model.preproc_kw = {}
            print('computing center mean.')
            model.preproc_kw['center_mean'] = np.mean(X_train, axis=0)
            print('computing center std.')
            if ut.is_int(X_train):
                ut.assert_inbounds(X_train, 0, 255, eq=True,
                                   verbose=ut.VERBOSE)
                model.preproc_kw['center_std'] = 255.0
            else:
                ut.assert_inbounds(X_train, 0.0, 1.0, eq=True,
                                   verbose=ut.VERBOSE)
                model.preproc_kw['center_std'] = 1.0
        if getattr(model, 'encoder', None) is None:
            if hasattr(model, 'initialize_encoder'):
                model.initialize_encoder(y_train)

    def reinit_weights(model, W=lasagne.init.Orthogonal()):
        """
        initailizes weights after the architecture has been defined.
        """
        print('Reinitializing all weights to %r' % (W,))
        weights_list = model.get_all_params(regularizable=True, trainable=True)
        #print(weights_list)
        for weights in weights_list:
            #print(weights)
            shape = weights.get_value().shape
            new_values = W.sample(shape)
            weights.set_value(new_values)

    # --- HASH ID

    def get_architecture_hashid(model):
        """
        Returns a hash identifying the architecture of the determenistic net
        """
        architecture_str = model.get_architecture_str(with_noise_layers=False)
        hashid = ut.hashstr27(architecture_str)
        return hashid

    def get_model_history_hashid(model):
        r"""
        Returns:
            str: history_hashid

        CommandLine:
            python -m ibeis_cnn.abstract_models --test-get_model_history_hashid

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis_cnn.abstract_models import *  # NOQA
            >>> # test the hashid
            >>> model = testdata_model_with_history()
            >>> history_hashid = model.get_model_history_hashid()
            >>> result = str(history_hashid)
            >>> print(result)
            hist_eras002_epochs0005_bdpueuzgkxvtwmpe

        """
        era_history_hash = [ut.hashstr27(repr(era))
                            for era in  model.era_history]
        hashid = ut.hashstr27(str(era_history_hash))
        total_epochs = model.get_total_epochs()
        total_eras = len(model.era_history)
        history_hashid = 'hist_eras%03d_epochs%04d_%s' % (
            total_eras, total_epochs, hashid)
        return history_hashid

    def get_total_epochs(model):
        total_epochs = sum([len(era['epoch_list'])
                            for era in model.era_history])
        return total_epochs

    # --- Input/Output

    def list_saved_checkpoints(model):
        dpath = model._get_model_dpath(None, True)
        checkpoint_dirs = sorted(ut.glob(dpath, '*', fullpath=False))
        return checkpoint_dirs

    def _get_model_dpath(model, dpath, checkpoint_tag):
        dpath = model.training_dpath if dpath is None else dpath
        if checkpoint_tag is not None:
            # checkpoint dir requested
            dpath = join(dpath, 'checkpoints')
            if checkpoint_tag is not True:
                # specific checkpoint requested
                dpath = join(dpath, checkpoint_tag)
        return dpath

    def _get_model_file_fpath(model, default_fname, fpath, dpath, fname,
                              checkpoint_tag):
        if fpath is None:
            fname = default_fname if fname is None else fname
            dpath = model._get_model_dpath(dpath, checkpoint_tag)
            fpath = join(dpath, fname)
        else:
            assert checkpoint_tag is None, 'fpath overrides all other settings'
            assert dpath is None, 'fpath overrides all other settings'
            assert fname is None, 'fpath overrides all other settings'
        return fpath

    def resolve_fuzzy_checkpoint_pattern(model, checkpoint_pattern,
                                         extern_dpath=None):
        r"""
        tries to find a matching checkpoint so you dont have to type a full
        hash
        """
        dpath = model._get_model_dpath(extern_dpath, checkpoint_pattern)
        if exists(dpath):
            checkpoint_tag = checkpoint_pattern
        else:
            checkpoint_dpath = dirname(dpath)
            checkpoint_globpat = '*' + checkpoint_pattern + '*'
            matching_dpaths = ut.glob(checkpoint_dpath, checkpoint_globpat)
            if len(matching_dpaths) == 0:
                raise RuntimeError(
                    'Could not resolve checkpoint_pattern=%r. No Matches' %
                    (checkpoint_pattern,))
            elif len(matching_dpaths) > 1:
                raise RuntimeError(
                    ('Could not resolve checkpoint_pattern=%r. '
                        'matching_dpaths=%r. Too many matches') %
                    (checkpoint_pattern, matching_dpaths))
            else:
                checkpoint_tag = basename(matching_dpaths[0])
                print('Resolved checkpoint pattern to checkpoint_tag=%r' %
                        (checkpoint_tag,))
        return checkpoint_tag

    def has_saved_state(model, checkpoint_tag=None):
        """
        Check if there are any saved model states matching the checkpoing tag.
        """
        fpath = model.get_model_state_fpath(checkpoint_tag=checkpoint_tag)
        if checkpoint_tag is not None:
            ut.assertpath(fpath)
        return ut.checkpath(fpath)

    def get_model_state_fpath(model, fpath=None, dpath=None, fname=None,
                              checkpoint_tag=None):
        default_fname = 'model_state_arch_%s.pkl' % (
            model.get_architecture_hashid())
        model_state_fpath = model._get_model_file_fpath(default_fname, fpath,
                                                        dpath, fname,
                                                        checkpoint_tag)
        return model_state_fpath

    def get_model_info_fpath(model, fpath=None, dpath=None, fname=None,
                             checkpoint_tag=None):
        default_fname = 'model_info_arch_%s.pkl' % (
            model.get_architecture_hashid())
        model_state_fpath = model._get_model_file_fpath(default_fname, fpath,
                                                        dpath, fname,
                                                        checkpoint_tag)
        return model_state_fpath

    def checkpoint_save_model_state(model):
        history_hashid = model.get_model_history_hashid()
        fpath = model.get_model_state_fpath(checkpoint_tag=history_hashid)
        ut.ensuredir(dirname(fpath))
        model.save_model_state(fpath=fpath)

    def checkpoint_save_model_info(model):
        history_hashid = model.get_model_history_hashid()
        fpath = model.get_model_info_fpath(checkpoint_tag=history_hashid)
        ut.ensuredir(dirname(fpath))
        model.save_model_info(fpath=fpath)

    def save_model_state(model, **kwargs):
        """ saves current model state """
        current_weights = model.get_all_param_values()
        model_state = {
            'best_results': model.best_results,
            'preproc_kw':   model.preproc_kw,
            'best_weights': model.best_weights,
            'current_weights': current_weights,
            'input_shape':  model.input_shape,
            'output_dims':  model.output_dims,
            'era_history':  model.era_history,
            'arch_tag': model.arch_tag,
            'data_shape': model.data_shape,
            'batch_size': model.data_shape,
        }
        model_state_fpath = model.get_model_state_fpath(**kwargs)
        print('saving model state to: %s' % (model_state_fpath,))
        with open(model_state_fpath, 'wb') as file_:
            pickle.dump(model_state, file_, protocol=2)  # Use protocol 2 to support python2 and 3
        print('finished saving')

    def save_model_info(model, **kwargs):
        """ save model information (history and results but no weights) """
        model_info = {
            'best_results': model.best_results,
            'input_shape':  model.input_shape,
            'output_dims':  model.output_dims,
            'era_history':  model.era_history,
        }
        model_info_fpath = model.get_model_state_fpath(**kwargs)
        print('saving model info to: %s' % (model_info_fpath,))
        with open(model_info_fpath, 'wb') as file_:
            pickle.dump(model_info, file_, protocol=2)  # Use protocol 2 to support python2 and 3

        print('finished saving')

    def load_model_state(model, **kwargs):
        """
        from six.moves import cPickle as pickle
        kwargs = {}
        TODO: resolve load_model_state and load_extern_weights into a single
            function that is less magic in what it does and more
            straightforward
        """
        model_state_fpath = model.get_model_state_fpath(**kwargs)
        print('[model] loading model state from: %s' % (model_state_fpath,))
        with open(model_state_fpath, 'rb') as file_:
            model_state = pickle.load(file_)
        if model.__class__.__name__ != 'BaseModel':
            assert model_state['input_shape'][1:] == model.input_shape[1:], (
                'architecture disagreement')
            assert model_state['output_dims'] == model.output_dims, (
                'architecture disagreement')
            model.preproc_kw   = model_state['preproc_kw']
            model.best_weights = model_state['best_weights']
        else:
            # HACK TO LOAD ABSTRACT MODEL FOR DIAGNOSITIC REASONS
            print("WARNING LOADING ABSTRACT MODEL")
        model.best_results = model_state['best_results']
        model.input_shape  = model_state['input_shape']
        model.output_dims  = model_state['output_dims']
        model.era_history  = model_state.get('era_history', [None])
        if model.__class__.__name__ != 'BaseModel':
            # hack for abstract model
            # model.output_layer is not None
            model.set_all_param_values(model.best_weights)

    def load_state_from_dict(model, dict_):
        # TODO: make this the general unserialize function that loads the model
        # state
        model.era_history  = dict_.get('era_history', model.era_history)

    def load_extern_weights(model, **kwargs):
        """ load weights from another model """
        model_state_fpath = model.get_model_state_fpath(**kwargs)
        print('[model] loading extern weights from: %s' % (model_state_fpath,))
        with open(model_state_fpath, 'rb') as file_:
            model_state = pickle.load(file_)
        if False or utils.VERBOSE_CNN:
            print('External Model State:')
            print(ut.dict_str(model_state, truncate=True))
        # check compatibility with this architecture
        assert model_state['input_shape'][1:] == model.input_shape[1:], (
            'architecture disagreement')
        assert model_state['output_dims'] == model.output_dims, (
            'architecture disagreement')
        # Just set the weights, no other training state variables
        model.set_all_param_values(model_state['best_weights'])
        # also need to make sure the same preprocessing is used
        # TODO make this a layer?
        model.preproc_kw = model_state['preproc_kw']
        model.era_history = model_state['era_history']

    def load_old_weights_kw(model, old_weights_fpath):
        print('[model] loading old model state from: %s' % (
            old_weights_fpath,))
        with open(old_weights_fpath, 'rb') as file_:
            oldkw = pickle.load(file_)
        # Model architecture and weight params
        data_shape  = oldkw['model_shape'][1:]
        input_shape = (None, data_shape[2], data_shape[0], data_shape[1])
        output_dims  = oldkw['output_dims']

        if model.output_dims is None:
            model.output_dims = output_dims

        # Perform checks
        assert input_shape[1:] == model.input_shape[1:], (
            'architecture disagreement')
        assert output_dims == model.output_dims, (
            'architecture disagreement')

        # Set class attributes
        model.best_weights = oldkw['best_weights']

        model.preproc_kw = {
            'center_mean' : oldkw['center_mean'],
            'center_std'  : oldkw['center_std'],
        }
        model.best_results = {
            'epoch'          : oldkw['best_epoch'],
            'test_accuracy'  : oldkw['best_test_accuracy'],
            'train_loss'     : oldkw['best_train_loss'],
            'valid_accuracy' : oldkw['best_valid_accuracy'],
            'valid_loss'     : oldkw['best_valid_loss'],
        }

        # Need to build architecture first
        model.initialize_architecture()

        model.encoder = oldkw.get('encoder', None)

        # Set architecture weights
        weights_list = model.best_weights
        model.set_all_param_values(weights_list)
        #learning_state = {
        #    'weight_decay'   : oldkw['regularization'],
        #    'learning_rate'  : oldkw['learning_rate'],
        #    'momentum'       : oldkw['momentum'],
        #}
        #batch_size = oldkw['batch_size']

    def load_old_weights_kw2(model, old_weights_fpath):
        print('[model] loading old model state from: %s' % (old_weights_fpath,))
        with open(old_weights_fpath, 'rb') as file_:
            oldkw = pickle.load(file_)

        # Set class attributes
        model.best_weights = oldkw['best_fit_weights']

        # Model architecture and weight params
        # data_shape  = model.best_weights[0].shape[1:]
        # input_shape = (None, data_shape[2], data_shape[0], data_shape[1])
        output_dims  = model.best_weights[-1][0]

        if model.output_dims is None:
            model.output_dims = output_dims

        model.preproc_kw = {
            'center_mean' : oldkw['data_whiten_mean'],
            'center_std'  : oldkw['data_whiten_std'],
        }
        model.best_results = {
            'epoch'          : oldkw['best_epoch'],
            'test_accuracy'  : oldkw['best_valid_accuracy'],
            'train_loss'     : oldkw['best_train_loss'],
            'valid_accuracy' : oldkw['best_valid_accuracy'],
            'valid_loss'     : oldkw['best_valid_loss'],
        }

        # Need to build architecture first
        model.initialize_architecture()
        model.encoder = oldkw.get('data_label_encoder', None)
        model.batch_size = oldkw['train_batch_size']

        # Set architecture weights
        weights_list = model.best_weights
        model.set_all_param_values(weights_list)

        #learning_state = {
        #    'weight_decay'   : oldkw['regularization'],
        #    'learning_rate'  : oldkw['learning_rate'],
        #    'momentum'       : oldkw['momentum'],
        #}
        #batch_size = oldkw['batch_size']

    # --- HISTORY

    def historyfoohack(model, X_train, y_train, dataset):
        #x_hashid = ut.hashstr_arr(X_train, 'x', alphabet=ut.ALPHABET_27)
        y_hashid = ut.hashstr_arr(y_train, 'y', alphabet=ut.ALPHABET_27)
        #
        train_hashid =  dataset.alias_key + '_' + y_hashid
        era_info = {
            'train_hashid': train_hashid,
            'valid_loss_list': [model.best_results['valid_loss']],
            'train_loss_list': [model.best_results['train_loss']],
            'epoch_list': [model.best_results['epoch']],
            'learning_rate': [model.learning_rate],
            'learning_state': [model.learning_state],
        }
        model.era_history = []
        model.era_history.append(era_info)

    def start_new_era(model, X_train, y_train, X_valid, y_valid, alias_key):
        """
        Used to denote a change in hyperparameters during training.
        """
        # TODO: fix the training data hashid stuff
        y_hashid = ut.hashstr_arr(y_train, 'y', alphabet=ut.ALPHABET_27)
        train_hashid =  alias_key + '_' + y_hashid
        era_info = {
            'train_hashid': train_hashid,
            'arch_hashid': model.get_architecture_hashid(),
            'arch_tag': model.arch_tag,
            'num_train': len(y_train),
            'num_valid': len(y_valid),
            'valid_loss_list': [],
            'train_loss_list': [],
            'epoch_list': [],
            'learning_rate': [model.learning_rate],
            'learning_state': [model.learning_state],
        }
        num_eras = len(model.era_history)
        print('starting new era %d' % (num_eras,))
        #if model.current_era is not None:
        model.current_era = era_info
        model.era_history.append(model.current_era)

    def record_epoch(model, epoch_info):
        """
        Records an epoch in an era.
        """
        # each key/val in an epoch_info dict corresponds to a key/val_list in
        # an era dict.
        for key in epoch_info:
            key_ = key + '_list'
            if key_ not in model.current_era:
                model.current_era[key_] = []
            model.current_era[key_].append(epoch_info[key])

    # --- STRINGS

    def get_state_str(model, other_override_reprs={}):
        era_history_str = ut.list_str(
            [ut.dict_str(era, truncate=True, sorted_=True)
             for era in model.era_history], strvals=True)

        override_reprs = {
            'best_results': ut.dict_str(model.best_results),
            'best_weights': ut.truncate_str(str(model.best_weights)),
            'preproc_kw': ('None' if model.preproc_kw is None else
                           ut.dict_str(model.preproc_kw, truncate=True)),
            'learning_state': ut.dict_str(model.learning_state),
            'learning_rate': model.learning_rate,
            'era_history': era_history_str,
        }
        override_reprs.update(other_override_reprs)
        keys = list(set(model.__dict__.keys()) - set(override_reprs.keys()))
        for key in keys:
            if ut.is_func_or_method(model.__dict__[key]):
                # rrr support
                continue
            override_reprs[key] = repr(model.__dict__[key])

        state_str = ut.dict_str(override_reprs, sorted_=True, strvals=True)
        return state_str

    def get_architecture_str(model, sep='_', with_noise_layers=True):
        """
        with_noise_layers is a boolean that specifies if layers that doesnt
        affect the flow of information in the determenistic setting are to be
        included. IE get rid of dropout.
        """
        if model.output_layer is None:
            return ''
        network_layers = model.get_all_layers()
        if with_noise_layers:
            #weighted_layers = [layer_ for layer_ in network_layers
            #                   if hasattr(layer_, 'W')]
            valid_layers = [layer_ for layer_ in network_layers]
        else:
            valid_layers = [
                layer_ for layer_ in network_layers if
                layer_.__class__.__name__ not in lasagne.layers.noise.__all__
            ]
        layer_str_list = [net_strs.make_layer_str(layer)
                          for layer in valid_layers]
        architecture_str = sep.join(layer_str_list)
        return architecture_str

    # --- PRINTING

    def print_state_str(model, **kwargs):
        print(model.get_state_str(**kwargs))

    def print_layer_info(model):
        net_strs.print_layer_info(model.get_all_layers())

    def print_architecture_str(model, sep='\n  '):
        architecture_str = model.get_architecture_str(sep=sep)
        if architecture_str is None or architecture_str == '':
            architecture_str = 'UNDEFINED'
        print('\nArchitecture:' + sep + architecture_str)

    def print_dense_architecture_str(model):
        print('\n---- Arch Str')
        model.print_architecture_str(sep='\n')
        print('\n---- Layer Info')
        model.print_layer_info()
        print('\n---- HashID')
        print('hashid=%r' % (model.get_architecture_hashid()),)
        print('----')
        # verify results

    # --- IMAGE SHOW

    def show_architecture_image(model, **kwargs):
        import plottool as pt
        layers = model.get_all_layers()
        img = draw_net.make_architecture_image(layers, **kwargs)
        pt.imshow(img)

    def show_era_history(model, fnum=None):
        r"""
        Args:
            fnum (int):  figure number(default = None)

        Returns:
            mpl.Figure: fig

        CommandLine:
            python -m ibeis_cnn --tf BaseModel.show_era_history --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis_cnn.models.abstract_models import *  # NOQA
            >>> model = testdata_model_with_history()
            >>> fnum = None
            >>> model.show_era_history(fnum)
            >>> ut.show_if_requested()
        """
        import plottool as pt
        fnum = pt.ensure_fnum(fnum)
        fig = pt.figure(fnum=fnum, pnum=(1, 1, 1), doclf=True, docla=True)
        next_pnum = pt.make_pnum_nextgen(nRows=2, nCols=2)
        model.show_era_loss(fnum=fnum, pnum=next_pnum(), yscale='log')
        model.show_era_loss(fnum=fnum, pnum=next_pnum(), yscale='linear')
        model.show_era_lossratio(fnum=fnum, pnum=next_pnum())

        model.show_weight_updates(fnum=fnum, pnum=next_pnum())

        pt.set_figtitle('Era History: ' + model.get_model_history_hashid())
        return fig

    def show_era_lossratio(model, fnum=None, pnum=(1, 1, 1)):
        import plottool as pt

        fnum = pt.ensure_fnum(fnum)

        fig = pt.figure(fnum=fnum, pnum=pnum)
        colors = pt.distinct_colors(len(model.era_history))
        num_eras = len(model.era_history)
        for index, era in enumerate(model.era_history):
            epochs = era['epoch_list']
            train_loss = np.array(era['train_loss_list'])
            valid_loss = np.array(era['valid_loss_list'])
            #print('era = %s' % (ut.dict_str(era),))
            #valid_loss_std = np.array(era['valid_loss_std_list'])  # NOQA
            trainvalid_ratio = train_loss / valid_loss

            era_color = colors[index]
            #yscale = 'linear'
            #yscale = 'log'
            if index == len(model.era_history) - 1:
                pt.plot(epochs, trainvalid_ratio, '-o', color=era_color,
                        label='train/valid')
            else:
                pt.plot(epochs, trainvalid_ratio, '-o', color=era_color)

        #append_phantom_legend_label
        pt.set_xlabel('epoch')
        pt.set_ylabel('train/valid ratio')

        if num_eras > 0:
            pt.legend()
        pt.dark_background()
        return fig

    def show_weight_updates(model, fnum=None, pnum=(1, 1, 1)):
        import plottool as pt
        fnum = pt.ensure_fnum(fnum)
        fig = pt.figure(fnum=fnum, pnum=pnum)
        num_eras = len(model.era_history)
        for index, era in enumerate(model.era_history):
            epochs = era['epoch_list']
            if 'param_update_mags_list' not in era:
                continue
            update_mags_list = era['param_update_mags_list']
            # Transpose keys and positions
            param_keys = list(set(ut.flatten([
                dict_.keys() for dict_ in update_mags_list
            ])))
            param_val_list = [
                [list_[param] for list_ in update_mags_list]
                for param in param_keys
            ]
            colors = pt.distinct_colors(len(param_val_list))
            for key, val, color in zip(param_keys, param_val_list, colors):
                update_mag_mean = ut.get_list_column(val, 0)
                update_mag_std = ut.get_list_column(val, 1)  # NOQA
                #pt.plot(epochs, update_mag_mean, marker='-x', color=color)
                if index == len(model.era_history) - 1:
                    pt.interval_line_plot(epochs, update_mag_mean,
                                          update_mag_std, marker='x',
                                          linestyle='-', color=color,
                                          label=key)
                else:
                    pt.interval_line_plot(epochs, update_mag_mean,
                                          update_mag_std, marker='x',
                                          linestyle='-', color=color)
                #, label=valid_label, yscale=yscale)
            pass
        if num_eras > 0:
            pt.legend()

        pt.dark_background()
        return fig

    def show_regularization_stuff(model, fnum=None, pnum=(1, 1, 1)):
        import plottool as pt
        fnum = pt.ensure_fnum(fnum)
        fig = pt.figure(fnum=fnum, pnum=pnum)
        num_eras = len(model.era_history)
        for index, era in enumerate(model.era_history):
            epochs = era['epoch_list']
            if 'update_mags_list' not in era:
                continue
            update_mags_list = era['update_mags_list']
            # Transpose keys and positions
            param_keys = list(set(ut.flatten([
                dict_.keys()
                for dict_ in update_mags_list
            ])))
            param_val_list = [
                [list_[param] for list_ in update_mags_list]
                for param in param_keys
            ]
            colors = pt.distinct_colors(len(param_val_list))
            for key, val, color in zip(param_keys, param_val_list, colors):
                update_mag_mean = ut.get_list_column(val, 0)
                update_mag_std = ut.get_list_column(val, 1)
                if index == len(model.era_history) - 1:
                    pt.interval_line_plot(epochs, update_mag_mean,
                                          update_mag_std, marker='x',
                                          linestyle='-', color=color,
                                          label=key)
                else:
                    pt.interval_line_plot(epochs, update_mag_mean,
                                          update_mag_std, marker='x',
                                          linestyle='-', color=color)
            pass
        if num_eras > 0:
            pt.legend()

        pt.dark_background()
        return fig

    def show_era_loss(model, fnum=None, pnum=(1, 1, 1), yscale='log'):
        import plottool as pt

        fnum = pt.ensure_fnum(fnum)

        fig = pt.figure(fnum=fnum, pnum=pnum)
        num_eras = len(model.era_history)
        colors = pt.distinct_colors(num_eras)
        for index, era in enumerate(model.era_history):
            epochs = era['epoch_list']
            train_loss = era['train_loss_list']
            valid_loss = era['valid_loss_list']
            if 'num_valid' in era:
                valid_label = 'valid_loss ' + str(era['num_valid'])
            if 'num_train' in era:
                train_label = 'train_loss ' + str(era['num_train'])

            era_color = colors[index]
            if index == len(model.era_history) - 1:
                pt.plot(epochs, valid_loss, '-x', color=era_color,
                        label=valid_label, yscale=yscale)
                pt.plot(epochs, train_loss, '-o', color=era_color,
                        label=train_label, yscale=yscale)
            else:
                pt.plot(epochs, valid_loss, '-x', color=era_color,
                        yscale=yscale)
                pt.plot(epochs, train_loss, '-o', color=era_color,
                        yscale=yscale)

        # append_phantom_legend_label
        pt.set_xlabel('epoch')
        pt.set_ylabel('loss')
        if num_eras > 0:
            pt.legend()
        pt.dark_background()
        return fig

    def show_weights_image(model, index=0, *args, **kwargs):
        import plottool as pt
        network_layers = model.get_all_layers()
        cnn_layers = [layer_ for layer_ in network_layers
                      if hasattr(layer_, 'W')]
        layer = cnn_layers[index]
        all_weights = layer.W.get_value()
        layername = net_strs.make_layer_str(layer)
        fig = draw_net.show_convolutional_weights(all_weights, **kwargs)
        history_hashid = model.get_model_history_hashid()
        figtitle = layername + '\n' + history_hashid
        pt.set_figtitle(figtitle, subtitle='shape=%r, sum=%.4f, l2=%.4f' %
                        (all_weights.shape, all_weights.sum(),
                         (all_weights ** 2).sum()))
        return fig

    # --- IMAGE WRITE

    imwrite_era_history = imwrite_wrapper(show_era_history)

    imwrite_weights = imwrite_wrapper(show_weights_image)

    #imwrite_architecture = imwrite_wrapper(show_architecture_image)

    def imwrite_architecture(model, fpath='arch_image.png'):
        # FIXME
        layers = model.get_all_layers()
        draw_net.imwrite_architecture(layers, fpath)
        ut.startfile(fpath)

    # ---- UTILITY

    def set_all_param_values(model, weights_list):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', '.*topo.*')
            lasagne.layers.set_all_param_values(
                model.output_layer, weights_list)

    def get_all_param_values(model):
        weights_list = lasagne.layers.get_all_param_values(model.output_layer)
        return weights_list

    def get_all_params(model, **tags):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', '.*topo.*')
            parameters = lasagne.layers.get_all_params(
                model.output_layer, **tags)
            return parameters

    def get_all_layers(model):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', '.*topo.*')
            warnings.filterwarnings('ignore', '.*layer.get_all_layers.*')
            assert model.output_layer is not None
            network_layers = lasagne.layers.get_all_layers(model.output_layer)
        return network_layers

    def get_output_layer(model):
        if model.output_layer is not None:
            return model.output_layer
        else:
            return None
            #assert model.network_layers is not None, (
            #    'need to initialize architecture first')

    @property
    def learning_rate(model):
        shared_learning_rate = model.shared_learning_rate
        if shared_learning_rate is None:
            return None
        else:
            return shared_learning_rate.get_value()

    @learning_rate.setter
    def learning_rate(model, rate):
        print('[model] setting learning rate to %.9f' % (rate))
        shared_learning_rate = model.shared_state.get('learning_rate', None)
        if shared_learning_rate is None:
            shared_learning_rate = theano.shared(np.cast['float32'](rate))
            model.shared_state['learning_rate'] = shared_learning_rate
        else:
            shared_learning_rate.set_value(np.cast['float32'](rate))

    @property
    def shared_learning_rate(model):
        return model.shared_state.get('learning_rate', None)

    # --- LASAGNE EXPRESSIONS

    def _build_loss_expressions(model, X_batch, y_batch):
        r"""
        Requires that a custom loss function is defined in the inherited class

        Args:
            X_batch (T.tensor4): symbolic expression for input
            y_batch (T.ivector): symbolic expression for labels
        """
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', '.*topo.*')
            warnings.filterwarnings('ignore', '.*get_all_non_bias_params.*')
            warnings.filterwarnings('ignore', '.*layer.get_output.*')

            network_output = lasagne.layers.get_output(model.output_layer,
                                                       X_batch)
            network_output.name = 'network_output'
            network_output_determ = lasagne.layers.get_output(
                model.output_layer, X_batch, deterministic=True)
            network_output_determ.name = 'network_output_determ'

            try:
                print('Building symbolic loss function')
                losses = model.loss_function(network_output, y_batch)
                loss = lasagne.objectives.aggregate(losses, mode='mean')
                loss.name = 'loss'

                print('Building symbolic loss function (determenistic)')
                losses_determ = model.loss_function(
                    network_output_determ, y_batch)
                loss_determ = lasagne.objectives.aggregate(
                    losses_determ, mode='mean')
                loss_determ.name = 'loss_determ'

                # Regularize
                # TODO: L2 should be one of many available options for
                # regularization
                L2 = lasagne.regularization.regularize_network_params(
                    model.output_layer, lasagne.regularization.l2)
                weight_decay = model.learning_state['weight_decay']
                regularization_term = weight_decay * L2
                regularization_term.name = 'regularization_term'
                #L2 = lasagne.regularization.l2(model.output_layer)
                loss_regularized = loss + regularization_term
                loss_regularized.name = 'loss_regularized'
            except TypeError:
                loss, loss_determ, loss_regularized = None, None, None

            loss_expr_dict = {
                'loss': loss,
                'loss_determ': loss_determ,
                'loss_regularized': loss_regularized,
                'network_output': network_output,
                'network_output_determ': network_output_determ,
            }

            return loss_expr_dict

    def build_theano_funcs(model,
                           input_type=T.tensor4, output_type=T.ivector,
                           request_backprop=True,
                           request_forward=True,
                           request_predict=False):
        """
        Builds the Theano functions (symbolic expressions) that will be used in
        the optimization.  Requires that a custom loss function is defined in
        the inherited class

        References:
            # refer to this link for info on tensor types:
            http://deeplearning.net/software/theano/library/tensor/basic.html
        """
        print('[model] building Theano primitives...')
        X = input_type('x')
        y = output_type('y')
        X_batch = input_type('x_batch')
        y_batch = output_type('y_batch')

        loss_expr_dict = model._build_loss_expressions(X_batch, y_batch)
        loss             = loss_expr_dict['loss']
        loss_determ      = loss_expr_dict['loss_determ']
        loss_regularized = loss_expr_dict['loss_regularized']
        network_output_determ   = loss_expr_dict['network_output_determ']

        # Run inference and get other_outputs
        unlabeled_outputs = model.build_unlabeled_output_expressions(
            network_output_determ)
        labeled_outputs   = model.build_labeled_output_expressions(
            network_output_determ, y_batch)
        updates = None

        if request_backprop:
            print('[model.build_theano_funcs] request_backprop')
            learning_rate_theano = model.shared_learning_rate
            momentum = model.learning_state['momentum']
            # Define updates network parameters based on the training loss
            parameters = model.get_all_params(trainable=True)
            gradients_regularized = theano.grad(
                loss_regularized, parameters, add_names=True)
            updates = lasagne.updates.nesterov_momentum(
                gradients_regularized, parameters, learning_rate_theano,
                momentum)

            # Build outputs to babysit training
            monitor_outputs = []
            for param in parameters:
                # The vector each param was udpated with
                # (one vector per channel)
                param_update_vec = updates[param] - param
                param_update_vec.name = 'param_update_vector_' + param.name
                flat_shape = (param_update_vec.shape[0],
                              T.prod(param_update_vec.shape[1:]))
                flat_param_update_vec = param_update_vec.reshape(flat_shape)
                param_update_mag = (flat_param_update_vec ** 2).sum(-1)
                param_update_mag.name = 'param_update_magnitude_' + param.name
                monitor_outputs.append(param_update_mag)

            theano_backprop = theano.function(
                inputs=[theano.In(X_batch), theano.In(y_batch)],
                outputs=([loss_regularized, loss] + labeled_outputs +
                         monitor_outputs),
                updates=updates,
                givens={
                    X: X_batch,
                    y: y_batch,
                },
            )
            theano_backprop.name = ':theano_backprob:explicit'
        else:
            theano_backprop = None

        if request_forward:
            print('[model.build_theano_funcs] request_forward')
            theano_forward = theano.function(
                inputs=[theano.In(X_batch), theano.In(y_batch)],
                outputs=[loss_determ] + labeled_outputs + unlabeled_outputs,
                updates=None,
                givens={
                    X: X_batch,
                    y: y_batch,
                },
            )
            theano_forward.name = ':theano_forward:explicit'
        else:
            theano_forward = None

        if request_predict:
            print('[model.build_theano_funcs] request_predict')
            theano_predict = theano.function(
                inputs=[theano.In(X_batch)],
                outputs=[network_output_determ] + unlabeled_outputs,
                updates=None,
                givens={
                    X: X_batch,
                },
            )
            theano_predict.name = ':theano_predict:explicit'
        else:
            theano_predict = None

        print('[model.build_theano_funcs] exit')
        theano_funcs  = TheanoFuncs(
            theano_backprop, theano_forward, theano_predict, updates)
        return theano_funcs

    def build_unlabeled_output_expressions(model, network_output):
        """
        override in inherited subclass to enable custom symbolic expressions
        based on the network output alone
        """
        return []

    def build_labeled_output_expressions(model, network_output, y_batch):
        """
        override in inherited subclass to enable custom symbolic expressions
        based on the network output and the labels
        """
        return []

    # Testing

    def make_random_testdata(model, num=1000, seed=0, cv2_format=False):
        print('made random testdata')
        rng = np.random.RandomState(seed)
        num_labels = num
        num_data   = num * model.data_per_label_input
        X_unshared = rng.rand(num_data, *model.data_shape)
        y_unshared = rng.rand(num_labels) * (model.output_dims + 1)
        X_unshared = X_unshared.astype(np.float32)
        y_unshared = y_unshared.astype(np.int32)
        if ut.VERBOSE:
            print('made random testdata')
            print('size(X_unshared) = %r' % (
                ut.get_object_size_str(X_unshared),))
            print('size(y_unshared) = %r' % (
                ut.get_object_size_str(y_unshared),))
        if cv2_format:
            pass
        return X_unshared, y_unshared


class AbstractCategoricalModel(BaseModel):
    """ base model for catagory classifiers """

    def __init__(model, **kwargs):
        super(AbstractCategoricalModel, model).__init__(**kwargs)
        model.encoder = None
        # categorical models have a concept of accuracy
        #model.requested_headers += ['valid_acc', 'test_acc']
        model.requested_headers += ['valid_acc']

    def initialize_encoder(model, labels):
        print('[model] encoding labels')
        model.encoder = sklearn.preprocessing.LabelEncoder()
        model.encoder.fit(labels)
        model.output_dims = len(list(np.unique(labels)))
        print('[model] model.output_dims = %r' % (model.output_dims,))

    def loss_function(model, network_output, truth):
        return T.nnet.categorical_crossentropy(network_output, truth)

    def build_unlabeled_output_expressions(model, network_output):
        # Network outputs define category probabilities
        probabilities = network_output
        predictions = T.argmax(probabilities, axis=1)
        predictions.name = 'predictions'
        confidences = probabilities.max(axis=1)
        confidences.name = 'confidences'
        unlabeled_outputs = [predictions, confidences]
        return unlabeled_outputs

    def build_labeled_output_expressions(model, network_output, y_batch):
        probabilities = network_output
        predictions = T.argmax(probabilities, axis=1)
        predictions.name = 'tmp_predictions'
        accuracy = T.mean(T.eq(predictions, y_batch))
        accuracy.name = 'accuracy'
        labeled_outputs = [accuracy]
        return labeled_outputs


class _PretrainedLayerInitializer(lasagne.init.Initializer):
    def __init__(self, pretrained_layer):
        self.pretrained_layer = pretrained_layer

    def sample(self, shape):
        if len(shape) == 1:
            assert shape[0] <= self.pretrained_layer.shape[0]
            pretrained_weights = self.pretrained_layer[:shape[0]]
        else:
            is_conv = len(shape) == 4
            assert len(shape) == len(self.pretrained_layer.shape), (
                'Layer shape mismatch. Expected %r got %r' % (self.pretrained_layer.shape,
                                                              shape))

            fanout, fanin = shape[:2]
            fanout_, fanin_ = self.pretrained_layer.shape[:2]
            assert fanout <= fanout_, (
                'Cannot cast weights to a larger fan-out dimension')
            assert fanin  <= fanin_,  (
                'Cannot cast weights to a larger fan-in dimension')

            if is_conv:
                height, width = shape[2:]
                height_, width_ = self.pretrained_layer.shape[2:]
                assert height == height_, (
                    'The height must be identical between the layer and weights')
                assert width  == width_,  (
                    'The width must be identical between the layer and weights')

            if is_conv:
                pretrained_weights = self.pretrained_layer[:fanout, :fanin, :, :]
            else:
                pretrained_weights = self.pretrained_layer[:fanout, :fanin]

        pretrained_sample = lasagne.utils.floatX(pretrained_weights)
        return pretrained_sample


class PretrainedNetwork(object):
    """
    Intialize weights from a specified (Caffe) pretrained network layers

    Args:
        layer (int) : int

    CommandLine:
        python -m ibeis_cnn --tf PretrainedNetwork:0
        python -m ibeis_cnn --tf PretrainedNetwork:1

    Example0:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.models import *  # NOQA
        >>> self = PretrainedNetwork('caffenet', show_network=True)
        >>> print('done')

    Example1:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.models import *  # NOQA
        >>> self = PretrainedNetwork('vggnet', show_network=True)
        >>> print('done')
    """
    def __init__(self, modelkey=None, show_network=False):
        from ibeis_cnn._plugin_grabmodels import ensure_model
        self.modelkey = modelkey
        weights_path = ensure_model(modelkey)
        try:
            self.pretrained_weights = ut.load_cPkl(weights_path)
        except Exception:
            raise IOError('The specified model was not found: %r' %
                          (weights_path, ))
        if show_network:
            net_strs.print_pretrained_weights(
                self.pretrained_weights, weights_path)

    def get_num_layers(self):
        return len(self.pretrained_weights)

    def get_layer_num_filters(self, layer_index):
        assert layer_index <= len(self.pretrained_weights), (
            'Trying to specify a layer that does not exist')
        shape = self.pretrained_weights[layer_index].shape
        fanout, fanin, height, width = shape
        return fanout

    def get_layer_filter_size(self, layer_index):
        assert layer_index <= len(self.pretrained_weights), (
            'Trying to specify a layer that does not exist')
        shape = self.pretrained_weights[layer_index].shape
        fanout, fanin, height, width = shape
        return (height, width)

    def get_conv2d_layer(self, layer_index, name=None, **kwargs):
        """ Assumes requested layer is convolutional

        Returns:
            lasange.layers.Layer: Layer
        """
        if name is None:
            name = '%s_layer%r' % (self.modelkey, layer_index)
        W = self.get_pretrained_layer(layer_index)
        try:
            b = self.get_pretrained_layer(layer_index + 1)
            assert W.shape[0] == b.shape[0]
        except:
            b = None
        print(W.shape)
        print(b.shape)
        num_filters = self.get_layer_num_filters(layer_index)
        filter_size = self.get_layer_filter_size(layer_index)
        Layer = functools.partial(
            Conv2DLayer, num_filters=num_filters,
            filter_size=filter_size, W=W, b=b, name=name, **kwargs)
        return Layer

    def get_pretrained_layer(self, layer_index, rand=False):
        assert layer_index <= len(self.pretrained_weights), (
            'Trying to specify a layer that does not exist')
        pretrained_layer = self.pretrained_weights[layer_index]
        weights_initializer = _PretrainedLayerInitializer(pretrained_layer)
        if rand:
            np.random.shuffle(weights_initializer)
        return weights_initializer


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis_cnn.abstract_models
        python -m ibeis_cnn.abstract_models --allexamples
        python -m ibeis_cnn.abstract_models --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
