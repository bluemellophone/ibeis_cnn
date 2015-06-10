# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import theano
import lasagne
from lasagne import init
import functools
import six
import theano.tensor as T
import numpy as np
#from lasagne import layers
from ibeis_cnn import net_strs
from ibeis_cnn import utils
from ibeis_cnn import custom_layers
from ibeis_cnn import batch_processing as batch
from ibeis_cnn import draw_net
import sklearn.preprocessing
import utool as ut
from os.path import join
import warnings
import cPickle as pickle
ut.noinject('ibeis_cnn.abstract_models')


Conv2DLayer = custom_layers.Conv2DLayer
MaxPool2DLayer = custom_layers.MaxPool2DLayer


def evaluate_layer_list(network_layers_def, verbose=None):
    """ compiles a sequence of partial functions into a network """
    if verbose is None:
        verbose = utils.VERBOSE_CNN
    network_layers = []
    layer_fn_iter = iter(network_layers_def)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore', '.*The uniform initializer no longer uses Glorot.*')
        try:
            count = 0
            if verbose:
                tt = ut.Timer(verbose=False)
                print('Evaluating layer %d' % (count,))
                tt.tic()
            prev_layer = six.next(layer_fn_iter)()
            network_layers.append(prev_layer)
            for count, layer_fn in enumerate(layer_fn_iter, start=1):
                if verbose:
                    print('  * took %.4s' % (tt.toc(),))
                    print('Evaluating layer %d' % (count,))
                    print('  * prev_layer = %r' % (prev_layer,))
                    tt.tic()
                prev_layer = layer_fn(prev_layer)
                network_layers.append(prev_layer)
        except Exception as ex:
            ut.printex(
                ex,
                ('Error buildling layers.\n'
                 'prev_layer.name=%r\n'
                 'count=%r') % (prev_layer, count))
            raise
    return network_layers


@six.add_metaclass(ut.ReloadingMetaclass)
class BaseModel(object):
    """
    Abstract model providing functionality for all other models to derive from
    """
    def __init__(model, output_dims=None, input_shape=None, batch_size=None,
                 training_dpath='.', momentum=.9, weight_decay=.0005,
                 learning_rate=.001):
        #model.network_layers = None  # We really don't need to save all of these
        model.output_layer = None
        model.output_dims = output_dims
        model.input_shape = input_shape
        model.batch_size = batch_size
        # bad name, says that this network will take
        # 2*N images in a batch and N labels that map to
        # two images a piece
        model.data_per_label = 1
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
        model.learning_rate = learning_rate

    # --- initialization steps

    def get_epoch_diagnostic_dpath(model, epoch=None):
        import utool as ut
        #if epoch is None:
        #    # use best epoch if not specified
        #    # (WARNING: Make sure the weights in the model are model.best_weights)
        #    # they may be out of sync
        #    epoch = model.best_results['epoch']
        history_hashid = model.get_model_history_hashid()
        diagnostic_dpath = ut.ensuredir(ut.unixjoin(model.training_dpath, 'diagnostics'))
        #epoch_dpath = ut.ensuredir(ut.unixjoin(diagnostic_dpath, 'epoch_%r' % (epoch,)))
        epoch_dpath = ut.ensuredir(ut.unixjoin(diagnostic_dpath, history_hashid))
        return epoch_dpath

    def initialize_architecture(model):
        raise NotImplementedError('reimplement')

    def ensure_training_state(model, X_train, y_train):
        # Check to make sure data agrees with input
        # FIXME: This check should not be in this fuhnction
        input_layer = model.get_all_layers()[0]
        expected_item_shape = tuple(ut.list_take(input_layer.shape[1:], [1, 2, 0]))
        given_item_shape = X_train.shape[1:]
        assert given_item_shape == expected_item_shape, (
            'inconsistent item shape: ' +
            ('expected_item_shape = %r, ' % (expected_item_shape,)) +
            ('given_item_shape = %r' % (given_item_shape,))
        )

        if model.preproc_kw is None:
            # TODO: move this to data preprocessing, not model preprocessing
            model.preproc_kw = {}
            model.preproc_kw['center_mean'] = np.mean(X_train, axis=0)
            model.preproc_kw['center_std'] = 255.0
        if hasattr(model, 'initialize_encoder'):
            model.initialize_encoder(y_train)

    def reinit_weights(model, W=init.Orthogonal()):
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

    # --- io

    def has_saved_state(model):
        return ut.checkpath(model.get_model_state_fpath())

    def get_model_state_fpath(model, fpath=None, dpath=None, fname=None):
        if fpath is None:
            default_fname = 'model_state_arch_%s' % (model.get_architecture_hashid())
            fname = default_fname if fname is None else fname
            dpath = model.training_dpath if dpath is None else dpath
            model_state_fpath = join(dpath, fname)
        else:
            model_state_fpath = fpath
        return model_state_fpath

    def get_architecture_hashid(model):
        architecture_str = model.get_architecture_str()
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
            >>> model = BaseModel()
            >>> # make a dummy history
            >>> X_train, y_train = [1, 2, 3], [0, 0, 1]
            >>> model.start_new_era(X_train, y_train, 'dummy_alias_key')
            >>> model.record_epoch({'epoch': 1, 'valid_loss': .9, 'train_loss': .8})
            >>> model.record_epoch({'epoch': 2, 'valid_loss': .7, 'train_loss': .7})
            >>> model.start_new_era(X_train, y_train, 'dummy_alias_key')
            >>> model.record_epoch({'epoch': 3, 'valid_loss': .8, 'train_loss': .6})
            >>> model.record_epoch({'epoch': 4, 'valid_loss': .7, 'train_loss': .5})
            >>> model.record_epoch({'epoch': 5, 'valid_loss': .6, 'train_loss': .2})
            >>> # test the hashid
            >>> history_hashid = model.get_model_history_hashid()
            >>> result = str(history_hashid)
            >>> print(result)
            hist_eras2_epochs5_epdfcmhfkebiejte
        """
        era_history_hash = [ut.hashstr27(repr(era)) for era in  model.era_history]
        hashid = ut.hashstr27(str(era_history_hash))
        total_epochs = sum([len(era['epoch_list']) for era in model.era_history])
        total_eras = len(model.era_history)
        history_hashid = 'hist_eras%d_epochs%d_%s' % (total_eras, total_epochs, hashid)
        return history_hashid

    def checkpoint_save_model_state(model):
        checkpoint_dir = ut.ensuredir(ut.unixjoin(model.training_dpath, 'checkpoints'))
        history_hashid = model.get_model_history_hashid()
        dpath = ut.ensuredir(ut.unixjoin(checkpoint_dir, history_hashid))
        model.save_model_state(dpath=dpath)

    def save_model_state(model, **kwargs):
        current_weights = model.get_all_param_values()
        model_state = {
            'best_results': model.best_results,
            'preproc_kw':   model.preproc_kw,
            'best_weights': model.best_weights,
            'current_weights': current_weights,
            'input_shape':  model.input_shape,
            'output_dims':  model.output_dims,
            'era_history':  model.era_history,
        }
        model_state_fpath = model.get_model_state_fpath(**kwargs)
        print('saving model state to: %s' % (model_state_fpath,))
        with open(model_state_fpath, 'wb') as file_:
            pickle.dump(model_state, file_, protocol=pickle.HIGHEST_PROTOCOL)
        print('finished saving')

    def load_model_state(model, **kwargs):
        """
        import cPickle as pickle
        kwargs = {}
        """
        model_state_fpath = model.get_model_state_fpath(**kwargs)
        print('[model] loading model state from: %s' % (model_state_fpath,))
        with open(model_state_fpath, 'rb') as file_:
            model_state = pickle.load(file_)
        if model.__class__.__name__ != 'BaseModel':
            assert model_state['input_shape'][1:] == model.input_shape[1:], 'architecture disagreement'
            assert model_state['output_dims'] == model.output_dims, 'architecture disagreement'
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
            model.set_all_param_values(model.best_weights)

    def load_extern_weights(model, **kwargs):
        """ load weights from another model """
        model_state_fpath = model.get_model_state_fpath(**kwargs)
        print('[model] loading extern weights from: %s' % (model_state_fpath,))
        with open(model_state_fpath, 'rb') as file_:
            model_state = pickle.load(file_)
        if True or utils.VERBOSE_CNN:
            print('External Model State:')
            print(ut.dict_str(model_state, truncate=True))
        # check compatibility with this architecture
        assert model_state['input_shape'][1:] == model.input_shape[1:], 'architecture disagreement'
        assert model_state['output_dims'] == model.output_dims, 'architecture disagreement'
        # Just set the weights, no other training state variables
        model.set_all_param_values(model_state['best_weights'])
        # also need to make sure the same preprocessing is used
        # TODO make this a layer?
        model.preproc_kw = model_state['preproc_kw']
        model.era_history = model_state['era_history']

    def load_old_weights_kw(model, old_weights_fpath):
        print('[model] loading old model state from: %s' % (old_weights_fpath,))
        with open(old_weights_fpath, 'rb') as file_:
            oldkw = pickle.load(file_)
        # Model architecture and weight params
        data_shape  = oldkw['model_shape'][1:]
        input_shape = (None, data_shape[2], data_shape[0], data_shape[1])
        output_dims  = oldkw['output_dims']

        # Perform checks
        assert input_shape[1:] == model.input_shape[1:], 'architecture disagreement'
        assert output_dims == model.output_dims, 'architecture disagreement'

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
            'valid_loss'     : oldkw['best_valid_loss'],
        }

        # Set architecture weights
        weights_list = model.best_weights
        model.set_all_param_values(weights_list)
        #learning_state = {
        #    'weight_decay'   : oldkw['regularization'],
        #    'learning_rate'  : oldkw['learning_rate'],
        #    'momentum'       : oldkw['momentum'],
        #}
        #batch_size = oldkw['batch_size']

    # --- utility

    def historyfoohack(model, X_train, y_train, trainset):
        #x_hashid = ut.hashstr_arr(X_train, 'x', alphabet=ut.ALPHABET_27)
        y_hashid = ut.hashstr_arr(y_train, 'y', alphabet=ut.ALPHABET_27)
        #
        train_hashid =  trainset.alias_key + '_' + y_hashid
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

    def start_new_era(model, X_train, y_train, alias_key):
        """
        Used to denote a change in hyperparameters during training.
        """
        y_hashid = ut.hashstr_arr(y_train, 'y', alphabet=ut.ALPHABET_27)
        train_hashid =  alias_key + '_' + y_hashid
        era_info = {
            'train_hashid': train_hashid,
            'arch_hashid': model.get_architecture_hashid(),
            'valid_loss_list': [],
            'train_loss_list': [],
            'epoch_list': [],
            'learning_rate': [model.learning_rate],
            'learning_state': [model.learning_state],
        }
        print('starting new era')
        #if model.current_era is not None:
        model.current_era = era_info
        model.era_history.append(model.current_era)

    def print_state_str(model, **kwargs):
        print(model.get_state_str(**kwargs))

    def show_era_history(model, fnum=None, pnum=(1, 1, 1)):
        import plottool as pt

        fnum = pt.ensure_fnum(fnum)

        fig = pt.figure(fnum, pnum)
        colors = pt.distinct_colors(len(model.era_history))
        for index, era in enumerate(model.era_history):
            epochs = era['epoch_list']
            train_loss = era['train_loss_list']
            valid_loss = era['valid_loss_list']
            era_color = colors[index]
            if index == len(model.era_history) - 1:
                pt.plot(epochs, valid_loss, '-x', color=era_color, label='valid_loss')
                pt.plot(epochs, train_loss, '-o', color=era_color, label='train_loss')
            else:
                pt.plot(epochs, valid_loss, '-x', color=era_color)
                pt.plot(epochs, train_loss, '-o', color=era_color)

        #append_phantom_legend_label
        pt.set_xlabel('epoch')
        pt.set_ylabel('loss')

        pt.legend()

        pt.set_figtitle('Era History: ' + model.get_model_history_hashid())
        pt.dark_background()
        return fig

    def get_state_str(model, other_override_reprs={}):
        era_history_str = ut.list_str(
            [ut.dict_str(era, truncate=True, sorted_=True)
             for era in model.era_history], strvals=True)

        override_reprs = {
            'best_results': ut.dict_str(model.best_results),
            'best_weights': ut.truncate_str(str(model.best_weights)),
            'preproc_kw': 'None' if model.preproc_kw is None else ut.dict_str(model.preproc_kw, truncate=True),
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

    def draw_convolutional_layers(model, target=[0]):
        # DEPRICATE
        output_files = draw_net.show_convolutional_layers(model.output_layer, model.training_dpath, target=target)
        return output_files

    def show_model_layer_weights(model, **kwargs):
        # RENAME
        draw_net.show_model_layer_weights(model, **kwargs)

    def save_model_layer_weights(model, *args, **kwargs):
        # RENAME
        draw_net.save_model_layer_weights(model, *args, **kwargs)

    def draw_all_conv_layer_weights(model, fnum=None):
        import plottool as pt
        if fnum is None:
            fnum = pt.next_fnum()
        conv_layers = [layer_ for layer_ in model.get_all_layers() if hasattr(layer_, 'W') and layer_.name.startswith('C')]
        for index in range(len(conv_layers)):
            model.save_model_layer_weights(index, fnum=fnum + index)

    def draw_architecture(model):
        filename = 'tmp.png'
        draw_net.draw_to_file(model.get_all_layers(), filename)
        ut.startfile(filename)

    def print_layer_info(model):
        net_strs.print_layer_info(model.get_output_layer())

    def set_all_param_values(model, weights_list):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', '.*topo.*')
            lasagne.layers.set_all_param_values(model.output_layer, weights_list)

    def get_all_param_values(model):
        weights_list = lasagne.layers.get_all_param_values(model.output_layer)
        return weights_list

    def get_all_params(model, **tags):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', '.*topo.*')
            parameters = lasagne.layers.get_all_params(model.output_layer, **tags)
            return parameters

    def get_architecture_str(model, sep='_'):
        # TODO: allow for removal of layers without any parameters
        #if getattr(model, 'network_layers', None) is None:
        #    if getattr(model, 'output_layer', None) is None:
        #        return None
        #    else:
        #        network_layers = lasagne.layers.get_all_layers(model.output_layer)
        #else:
        #    network_layers = model.network_layers
        network_layers = model.get_all_layers()
        layer_str_list = [net_strs.make_layer_str(layer) for layer in network_layers]
        architecture_str = sep.join(layer_str_list)
        return architecture_str

    def print_architecture_str(model, sep='\n  '):
        architecture_str = model.get_architecture_str(sep=sep)
        if architecture_str is None:
            architecture_str = 'UNMANGAGED'
        print('\nArchitecture:' + sep + architecture_str)

    def get_all_layers(model):
        assert model.output_layer is not None
        network_layers = lasagne.layers.get_all_layers(model.output_layer)
        return network_layers

    def get_output_layer(model):
        if model.output_layer is not None:
            return model.output_layer
        else:
            assert model.network_layers is not None, 'need to initialize architecture first'
            output_layer = model.network_layers[-1]
            return output_layer

    def learning_rate_update(model, x):
        return x / 2.0

    def learning_rate_shock(model, x):
        return x * 2.0

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
            shared_learning_rate = theano.shared(utils.float32(rate))
            model.shared_state['learning_rate'] = shared_learning_rate
        else:
            shared_learning_rate.set_value(utils.float32(rate))

    @property
    def shared_learning_rate(model):
        return model.shared_state.get('learning_rate', None)

    def build_theano_funcs(model, **kwargs):
        print('[model] creating Theano primitives...')
        theano_funcs = batch.build_theano_funcs(model, **kwargs)
        return theano_funcs

    def build_loss_expressions(model, X_batch, y_batch):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', '.*topo.*')
            warnings.filterwarnings('ignore', '.*get_all_non_bias_params.*')
            warnings.filterwarnings('ignore', '.*layer.get_output.*')
            print('Building symbolic loss function')
            objective = lasagne.objectives.Objective(
                model.output_layer, loss_function=model.loss_function, aggregation='mean')
            loss = objective.get_loss(X_batch, target=y_batch)
            loss.name = 'loss'

            print('Building symbolic loss function (determenistic)')
            #loss_determ = objective.get_loss(X_batch, target=y_batch, deterministic=True)
            loss_determ = objective.get_loss(X_batch, target=y_batch, deterministic=True, aggregation='mean')
            loss_determ.name = 'loss_determ'

            # Regularize
            # TODO: L2 should be one of many available options for regularization
            L2 = lasagne.regularization.l2(model.output_layer)
            weight_decay = model.learning_state['weight_decay']
            loss_regularized = loss + L2 * weight_decay
            loss_regularized.name = 'loss_regularized'
            return loss, loss_determ, loss_regularized

    def build_unlabeled_output_expressions(model, network_output):
        # override in more specific subclasses
        return []

    def build_labeled_output_expressions(model, network_output, y_batch):
        # override in more specific subclasses
        return []


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


class _PretrainedLayerInitializer(init.Initializer):
    def __init__(self, pretrained_layer):
        self.pretrained_layer = pretrained_layer

    def sample(self, shape):
        fanout, fanin, height, width = shape
        fanout_, fanin_, height_, width_ = self.pretrained_layer.shape
        assert fanout <= fanout_, 'Cannot cast weights to a larger fan-out dimension'
        assert fanin  <= fanin_,  'Cannot cast weights to a larger fan-in dimension'
        assert height == height_, 'The height must be identical between the layer and weights'
        assert width  == width_,  'The width must be identical between the layer and weights'
        return lasagne.utils.floatX(self.pretrained_layer[:fanout, :fanin, :, :])


class PretrainedNetwork(object):
    """
    Intialize weights from a specified (Caffe) pretrained network layers

    Args:
        layer (int) : int

    CommandLine:
        python -m ibeis_cnn.models --test-PretrainedNetwork:0
        python -m ibeis_cnn.models --test-PretrainedNetwork:1

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
        num_filters = self.get_layer_num_filters(layer_index)
        filter_size = self.get_layer_filter_size(layer_index)
        Layer = functools.partial(
            Conv2DLayer, num_filters=num_filters,
            filter_size=filter_size, W=W, name=name, **kwargs)
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
