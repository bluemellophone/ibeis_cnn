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
import sklearn.preprocessing
import utool as ut
from os.path import join
import warnings
import cPickle as pickle
ut.noinject('ibeis_cnn.abstract_models')


Conv2DLayer = custom_layers.Conv2DLayer
MaxPool2DLayer = custom_layers.MaxPool2DLayer


def evaluate_layer_list(network_layers_def, verbose=utils.VERBOSE_CNN):
    """ compiles a sequence of partial functions into a network """
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


class BaseModel(object):
    """
    Abstract model providing functionality for all other models to derive from
    """
    def __init__(model, output_dims=None, input_shape=None, batch_size=None, training_dpath='.'):
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
        # Training state
        model.requested_headers = ['epoch', 'train_loss', 'valid_loss', 'trainval_rat', 'duration']
        model.preproc_kw   = None
        model.best_results = {
            'epoch': -1,
            'train_loss':     np.inf,
            'valid_loss':     np.inf,
        }
        model.best_weights = None
        model.learning_state = {
            'momentum': .9,
            #'learning_rate': .001,
            'weight_decay': .0005,
        }
        # Theano shared state
        model.shared_state = {
            'learning_rate': None,
        }

    # --- initialization steps

    def initialize_architecture(model):
        raise NotImplementedError('reimlement')

    def ensure_training_state(model, X_train, y_train):
        if model.best_results is None:
            model.best_results = {
                'valid_accuracy': 0.0,
                'test_accuracy':  0.0,
                'train_loss':     np.inf,
                'valid_loss':     np.inf,
                'valid_accuracy': 0.0,
                'test_accuracy':  0.0,
            }
        if model.preproc_kw is None:
            # TODO: move this to data preprocessing, not model preprocessing
            model.preproc_kw['center_mean'] = np.mean(X_train, axis=0)
            model.preproc_kw['center_mean'] = 255.0

    # --- utility

    def draw_convolutional_layers(model, target=[0]):
        from ibeis_cnn import draw_net
        output_files = draw_net.show_convolutional_layers(model.output_layer, model.training_dpath, target=target)
        return output_files

    def draw_architecture(model):
        from ibeis_cnn import draw_net
        filename = 'tmp.png'
        draw_net.draw_to_file(model.get_network_layers(), filename)
        ut.startfile(filename)

    def get_architecture_hashid(model):
        architecture_str = model.get_architecture_str()
        hashid = ut.hashstr(architecture_str, alphabet=ut.ALPHABET_16, hashlen=16)
        return hashid

    def print_layer_info(model):
        net_strs.print_layer_info(model.get_output_layer())

    def load_old_weights_kw(model, old_weights_fpath):
        with open(old_weights_fpath, 'rb') as pfile:
            oldkw = pickle.load(pfile)
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

    def has_saved_state(model):
        return ut.checkpath(model.get_model_state_fpath())

    def save_model_state(model, **kwargs):
        model_state = {
            'best_results': model.best_results,
            'preproc_kw':   model.preproc_kw,
            'best_weights': model.best_weights,
            'input_shape':  model.input_shape,
            'output_dims':  model.output_dims,
        }
        model_state_fpath = model.get_model_state_fpath(**kwargs)
        print('saving model state to: %s' % (model_state_fpath,))
        with open(model_state_fpath, 'wb') as file_:
            pickle.dump(model_state, file_, protocol=pickle.HIGHEST_PROTOCOL)

    def load_model_state(model, **kwargs):
        """
        kwargs = {}
        """
        model_state_fpath = model.get_model_state_fpath(**kwargs)
        print('loading model state from: %s' % (model_state_fpath,))
        with open(model_state_fpath, 'rb') as file_:
            model_state = pickle.load(file_)
        assert model_state['input_shape'][1:] == model.input_shape[1:], 'architecture disagreement'
        assert model_state['output_dims'] == model.output_dims, 'architecture disagreement'
        model.best_results = model_state['best_results']
        model.preproc_kw   = model_state['preproc_kw']
        model.best_weights = model_state['best_weights']
        model.input_shape  = model_state['input_shape']
        model.output_dims  = model_state['output_dims']
        model.set_all_param_values(model.best_weights)

    def load_extern_weights(model, **kwargs):
        """ load weights from another model """
        model_state_fpath = model.get_model_state_fpath(**kwargs)
        print('loading extern weights from: %s' % (model_state_fpath,))
        with open(model_state_fpath, 'rb') as file_:
            model_state = pickle.load(file_)
        assert model_state['input_shape'][1:] == model.input_shape[1:], 'architecture disagreement'
        assert model_state['output_dims'] == model.output_dims, 'architecture disagreement'
        # Just set the weights, no other training state variables
        model.set_all_param_values(model_state['best_weights'])

    def get_model_state_fpath(model, fpath=None, dpath=None, fname=None):
        if fpath is None:
            fname = 'model_state.pkl' if fname is None else fname
            dpath = model.training_dpath if dpath is None else dpath
            model_state_fpath = join(dpath, fname)
        else:
            model_state_fpath = fpath
        return model_state_fpath

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
            parameters = lasagne.layers.get_all_params(model.output_layer, trainable=True)
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
        network_layers = model.get_network_layers()
        layer_str_list = [net_strs.make_layer_str(layer) for layer in network_layers]
        architecture_str = sep.join(layer_str_list)
        return architecture_str

    def print_architecture_str(model, sep='\n  '):
        architecture_str = model.get_architecture_str(sep=sep)
        if architecture_str is None:
            architecture_str = 'UNMANGAGED'
        print('\nArchitecture:' + sep + architecture_str)

    def get_network_layers(model):
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
        if model.learning_rate is None:
            model.learning_rate = .001
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


class AbstractCategoricalModel(BaseModel):
    """ base model for catagory classifiers """

    def __init__(model, **kwargs):
        super(AbstractCategoricalModel, model).__init__(**kwargs)
        model.encoder = None

    def initialize_encoder(model, labels):
        print('[model] encoding labels')
        model.encoder = sklearn.preprocessing.LabelEncoder()
        model.encoder.fit(labels)
        model.output_dims = len(list(np.unique(labels)))
        print('[model] model.output_dims = %r' % (model.output_dims,))

    def loss_function(model, output, truth):
        return T.nnet.categorical_crossentropy(output, truth)
