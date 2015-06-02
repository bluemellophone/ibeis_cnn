# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
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
    def __init__(self, output_dims=None, input_shape=None, batch_size=None, training_dpath='.'):
        #self.network_layers = None  # We really don't need to save all of these
        self.output_layer = None
        self.output_dims = None
        self.input_shape = None
        self.batch_size = None
        # bad name, says that this network will take
        # 2*N images in a batch and N labels that map to
        # two images a piece
        self.data_per_label = 1
        self.training_dpath = training_dpath  # TODO
        # Training state
        self.preproc_kw   = None
        self.best_results = None
        self.best_weights = None

    # --- initialization steps

    def initialize_architecture(self):
        raise NotImplementedError('reimlement')

    # --- utility

    def draw_convolutional_layers(self, target=[0]):
        from ibeis_cnn import draw_net
        output_files = draw_net.show_convolutional_layers(self.output_layer, self.training_dpath, target=target)
        return output_files

    def draw_architecture(self):
        from ibeis_cnn import draw_net
        filename = 'tmp.png'
        draw_net.draw_to_file(self.get_network_layers(), filename)
        ut.startfile(filename)

    def get_architecture_hashid(self):
        architecture_str = self.get_architecture_str()
        hashid = ut.hashstr(architecture_str, alphabet=ut.ALPHABET_16, hashlen=16)
        return hashid

    def print_layer_info(self):
        net_strs.print_layer_info(self.get_output_layer())

    def load_old_weights_kw(self, old_weights_fpath):
        with open(old_weights_fpath, 'rb') as pfile:
            oldkw = pickle.load(pfile)
        # Model architecture and weight params
        data_shape  = oldkw['model_shape'][1:]
        input_shape = (None, data_shape[2], data_shape[0], data_shape[1])
        output_dims  = oldkw['output_dims']

        # Perform checks
        assert input_shape[1:] == self.input_shape[1:], 'architecture disagreement'
        assert output_dims == self.output_dims, 'architecture disagreement'

        # Set class attributes
        self.best_weights = oldkw['best_weights']

        self.preproc_kw = {
            'center_mean' : oldkw['center_mean'],
            'center_std'  : oldkw['center_std'],
        }
        self.best_results = {
            'epoch'          : oldkw['best_epoch'],
            'test_accuracy'  : oldkw['best_test_accuracy'],
            'train_loss'     : oldkw['best_train_loss'],
            'valid_accuracy' : oldkw['best_valid_accuracy'],
            'valid_loss'     : oldkw['best_valid_loss'],
            'valid_loss'     : oldkw['best_valid_loss'],
        }

        # Set architecture weights
        weights_list = self.best_weights
        self.set_all_param_values(weights_list)
        #training_state = {
        #    'regularization' : oldkw['best_valid_loss'],
        #    'learning_rate'  : oldkw['learning_rate'],
        #    'momentum'       : oldkw['momentum'],
        #}
        #batch_size = oldkw['batch_size']

    def has_saved_state(self):
        return ut.checkpath(self.get_model_state_fpath())

    def save_model_state(self, **kwargs):
        model_state = {
            'best_results': self.best_results,
            'preproc_kw':   self.preproc_kw,
            'best_weights': self.best_weights,
            'input_shape':  self.input_shape,
            'output_dims':  self.output_dims,
        }
        model_state_fpath = self.get_model_state_fpath(**kwargs)
        print('saving model state to: %s' % (model_state_fpath,))
        with open(model_state_fpath, 'wb') as file_:
            pickle.dump(model_state, file_, protocol=pickle.HIGHEST_PROTOCOL)

    def load_model_state(self, **kwargs):
        """
        kwargs = {}
        """
        model_state_fpath = self.get_model_state_fpath(**kwargs)
        print('loading model state from: %s' % (model_state_fpath,))
        with open(model_state_fpath, 'rb') as file_:
            model_state = pickle.load(file_)
        assert model_state['input_shape'][1:] == self.input_shape[1:], 'architecture disagreement'
        assert model_state['output_dims'] == self.output_dims, 'architecture disagreement'
        self.best_results = model_state['best_results']
        self.preproc_kw   = model_state['preproc_kw']
        self.best_weights = model_state['best_weights']
        self.input_shape  = model_state['input_shape']
        self.output_dims  = model_state['output_dims']
        self.set_all_param_values(self.best_weights)

    def load_extern_weights(self, **kwargs):
        """ load weights from another model """
        model_state_fpath = self.get_model_state_fpath(**kwargs)
        print('loading extern weights from: %s' % (model_state_fpath,))
        with open(model_state_fpath, 'rb') as file_:
            model_state = pickle.load(file_)
        assert model_state['input_shape'][1:] == self.input_shape[1:], 'architecture disagreement'
        assert model_state['output_dims'] == self.output_dims, 'architecture disagreement'
        # Just set the weights, no other training state variables
        self.set_all_param_values(model_state['best_weights'])

    def get_model_state_fpath(self, fpath=None, dpath=None, fname=None):
        if fpath is None:
            fname = 'model_state.pkl' if fname is None else fname
            dpath = self.training_dpath if dpath is None else dpath
            model_state_fpath = join(dpath, fname)
        else:
            model_state_fpath = fpath
        return model_state_fpath

    def set_all_param_values(self, weights_list):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', '.*topo.*')
            lasagne.layers.set_all_param_values(self.output_layer, weights_list)

    def get_architecture_str(self, sep='_'):
        # TODO: allow for removal of layers without any parameters
        #if getattr(self, 'network_layers', None) is None:
        #    if getattr(self, 'output_layer', None) is None:
        #        return None
        #    else:
        #        network_layers = lasagne.layers.get_all_layers(self.output_layer)
        #else:
        #    network_layers = self.network_layers
        network_layers = self.get_network_layers()
        layer_str_list = [net_strs.make_layer_str(layer) for layer in network_layers]
        architecture_str = sep.join(layer_str_list)
        return architecture_str

    def print_architecture_str(self, sep='\n  '):
        architecture_str = self.get_architecture_str(sep=sep)
        if architecture_str is None:
            architecture_str = 'UNMANGAGED'
        print('\nArchitecture:' + sep + architecture_str)

    def get_network_layers(self):
        assert self.output_layer is not None
        network_layers = lasagne.layers.get_all_layers(self.output_layer)
        return network_layers

    def get_output_layer(self):
        if self.output_layer is not None:
            return self.output_layer
        else:
            assert self.network_layers is not None, 'need to initialize architecture first'
            output_layer = self.network_layers[-1]
            return output_layer

    def learning_rate_update(model, x):
        return x / 2.0

    def learning_rate_shock(model, x):
        return x * 2.0


class AbstractCategoricalModel(BaseModel):
    """ base model for catagory classifiers """

    def __init__(self):
        super(AbstractCategoricalModel, self).__init__()
        self.encoder = None

    def initialize_encoder(self, labels):
        print('[model] encoding labels')
        self.encoder = sklearn.preprocessing.LabelEncoder()
        self.encoder.fit(labels)
        self.output_dims = len(list(np.unique(labels)))
        print('[model] self.output_dims = %r' % (self.output_dims,))

    def loss_function(model, output, truth):
        return T.nnet.categorical_crossentropy(output, truth)
