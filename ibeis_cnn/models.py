"""
file model.py
allows the definition of different models to be trained
for initialization: Lasagne/lasagne/init.py
for nonlinearities: Lasagne/lasagne/nonlinearities.py
for layers: Lasagne/lasagne/layers/
"""
from __future__ import absolute_import, division, print_function
from lasagne import layers
from lasagne import nonlinearities
from lasagne import init
from lasagne.utils import floatX
from ibeis_cnn import lasange_ext
import random
import functools
import utool as ut
import six
import theano.tensor as T
import numpy as np
from ibeis_cnn import utils
from ibeis_cnn import custom_layers  # NOQA
print, rrr, profile = ut.inject2(__name__, '[ibeis_cnn.train]')
# from os.path import join
# import cPickle as pickle

FORCE_CPU = False  # ut.get_argflag('--force-cpu')
try:
    if FORCE_CPU:
        raise ImportError('GPU is forced off')
    # use cuda_convnet for a speed improvement
    # will not be available without a GPU
    import lasagne.layers.cuda_convnet as convnet
    Conv2DLayer = convnet.Conv2DCCLayer
    MaxPool2DLayer = convnet.MaxPool2DCCLayer
    USING_GPU = True
except ImportError as ex:
    ut.printex(ex, 'WARNING: GPU seems unavailable')
    Conv2DLayer = layers.Conv2DLayer
    MaxPool2DLayer = layers.MaxPool2DLayer
    USING_GPU = False


def save_pretrained_weights_slice(pretrained_weights, weights_path, slice_=slice(None)):
    """
    Used to save a slice of pretrained weights. The doctest will publish a new set of weights

    CommandLine:
        python -m ibeis_cnn.models --test-save_pretrained_weights_slice --net='vggnet_full' --slice='slice(0,6)'
        python -m ibeis_cnn.models --test-save_pretrained_weights_slice --net='vggnet_full' --slice='slice(0,30)'
        python -m ibeis_cnn.models --test-save_pretrained_weights_slice --net='caffenet_full' --slice='slice(0,6)'
        python -m ibeis_cnn.models --test-save_pretrained_weights_slice --net='caffenet_full' --slice='slice(0,?)'

    Example:
        >>> # DISABLE_DOCTEST
        >>> # Build a new subset of an existing model
        >>> from ibeis_cnn.models import *  # NOQA
        >>> from ibeis_cnn._plugin_grabmodels import ensure_model
        >>> # Get base model weights
        >>> modelname = ut.get_argval('--net', type_=str, default='vggnet_full')
        >>> weights_path = ensure_model(modelname)
        >>> pretrained_weights = ut.load_cPkl(weights_path)
        >>> # Get the slice you want
        >>> slice_str = ut.get_argval('--slice', type_=str, default='slice(0, 6)')
        >>> slice_ = eval(slice_str, globals(), locals())
        >>> # execute function
        >>> sliced_weights_path = save_pretrained_weights_slice(pretrained_weights, weights_path, slice_)
        >>> # PUT YOUR PUBLISH PATH HERE
        >>> publish_fpath = ut.truepath('~/Dropbox/IBEIS')
        >>> ut.copy(sliced_weights_path, publish_fpath)
    """
    # slice and save
    suffix = '.slice_%r_%r_%r' % (slice_.start, slice_.stop, slice_.step)
    sliced_weights_path = ut.augpath(weights_path, suffix)
    sliced_pretrained_weights = pretrained_weights[slice_]
    ut.save_cPkl(sliced_weights_path, sliced_pretrained_weights)
    # print info
    print_pretrained_weights(pretrained_weights, weights_path)
    print_pretrained_weights(sliced_pretrained_weights, sliced_weights_path)
    return sliced_weights_path


def print_pretrained_weights(pretrained_weights, lbl=''):
    r"""
    Args:
        pretrained_weights (list of ndarrays): represents layer weights
        lbl (str): label
    """
    print('Initialization network: %r' % (lbl))
    print('Total memory: %s' % (ut.get_object_size_str(pretrained_weights)))
    for index, layer_ in enumerate(pretrained_weights):
        print(' layer {:2}: shape={:<18}, memory={}'.format(index, layer_.shape, ut.get_object_size_str(layer_)))


def make_layer_str(layer):
    r"""
    Args:
        layer (lasange.Layer): a network layer
    """
    # filter_size is a scalar in Conv2DLayers because non-uniform shapes is not supported
    common_attrs = ['shape', 'num_filters', 'num_units', 'ds', 'filter_shape', 'stride', 'strides', 'p', 'axis']
    ignore_attrs = ['nonlinearity', 'b', 'W']
    #func_attrs = ['get_output_shape']
    func_attrs = []
    layer_attrs_dict = {
        'DropoutLayer': ['p'],
        'InputLayer': ['shape'],
        'Conv2DCCLayer': ['num_filters', 'filter_size', 'stride'],
        'MaxPool2DCCLayer': ['stride', 'pool_size'],
        'DenseLayer': ['num_units'],
        'L2NormalizeLayer': ['axis'],
    }
    layer_type = '{0}'.format(layer.__class__.__name__)
    request_attrs = sorted(list(set(layer_attrs_dict.get(layer_type, []) + common_attrs)))
    isvalid_list = [hasattr(layer, attr) for attr in request_attrs]
    attr_key_list = ut.list_compress(request_attrs, isvalid_list)

    DEBUG = False
    if DEBUG:
        #if layer_type == 'Conv2DCCLayer':
        #    ut.embed()
        print('---')
        print(' * ' + layer_type)
        print(' * does not have keys: %r' % (ut.filterfalse_items(request_attrs, isvalid_list),))
        print(' * missing keys: %r' % ((set(layer.__dict__.keys()) - set(ignore_attrs)) - set(attr_key_list),))
        print(' * has keys: %r' % (attr_key_list,))

    attr_val_list = [getattr(layer, attr) for attr in attr_key_list]
    attr_str_list = ['%s=%r' % item for item in zip(attr_key_list, attr_val_list)]

    for func_attr in func_attrs:
        if hasattr(layer, func_attr):
            attr_str_list.append('%s=%r' % (func_attr, getattr(layer, func_attr).__call__()))

    layer_name = getattr(layer, 'name', None)
    if layer_name is not None:
        attr_str_list = ['name=%s' % (layer_name,)] + attr_str_list

    if hasattr(layer, 'nonlinearity'):
        try:
            nonlinearity = layer.nonlinearity.__name__
        except AttributeError:
            nonlinearity = layer.nonlinearity.__class__.__name__
        attr_str_list.append('nonlinearity={0}'.format(nonlinearity))
    attr_str = ','.join(attr_str_list)

    layer_str = layer_type + '(' + attr_str + ')'
    return layer_str


def evaluate_layer_list(network_layers_def):
    """ compiles a sequence of partial functions into a network """
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', '.*The uniform initializer no longer uses Glorot.*')
        network_layers = []
        layer_fn_iter = iter(network_layers_def)
        prev_layer = six.next(layer_fn_iter)()
        network_layers.append(prev_layer)
        for layer_fn in layer_fn_iter:
            prev_layer = layer_fn(prev_layer)
            network_layers.append(prev_layer)
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
        return floatX(self.pretrained_layer[:fanout, :fanin, :, :])


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
            raise IOError('The specified model was not found: %r' % (weights_path, ))
        if show_network:
            print_pretrained_weights(self.pretrained_weights, weights_path)

    def get_num_layers(self):
        return len(self.pretrained_weights)

    def get_layer_num_filters(self, layer_index):
        assert layer_index <= len(self.pretrained_weights), 'Trying to specify a layer that does not exist'
        fanout, fanin, height, width = self.pretrained_weights[layer_index].shape
        return fanout

    def get_layer_filter_size(self, layer_index):
        assert layer_index <= len(self.pretrained_weights), 'Trying to specify a layer that does not exist'
        fanout, fanin, height, width = self.pretrained_weights[layer_index].shape
        return (height, width)

    def get_conv2d_layer(self, layer_index, name=None, trainable=True, **kwargs):
        """ Assumes requested layer is convolutional

        Returns:
            lasange.layers.Layer: Layer
        """
        if name is None:
            name = '%s_layer%r' % (self.modelkey, layer_index)
        W = self.get_pretrained_layer(layer_index)
        num_filters = self.get_layer_num_filters(layer_index)
        filter_size = self.get_layer_filter_size(layer_index)
        Layer = functools.partial(Conv2DLayer, num_filters=num_filters,
                                  filter_size=filter_size, W=W, name=name,
                                  trainable=trainable, **kwargs)
        return Layer

    def get_pretrained_layer(self, layer_index, rand=False):
        assert layer_index <= len(self.pretrained_weights), 'Trying to specify a layer that does not exist'
        pretrained_layer = self.pretrained_weights[layer_index]
        weights_initializer = _PretrainedLayerInitializer(pretrained_layer)
        if rand:
            np.random.shuffle(weights_initializer)
        return weights_initializer


class BaseModel(object):
    def __init__(self):
        self.network_layers = None
        # bad name, says that this network will take
        # 2*N images in a batch and N labels that map to
        # two images a piece
        self.data_per_label = 1
        self.net_dir = '.'

    def draw_architecture(self):
        from ibeis_cnn import draw_net
        filename = 'tmp.png'
        draw_net.draw_to_file(self.network_layers, filename)
        ut.startfile(filename)

    def get_architecture_hashid(self):
        architecture_str = self.get_architecture_str()
        hashid = ut.hashstr(architecture_str, alphabet=ut.ALPHABET_16, hashlen=16)
        return hashid

    def print_layer_info(self):
        utils.print_layer_info(self.get_output_layer())

    def get_architecture_str(self, sep='_'):
        layer_str_list = [make_layer_str(layer) for layer in self.network_layers]
        architecture_str = sep.join(layer_str_list)
        return architecture_str

    def print_architecture_str(self, sep='\n  '):
        architecture_str = self.get_architecture_str(sep=sep)
        print('\nArchitecture:' + sep + architecture_str)

    def get_output_layer(self):
        assert self.network_layers is not None, 'need to initialize architecture first'
        output_layer = self.network_layers[-1]
        return output_layer

    def learning_rate_update(model, x):
        return x / 2.0

    def learning_rate_shock(model, x):
        return x * 2.0


@six.add_metaclass(ut.ReloadingMetaclass)
class DummyModel(BaseModel):
    def __init__(model, autoinit=False, batch_size=8, input_shape=(None, 1, 4, 4)):
        super(DummyModel, model).__init__()
        model.network_layers = None
        model.data_per_label = 1
        model.input_shape = input_shape
        model.output_dims = 5
        model.learning_rate = .001
        model.batch_size = 8
        model.momentum = .9
        if autoinit:
            model.initialize_architecture()

    def make_random_testdata(model, num=1000, seed=0):
        np.random.seed(seed)
        X_unshared = np.random.rand(num, * model.input_shape[1:]).astype(np.float32)
        y_unshared = (np.random.rand(num) * model.output_dims).astype(np.int32)
        if ut.VERBOSE:
            print('made random testdata')
            print('size(X_unshared) = %r' % (ut.get_object_size_str(X_unshared),))
            print('size(y_unshared) = %r' % (ut.get_object_size_str(y_unshared),))
        return X_unshared, y_unshared

    def loss_function(model, output, truth):
        return T.nnet.categorical_crossentropy(output, truth)

    def make_prediction_expr(model, newtork_output):
        prediction = T.argmax(newtork_output, axis=1)
        prediction.name = 'prediction'
        return prediction

    def make_accuracy_expr(model, prediction, y_batch):
        accuracy = T.mean(T.eq(prediction, y_batch))
        accuracy.name = 'accuracy'
        return accuracy

    #def get_loss_function(model):
    #    return T.nnet.categorical_crossentropy

    def initialize_architecture(model, verbose=True):
        input_shape = model.input_shape
        _P = functools.partial
        network_layers_def = [
            _P(layers.InputLayer, shape=input_shape),
            _P(Conv2DLayer, num_filters=16, filter_size=(3, 3)),
            _P(Conv2DLayer, num_filters=16, filter_size=(2, 2)),
            _P(layers.DenseLayer, num_units=8),
            _P(layers.DenseLayer, num_units=model.output_dims,
               nonlinearity=nonlinearities.softmax,
               W=init.Orthogonal(),),
        ]
        network_layers = evaluate_layer_list(network_layers_def)
        model.network_layers = network_layers
        output_layer = model.network_layers[-1]
        if verbose:
            print('initialize_architecture')
        if ut.VERBOSE:
            model.print_architecture_str()
            model.print_layer_info()
        return output_layer

    #def initialize_symbolic_inputs(model):
    #    model.symbolic_index = T.lscalar(name='index')
    #    model.symbolic_X = T.tensor4(name='X')
    #    model.symbolic_y = T.ivector(name='y')

    #def initialize_symbolic_updates(model):
    #    pass

    #def initialize_symbolic_outputs(model):
    #    pass


class SiameseCenterSurroundModel(BaseModel):
    """
    Model for individual identification

    TODO:
        RBM / EBM  - http://deeplearning.net/tutorial/rbm.html
    """
    def __init__(model, autoinit=False, batch_size=128, input_shape=(None, 3, 64, 64)):
        super(SiameseCenterSurroundModel, model).__init__()
        model.network_layers = None
        model.input_shape = input_shape
        model.batch_size = batch_size
        model.output_dims = 1
        # bad name, says that this network will take
        # 2*N images in a batch and N labels that map to
        # two images a piece
        model.data_per_label = 2
        model.needs_padding = False
        if autoinit:
            model.initialize_architecture()

    def build_model(model, batch_size, input_width, input_height,
                    input_channels, output_dims, verbose=True):
        r"""
        CommandLine:
            python -m ibeis_cnn.models --test-SiameseCenterSurroundModel.build_model
            python -m ibeis_cnn.models --test-SiameseCenterSurroundModel.build_model --verbcnn
            python -m ibeis_cnn.train --test-train_patchmatch_pz --vtd --max-examples=5 --batch_size=128 --learning_rate .0000001 --verbcnn

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis_cnn.models import *  # NOQA
            >>> # build test data
            >>> model = SiameseCenterSurroundModel()
            >>> batch_size = 128
            >>> input_width, input_height, input_channels = 64, 64, 3
            >>> output_dims = None
            >>> verbose = True
            >>> # execute function
            >>> output_layer = model.build_model(batch_size, input_width, input_height, input_channels, output_dims, verbose)
            >>> print('\n---- Arch Str')
            >>> model.print_architecture_str(sep='\n')
            >>> print('\n---- Layer Info')
            >>> model.print_layer_info()
            >>> print('\n---- HashID')
            >>> print('hashid=%r' % (model.get_architecture_hashid()),)
            >>> print('----')
            >>> # verify results
            >>> result = str(output_layer)
            >>> print(result)
        """
        #print('build model may override settings')
        input_shape = (batch_size, input_channels, input_width, input_height)
        model.input_shape = input_shape
        model.batch_size = batch_size
        model.output_dims = 1
        model.initialize_architecture(verbose=verbose)
        output_layer = model.get_output_layer()
        return output_layer

    def initialize_architecture(model, verbose=True):
        # TODO: remove output dims
        _P = functools.partial
        (_, input_channels, input_width, input_height) = model.input_shape
        if verbose:
            print('[model] Initialize siamese model architecture')
            print('[model]   * batch_size     = %r' % (model.batch_size,))
            print('[model]   * input_width    = %r' % (input_width,))
            print('[model]   * input_height   = %r' % (input_height,))
            print('[model]   * input_channels = %r' % (input_channels,))
            print('[model]   * output_dims    = %r' % (model.output_dims,))

        leaky = dict(nonlinearity=nonlinearities.LeakyRectify(leakiness=(1. / 10.)))
        leaky_orthog = dict(W=init.Orthogonal(), **leaky)

        #vggnet = PretrainedNetwork('vggnet')
        caffenet = PretrainedNetwork('caffenet')

        #def conv_pool(num_filters, filter_size, pool_size, stride):
        #    conv_ = _P(Conv2DLayer, num_filters=num_filters, filter_size=filter_size, **leaky_orthog)
        #    maxpool_ =  _P(MaxPool2DLayer, pool_size=pool_size, stride=stride)
        #    return [conv_, maxpool_]

        network_layers_def = (
            [
                _P(layers.InputLayer, shape=model.input_shape),
                _P(custom_layers.CenterSurroundLayer),

                #layers.GaussianNoiseLayer,
                caffenet.get_conv2d_layer(0, trainable=False, **leaky),
                _P(MaxPool2DLayer, pool_size=(3, 3), stride=(1, 1), name='M0'),
                _P(Conv2DLayer, num_filters=16, filter_size=(6, 6), name='C1', **leaky_orthog),
            ] +
            #conv_pool(num_filters=16, filter_size=(6, 6), pool_size=(2, 2), stride=(2, 2)) +
            #conv_pool(num_filters=32, filter_size=(3, 3), pool_size=(2, 2), stride=(2, 2)) +
            [

                _P(layers.DropoutLayer, p=0.5),
                #_P(custom_layers.L2NormalizeLayer, axis=2),
                _P(layers.DenseLayer, num_units=64, name='D1', **leaky_orthog),
                _P(layers.DropoutLayer, p=0.5),
                #_P(layers.DenseLayer, num_units=512, **leaky_orthog),
                #_P(layers.DropoutLayer, p=0.5),
                #_P(layers.merge.ConcatLayer),
                _P(custom_layers.SiameseConcatLayer, axis=1, data_per_label=4),  # 4 when CenterSurroundIsOn
                #_P(custom_layers.SiameseConcatLayer, data_per_label=2),
                _P(layers.DenseLayer, num_units=64, name='D2',  **leaky_orthog),
                _P(layers.DropoutLayer, p=0.5),
                _P(layers.DenseLayer, num_units=1, name='D3', **leaky_orthog),
            ]
        )

        # connect and record layers
        network_layers = evaluate_layer_list(network_layers_def)
        model.network_layers = network_layers
        output_layer = model.network_layers[-1]
        return output_layer

    def loss_function(model, E, Y, T=T, verbose=True):
        """

        CommandLine:
            python -m ibeis_cnn.models --test-loss_function

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis_cnn.models import *  # NOQA
            >>> from ibeis_cnn import train
            >>> from ibeis_cnn import batch_processing as batch
            >>> data, labels = train.testdata_patchmatch()
            >>> model = SiameseCenterSurroundModel(autoinit=True, input_shape=(128, 3, 41, 41))
            >>> theano_forward = batch.create_unbuffered_network_output_func(model)
            >>> batch_size = model.batch_size
            >>> Xb, yb = data[0:batch_size], labels[0:batch_size // 2]
            >>> network_output = theano_forward(Xb)[0]
            >>> E = network_output
            >>> Y = yb
            >>> T = np
            >>> # execute function
            >>> verbose = True
            >>> avg_loss = model.loss_function(E, Y, T=T)
            >>> result = str(avg_loss)
            >>> print(result)
        """
        if verbose:
            print('[model] Build siamese loss function')
        # Contrastive loss function
        Q = 2
        E = E.flatten()
        genuine_loss = (1 - Y) * (2 / Q) * (E ** 2)
        imposter_loss = (Y) * 2 * Q * T.exp((-2.77 * E) / Q)
        loss = genuine_loss + imposter_loss
        avg_loss = T.mean(loss)
        loss.name = 'loss'
        avg_loss.name = 'avg_loss'
        return avg_loss

        #avg_loss = lasange_ext.siamese_loss(G, Y_padded, data_per_label=2)
        #return avg_loss


class SiameseModel(BaseModel):
    """
    Model for individual identification
    """
    def __init__(model):
        super(SiameseModel, model).__init__()
        model.network_layers = None
        # bad name, says that this network will take
        # 2*N images in a batch and N labels that map to
        # two images a piece
        model.data_per_label = 2
        model.needs_padding = True

    def build_model(model, batch_size, input_width, input_height,
                    input_channels, output_dims, verbose=True):
        r"""
        CommandLine:
            python -m ibeis_cnn.models --test-SiameseModel.build_model

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis_cnn.models import *  # NOQA
            >>> # build test data
            >>> model = SiameseModel()
            >>> batch_size = 128
            >>> input_width, input_height, input_channels  = 64, 64, 3
            >>> output_dims = None
            >>> verbose = True
            >>> output_layer = model.build_model(batch_size, input_width, input_height, input_channels, output_dims, verbose)
            >>> print('----')
            >>> model.print_architecture_str(sep='\n')
            >>> print('hashid=%r' % (model.get_architecture_hashid()),)
            >>> print('----')
            >>> result = str(output_layer)
            >>> print(result)
        """
        input_shape = (None, input_channels, input_width, input_height)
        model.input_shape = input_shape
        model.batch_size = batch_size
        model.output_dims = 256
        model.initialize_architecture(verbose=verbose)
        output_layer = model.get_output_layer()
        return output_layer

    def initialize_architecture(model, verbose=True):
        # TODO: remove output dims
        _P = functools.partial
        (_, input_channels, input_width, input_height) = model.input_shape
        model.output_dims
        if verbose:
            print('[model] Initialize siamese model architecture')
            print('[model]   * batch_size     = %r' % (model.batch_size,))
            print('[model]   * input_width    = %r' % (input_width,))
            print('[model]   * input_height   = %r' % (input_height,))
            print('[model]   * input_channels = %r' % (input_channels,))
            print('[model]   * output_dims    = %r' % (model.output_dims,))

        #num_filters=32,
        #filter_size=(3, 3),
        ## nonlinearity=nonlinearities.rectify,
        #nonlinearity=nonlinearities.LeakyRectify(leakiness=(1. / 10.)),

        #rlu_glorot = dict(nonlinearity=nonlinearities.rectify, W=init.GlorotUniform())
        #rlu_orthog = dict(nonlinearity=nonlinearities.rectify, W=init.Orthogonal())
        leaky = dict(nonlinearity=nonlinearities.LeakyRectify(leakiness=(1. / 10.)))
        leaky_orthog = dict(W=init.Orthogonal(), **leaky)
        # variable batch size (None), channel, width, height
        #input_shape = (batch_size * model.data_per_label, input_channels, input_width, input_height)

        vggnet = PretrainedNetwork('vggnet')
        #caffenet = PretrainedNetwork('caffenet')

        network_layers_def = [
            _P(layers.InputLayer, shape=model.input_shape),

            #layers.GaussianNoiseLayer,
            vggnet.get_conv2d_layer(0, **leaky),

            _P(Conv2DLayer, num_filters=16, filter_size=(3, 3), **leaky_orthog),

            _P(MaxPool2DLayer, pool_size=(2, 2), stride=(2, 2)),
            _P(Conv2DLayer, num_filters=16, filter_size=(3, 3), **leaky_orthog),

            _P(MaxPool2DLayer, pool_size=(2, 2), stride=(2, 2)),

            #_P(layers.DenseLayer, num_units=512, **leaky_orthog),
            #_P(layers.DropoutLayer, p=0.5),
            _P(layers.DenseLayer, num_units=256, **leaky_orthog),
            _P(layers.DropoutLayer, p=0.5),
            #_P(lasange_ext.l1),  # TODO: make a layer

            #_P(layers.DenseLayer, num_units=256, **leaky_orthog),
            #_P(layers.DropoutLayer, p=0.5),
            #_P(layers.FeaturePoolLayer, pool_size=2),

            #_P(layers.DenseLayer, num_units=1024, **leaky_orthog),
            #_P(layers.FeaturePoolLayer, pool_size=2,),
            #_P(layers.DropoutLayer, p=0.5),

            #_P(layers.DenseLayer, num_units=output_dims,
            #   nonlinearity=nonlinearities.softmax,
            #   W=init.Orthogonal(),),
        ]

        # connect and record layers
        network_layers = evaluate_layer_list(network_layers_def)
        model.network_layers = network_layers
        output_layer = model.network_layers[-1]
        return output_layer

    def loss_function(self, G, Y_padded, T=T, verbose=True):
        if verbose:
            print('[model] Build center surround siamese loss function')
        avg_loss = lasange_ext.siamese_loss(G, Y_padded, data_per_label=2)
        return avg_loss


class ViewpointModel(BaseModel):
    def __init__(self):
        super(ViewpointModel, self).__init__()

    def augment(self, Xb, yb=None):
        # Invert label function
        def _invert_label(label):
            label = label.replace('LEFT',  '^L^')
            label = label.replace('RIGHT', '^R^')
            label = label.replace('^R^', 'LEFT')
            label = label.replace('^L^', 'RIGHT')
            return(label)
        # Map
        points, channels, height, width = Xb.shape
        for index in range(points):
            if random.uniform(0.0, 1.0) <= 0.5:
                Xb[index] = Xb[index, :, ::-1]
                if yb is not None:
                    yb[index] = _invert_label(yb[index])
        return Xb, yb

    def label_order_mapping(self, category_list):
        if len(category_list) == 8:
            species_list = [
                'ZEBRA_PLAINS',
            ]
        else:
            species_list = [
                'ZEBRA_PLAINS',
                'ZEBRA_GREVYS',
                'ELEPHANT_SAVANNA',
                'GIRAFFE_RETICULATED',
                'GIRAFFE_MASAI',
            ]
        viewpoint_mapping = {
            'LEFT':        0,
            'FRONT_LEFT':  1,
            'FRONT':       2,
            'FRONT_RIGHT': 3,
            'RIGHT':       4,
            'BACK_RIGHT':  5,
            'BACK':        6,
            'BACK_LEFT':   7,
        }
        viewpoints = len(viewpoint_mapping.keys())
        category_mapping = {}
        for index, species in enumerate(species_list):
            for viewpoint, value in viewpoint_mapping.iteritems():
                key = '%s:%s' % (species, viewpoint, )
                base = viewpoints * index
                category_mapping[key] = base + value
        return category_mapping

    def learning_rate_update(self, x):
        return x / 2.0

    def learning_rate_shock(self, x):
        return x * 2.0

    def build_model(self, batch_size, input_width, input_height, input_channels, output_dims):
        _CaffeNet = PretrainedNetwork('caffenet')

        l_in = layers.InputLayer(
            # variable batch size (None), channel, width, height
            shape=(None, input_channels, input_width, input_height)
        )

        l_noise = layers.GaussianNoiseLayer(
            l_in,
        )

        l_conv0 = Conv2DLayer(
            l_noise,
            num_filters=32,
            filter_size=(11, 11),
            # nonlinearity=nonlinearities.rectify,
            nonlinearity=nonlinearities.LeakyRectify(leakiness=(1. / 10.)),
            W=_CaffeNet.get_pretrained_layer(0),
        )

        l_conv0_dropout = layers.DropoutLayer(l_conv0, p=0.10)

        l_conv1 = Conv2DLayer(
            l_conv0_dropout,
            num_filters=32,
            filter_size=(5, 5),
            # nonlinearity=nonlinearities.rectify,
            nonlinearity=nonlinearities.LeakyRectify(leakiness=(1. / 10.)),
            W=_CaffeNet.get_pretrained_layer(2),
        )

        l_pool1 = MaxPool2DLayer(
            l_conv1,
            pool_size=(2, 2),
            stride=(2, 2),
        )

        l_conv2_dropout = layers.DropoutLayer(l_pool1, p=0.10)

        l_conv2 = Conv2DLayer(
            l_conv2_dropout,
            num_filters=64,
            filter_size=(3, 3),
            # nonlinearity=nonlinearities.rectify,
            nonlinearity=nonlinearities.LeakyRectify(leakiness=(1. / 10.)),
            W=init.Orthogonal(),
        )

        l_pool2 = MaxPool2DLayer(
            l_conv2,
            pool_size=(2, 2),
            stride=(2, 2),
        )

        l_conv3_dropout = layers.DropoutLayer(l_pool2, p=0.30)

        l_conv3 = Conv2DLayer(
            l_conv3_dropout,
            num_filters=128,
            filter_size=(3, 3),
            # nonlinearity=nonlinearities.rectify,
            nonlinearity=nonlinearities.LeakyRectify(leakiness=(1. / 10.)),
            W=init.Orthogonal(),
        )

        l_pool3 = MaxPool2DLayer(
            l_conv3,
            pool_size=(2, 2),
            stride=(2, 2),
        )

        l_conv4_dropout = layers.DropoutLayer(l_pool3, p=0.30)

        l_conv4 = Conv2DLayer(
            l_conv4_dropout,
            num_filters=128,
            filter_size=(3, 3),
            # nonlinearity=nonlinearities.rectify,
            nonlinearity=nonlinearities.LeakyRectify(leakiness=(1. / 10.)),
            W=init.Orthogonal(),
        )

        l_pool4 = MaxPool2DLayer(
            l_conv4,
            pool_size=(2, 2),
            stride=(2, 2),
        )

        l_hidden1 = layers.DenseLayer(
            l_pool4,
            num_units=512,
            # nonlinearity=nonlinearities.rectify,
            nonlinearity=nonlinearities.LeakyRectify(leakiness=(1. / 10.)),
            W=init.Orthogonal(),
        )

        l_hidden1_maxout = layers.FeaturePoolLayer(
            l_hidden1,
            pool_size=2,
        )

        l_hidden1_dropout = layers.DropoutLayer(l_hidden1_maxout, p=0.5)

        l_hidden2 = layers.DenseLayer(
            l_hidden1_dropout,
            num_units=512,
            # nonlinearity=nonlinearities.rectify,
            nonlinearity=nonlinearities.LeakyRectify(leakiness=(1. / 10.)),
            W=init.Orthogonal(),
        )

        l_hidden2_maxout = layers.FeaturePoolLayer(
            l_hidden2,
            pool_size=2,
        )

        l_hidden2_dropout = layers.DropoutLayer(l_hidden2_maxout, p=0.5)

        l_out = layers.DenseLayer(
            l_hidden2_dropout,
            num_units=output_dims,
            nonlinearity=nonlinearities.softmax,
            W=init.Orthogonal(),
        )

        return l_out


class QualityModel(BaseModel):
    def __init__(self):
        super(QualityModel, self).__init__()

    def label_order_mapping(self, category_list):
        quality_mapping = {
            'JUNK':      0,
            'POOR':      1,
            'GOOD':      2,
            'OK':        3,
            'EXCELLENT': 4,
        }
        return quality_mapping

    def learning_rate_update(self, x):
        return x / 2.0

    def learning_rate_shock(self, x):
        return x * 2.0

    def build_model(self, batch_size, input_width, input_height, input_channels, output_dims):
        _CaffeNet = PretrainedNetwork('caffenet')

        l_in = layers.InputLayer(
            # variable batch size (None), channel, width, height
            shape=(None, input_channels, input_width, input_height)
        )

        l_noise = layers.GaussianNoiseLayer(
            l_in,
        )

        l_conv0 = Conv2DLayer(
            l_noise,
            num_filters=32,
            filter_size=(11, 11),
            # nonlinearity=nonlinearities.rectify,
            nonlinearity=nonlinearities.LeakyRectify(leakiness=(1. / 10.)),
            W=_CaffeNet.get_pretrained_layer(0),
        )

        l_conv0_dropout = layers.DropoutLayer(l_conv0, p=0.10)

        l_conv1 = Conv2DLayer(
            l_conv0_dropout,
            num_filters=32,
            filter_size=(5, 5),
            # nonlinearity=nonlinearities.rectify,
            nonlinearity=nonlinearities.LeakyRectify(leakiness=(1. / 10.)),
            W=_CaffeNet.get_pretrained_layer(2),
        )

        l_pool1 = MaxPool2DLayer(
            l_conv1,
            pool_size=(2, 2),
            stride=(2, 2),
        )

        l_conv2_dropout = layers.DropoutLayer(l_pool1, p=0.10)

        l_conv2 = Conv2DLayer(
            l_conv2_dropout,
            num_filters=64,
            filter_size=(3, 3),
            # nonlinearity=nonlinearities.rectify,
            nonlinearity=nonlinearities.LeakyRectify(leakiness=(1. / 10.)),
            W=init.Orthogonal(),
        )

        l_pool2 = MaxPool2DLayer(
            l_conv2,
            pool_size=(2, 2),
            stride=(2, 2),
        )

        l_conv3_dropout = layers.DropoutLayer(l_pool2, p=0.30)

        l_conv3 = Conv2DLayer(
            l_conv3_dropout,
            num_filters=128,
            filter_size=(3, 3),
            # nonlinearity=nonlinearities.rectify,
            nonlinearity=nonlinearities.LeakyRectify(leakiness=(1. / 10.)),
            W=init.Orthogonal(),
        )

        l_pool3 = MaxPool2DLayer(
            l_conv3,
            pool_size=(2, 2),
            stride=(2, 2),
        )

        l_conv4_dropout = layers.DropoutLayer(l_pool3, p=0.30)

        l_conv4 = Conv2DLayer(
            l_conv4_dropout,
            num_filters=128,
            filter_size=(3, 3),
            # nonlinearity=nonlinearities.rectify,
            nonlinearity=nonlinearities.LeakyRectify(leakiness=(1. / 10.)),
            W=init.Orthogonal(),
        )

        l_pool4 = MaxPool2DLayer(
            l_conv4,
            pool_size=(2, 2),
            stride=(2, 2),
        )

        l_hidden1 = layers.DenseLayer(
            l_pool4,
            num_units=512,
            # nonlinearity=nonlinearities.rectify,
            nonlinearity=nonlinearities.LeakyRectify(leakiness=(1. / 10.)),
            W=init.Orthogonal(),
        )

        l_hidden1_maxout = layers.FeaturePoolLayer(
            l_hidden1,
            pool_size=2,
        )

        l_hidden1_dropout = layers.DropoutLayer(l_hidden1_maxout, p=0.5)

        l_hidden2 = layers.DenseLayer(
            l_hidden1_dropout,
            num_units=512,
            # nonlinearity=nonlinearities.rectify,
            nonlinearity=nonlinearities.LeakyRectify(leakiness=(1. / 10.)),
            W=init.Orthogonal(),
        )

        l_hidden2_maxout = layers.FeaturePoolLayer(
            l_hidden2,
            pool_size=2,
        )

        l_hidden2_dropout = layers.DropoutLayer(l_hidden2_maxout, p=0.5)

        l_out = layers.DenseLayer(
            l_hidden2_dropout,
            num_units=output_dims,
            nonlinearity=nonlinearities.softmax,
            W=init.Orthogonal(),
        )

        return l_out


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis_cnn.models
        python -m ibeis_cnn.models --allexamples
        python -m ibeis_cnn.models --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
