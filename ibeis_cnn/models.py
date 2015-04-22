"""
file model.py
allows the definition of different models to be trained
for initialization: Lasagne/lasagne/init.py
for nonlinearities: Lasagne/lasagne/nonlinearities.py
for layers: Lasagne/lasagne/layers/
"""
from __future__ import absolute_import, division, print_function
import lasagne
from lasagne import layers
from lasagne import nonlinearities
from lasagne import init
import random
import utool as ut
import numpy as np
import six

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


class MultiImageSliceLayer(lasagne.layers.Layer):
    """
    orig CyclicSliceLayer
    References:
        https://github.com/benanne/kaggle-ndsb/blob/master/dihedral.py#L89

    This layer stacks rotations of 0, 90, 180, and 270 degrees of the input
    along the batch dimension.
    If the input has shape (batch_size, num_channels, r, c),
    then the output will have shape (4 * batch_size, num_channels, r, c).
    Note that the stacking happens on axis 0, so a reshape to
    (4, batch_size, num_channels, r, c) will separate the slice axis.
    """
    def __init__(self, input_layer):
        super(MultiImageSliceLayer, self).__init__(input_layer)

    def get_output_shape_for(self, input_shape):
        return (4 * input_shape[0],) + input_shape[1:]

    def get_output_for(self, input_, *args, **kwargs):
        return lasagne.utils.concatenate([
            #array_tf_0(input_),
            #array_tf_90(input_),
            #array_tf_180(input_),
            #array_tf_270(input_),
        ], axis=0)


class MultiImageRollLayer(lasagne.layers.Layer):
    """
    orig CyclicConvRollLayer


    This layer turns (n_views * batch_size, num_channels, r, c) into
    (n_views * batch_size, n_views * num_channels, r, c) by rolling
    and concatenating feature maps.
    It also applies the correct inverse transforms to the r and c
    dimensions to align the feature maps.

    References:
        https://github.com/benanne/kaggle-ndsb/blob/master/dihedral.py#L224
    """
    def __init__(self, input_layer):
        super(MultiImageRollLayer, self).__init__(input_layer)
        self.inv_tf_funcs = []  # array_tf_0, array_tf_270, array_tf_180, array_tf_90]
        self.compute_permutation_matrix()

    def compute_permutation_matrix(self):
        map_identity = np.arange(4)
        map_rot90 = np.array([1, 2, 3, 0])

        valid_maps = []
        current_map = map_identity
        for k in range(4):
            valid_maps.append(current_map)
            current_map = current_map[map_rot90]

        self.perm_matrix = np.array(valid_maps)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], 4 * input_shape[1]) + input_shape[2:]

    def get_output_for(self, input_, *args, **kwargs):
        s = input_.shape
        input_unfolded = input_.reshape((4, s[0] // 4, s[1], s[2], s[3]))

        permuted_inputs = []
        for p, inv_tf in zip(self.perm_matrix, self.inv_tf_funcs):
            input_permuted = inv_tf(input_unfolded[p].reshape(s))
            permuted_inputs.append(input_permuted)

        return lasagne.utils.concatenate(permuted_inputs, axis=1)  # concatenate long the channel axis


import theano.tensor as T


class CyclicPoolLayer(lasagne.layers.Layer):
    """
    Utility layer that unfolds the viewpoints dimension and pools over it.
    Note that this only makes sense for dense representations, not for
    feature maps (because no inverse transforms are applied to align them).
    """
    def __init__(self, input_layer, pool_function=T.mean):
        super(CyclicPoolLayer, self).__init__(input_layer)
        self.pool_function = pool_function

    def get_output_shape_for(self, input_shape):
        return (input_shape[0] // 4, input_shape[1])

    def get_output_for(self, input_, *args, **kwargs):
        unfolded_input = input_.reshape((4, input_.shape[0] // 4, input_.shape[1]))
        return self.pool_function(unfolded_input, axis=0)


def MaxPool2DLayer_(*args, **kwargs):
    """ wrapper for gpu / cpu compatibility """
    if not USING_GPU and 'strides' in kwargs:
        # cpu does not have stride kwarg. :(
        del kwargs['strides']
    return MaxPool2DLayer(*args, **kwargs)


class IdentificationModel(object):
    """
    Model for individual identification
    """
    def __init__(self):
        pass

    def learning_rate_update(self, x):
        return x / 10.0

    def learning_rate_shock(self, x):
        return x * 10.0

    def build_model(self, batch_size, input_width, input_height,
                    input_channels, output_dims, verbose=ut.VERBOSE):
        from functools import partial as _P  # NOQA
        if verbose:
            print('[model] Build model')
            print('[model]   * batch_size     = %r' % (batch_size,))
            print('[model]   * input_width    = %r' % (input_width,))
            print('[model]   * input_height   = %r' % (input_height,))
            print('[model]   * input_channels = %r' % (input_channels,))
            print('[model]   * output_dims    = %r' % (output_dims,))

        # JON, ADD THIS INSTEAD W=init.Orthogonal  -- Jason

        rlu_glorot = dict(nonlinearity=nonlinearities.rectify, W=init.GlorotUniform())
        # variable batch size (None), channel, width, height
        input_shape = (None, input_channels, input_width, input_height)

        network_layers_def = [
            _P(layers.InputLayer, shape=input_shape),

            #layers.GaussianNoiseLayer,

            # Convolve + Max Pool 1
            _P(Conv2DLayer, num_filters=32, filter_size=(5, 5), **rlu_glorot),
            _P(MaxPool2DLayer_, ds=(2, 2), strides=(2, 2)),

            # Convolve + Max Pool 2
            _P(Conv2DLayer, num_filters=64, filter_size=(4, 4), **rlu_glorot),
            _P(MaxPool2DLayer_, ds=(2, 2), strides=(2, 2),),

            # Convolve + Max Pool 3
            _P(Conv2DLayer, num_filters=128, filter_size=(3, 3), **rlu_glorot),
            _P(MaxPool2DLayer_, ds=(2, 2), strides=(2, 2),),

            # Convolve + Max Pool 4
            _P(Conv2DLayer, num_filters=64, filter_size=(2, 2), **rlu_glorot),
            _P(MaxPool2DLayer_, ds=(2, 2), strides=(2, 2),),

            # Dense Layer + Feature Pool + Dropout 1
            _P(layers.DenseLayer, num_units=1024, **rlu_glorot),
            _P(layers.FeaturePoolLayer, ds=2),
            _P(layers.DropoutLayer, p=0.5),

            ## Dense Layer + Feature Pool + Dropout 2
            #_P(layers.DenseLayer, num_units=1024, **rlu_glorot),
            #_P(layers.FeaturePoolLayer, ds=2,),
            #_P(layers.DropoutLayer, p=0.5),

            # Softmax output
            _P(layers.DenseLayer, num_units=output_dims,
               nonlinearity=nonlinearities.softmax,
               W=init.GlorotUniform(),),
        ]

        # connect and record layers
        network_layers = []
        layer_fn_iter = iter(network_layers_def)
        prev_layer = six.next(layer_fn_iter)()
        network_layers.append(prev_layer)
        for layer_fn in layer_fn_iter:
            prev_layer = layer_fn(prev_layer)
            network_layers.append(prev_layer)

        self.network_layers = network_layers
        output_layer = prev_layer
        return output_layer


class PZ_GIRM_Model(object):
    def __init__(self):
        pass

    def augment(self, Xb, yb):
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
                yb[index] = _invert_label(yb[index])
        return Xb, yb

    def label_order_mapping(self, category_list):
        species_list = [
            'ZEBRA_PLAINS',
            'ZEBRA_GREVYS',
            'ELEPHANT_SAVANNA',
            'GIRAFFE_RETICULATED:',
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
        return x / 10.0

    def learning_rate_shock(self, x):
        return x * 10.0

    def build_model(self, batch_size, input_width, input_height, input_channels, output_dims):
        l_in = layers.InputLayer(
            # variable batch size (None), channel, width, height
            shape=(None, input_channels, input_width, input_height)
        )

        l_noise = layers.GaussianNoiseLayer(
            l_in,
        )

        l_conv1 = Conv2DLayer(
            l_noise,
            num_filters=32,
            filter_size=(5, 5),
            # nonlinearity=nonlinearities.rectify,
            nonlinearity=nonlinearities.LeakyRectify(leakiness=(1. / 10.)),
            W=init.Orthogonal(),
        )

        l_pool1 = MaxPool2DLayer_(
            l_conv1,
            ds=(2, 2),
            strides=(2, 2),
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

        l_pool2 = MaxPool2DLayer_(
            l_conv2,
            ds=(2, 2),
            strides=(2, 2),
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

        l_pool3 = MaxPool2DLayer_(
            l_conv3,
            ds=(2, 2),
            strides=(2, 2),
        )

        l_hidden1 = layers.DenseLayer(
            l_pool3,
            num_units=512,
            # nonlinearity=nonlinearities.rectify,
            nonlinearity=nonlinearities.LeakyRectify(leakiness=(1. / 10.)),
            W=init.Orthogonal(),
        )

        l_hidden1_maxout = layers.FeaturePoolLayer(
            l_hidden1,
            ds=2,
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
            ds=2,
        )

        l_hidden2_dropout = layers.DropoutLayer(l_hidden2_maxout, p=0.5)

        l_out = layers.DenseLayer(
            l_hidden2_dropout,
            num_units=output_dims,
            nonlinearity=nonlinearities.softmax,
            W=init.Orthogonal(),
        )

        return l_out


class PZ_GIRM_LARGE_Model(object):
    def __init__(self):
        pass

    def augment(self, Xb, yb):
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
                yb[index] = _invert_label(yb[index])
        return Xb, yb

    def label_order_mapping(self, category_list):
        category_mapping = {
            'ZEBRA_PLAINS:LEFT':        0,
            'ZEBRA_PLAINS:FRONT_LEFT':  1,
            'ZEBRA_PLAINS:FRONT':       2,
            'ZEBRA_PLAINS:FRONT_RIGHT': 3,
            'ZEBRA_PLAINS:RIGHT':       4,
            'ZEBRA_PLAINS:BACK_RIGHT':  5,
            'ZEBRA_PLAINS:BACK':        6,
            'ZEBRA_PLAINS:BACK_LEFT':   7,
        }
        return category_mapping

    def learning_rate_update(self, x):
        return x / 10.0

    def learning_rate_shock(self, x):
        return x * 10.0

    def build_model(self, batch_size, input_width, input_height, input_channels, output_dims):
        l_in = layers.InputLayer(
            # variable batch size (None), channel, width, height
            shape=(None, input_channels, input_width, input_height)
        )

        l_noise = layers.GaussianNoiseLayer(
            l_in,
        )

        l_conv1 = Conv2DLayer(
            l_noise,
            num_filters=32,
            filter_size=(5, 5),
            # nonlinearity=nonlinearities.rectify,
            nonlinearity=nonlinearities.LeakyRectify(leakiness=(1. / 10.)),
            W=init.Orthogonal(),
        )

        l_pool1 = MaxPool2DLayer_(
            l_conv1,
            ds=(2, 2),
            strides=(2, 2),
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

        l_pool2 = MaxPool2DLayer_(
            l_conv2,
            ds=(2, 2),
            strides=(2, 2),
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

        l_pool3 = MaxPool2DLayer_(
            l_conv3,
            ds=(2, 2),
            strides=(2, 2),
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

        l_pool4 = MaxPool2DLayer_(
            l_conv4,
            ds=(2, 2),
            strides=(2, 2),
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
            ds=2,
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
            ds=2,
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
