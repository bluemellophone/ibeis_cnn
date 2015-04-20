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


def array_tf_0(arr):
    return arr


def array_tf_90(arr):
    axes_order = range(arr.ndim - 2) + [arr.ndim - 1, arr.ndim - 2]
    slices = [slice(None) for _ in range(arr.ndim - 2)] + [slice(None), slice(None, None, -1)]
    return arr[tuple(slices)].transpose(axes_order)


def array_tf_180(arr):
    slices = [slice(None) for _ in range(arr.ndim - 2)] + [slice(None, None, -1), slice(None, None, -1)]
    return arr[tuple(slices)]


def array_tf_270(arr):
    axes_order = range(arr.ndim - 2) + [arr.ndim - 1, arr.ndim - 2]
    slices = [slice(None) for _ in range(arr.ndim - 2)] + [slice(None, None, -1), slice(None)]
    return arr[tuple(slices)].transpose(axes_order)


def array_tf_0f(arr):  # horizontal flip
    slices = [slice(None) for _ in range(arr.ndim - 2)] + [slice(None), slice(None, None, -1)]
    return arr[tuple(slices)]


def array_tf_90f(arr):
    axes_order = range(arr.ndim - 2) + [arr.ndim - 1, arr.ndim - 2]
    slices = [slice(None) for _ in range(arr.ndim - 2)] + [slice(None), slice(None)]
    # slicing does nothing here, technically I could get rid of it.
    return arr[tuple(slices)].transpose(axes_order)


def array_tf_180f(arr):
    slices = [slice(None) for _ in range(arr.ndim - 2)] + [slice(None, None, -1), slice(None)]
    return arr[tuple(slices)]


def array_tf_270f(arr):
    axes_order = range(arr.ndim - 2) + [arr.ndim - 1, arr.ndim - 2]
    slices = [slice(None) for _ in range(arr.ndim - 2)] + [slice(None, None, -1), slice(None, None, -1)]
    return arr[tuple(slices)].transpose(axes_order)


# c01b versions of the helper functions


def array_tf_0_c01b(arr):
    return arr


def array_tf_90_c01b(arr):
    axes_order = [0, 2, 1, 3]
    slices = [slice(None), slice(None), slice(None, None, -1), slice(None)]
    return arr[tuple(slices)].transpose(axes_order)


def array_tf_180_c01b(arr):
    slices = [slice(None), slice(None, None, -1), slice(None, None, -1), slice(None)]
    return arr[tuple(slices)]


def array_tf_270_c01b(arr):
    axes_order = [0, 2, 1, 3]
    slices = [slice(None), slice(None, None, -1), slice(None), slice(None)]
    return arr[tuple(slices)].transpose(axes_order)


def array_tf_0f_c01b(arr):  # horizontal flip
    slices = [slice(None), slice(None), slice(None, None, -1), slice(None)]
    return arr[tuple(slices)]


def array_tf_90f_c01b(arr):
    axes_order = [0, 2, 1, 3]
    slices = [slice(None), slice(None), slice(None), slice(None)]
    # slicing does nothing here, technically I could get rid of it.
    return arr[tuple(slices)].transpose(axes_order)


def array_tf_180f_c01b(arr):
    slices = [slice(None), slice(None, None, -1), slice(None), slice(None)]
    return arr[tuple(slices)]


def array_tf_270f_c01b(arr):
    axes_order = [0, 2, 1, 3]
    slices = [slice(None), slice(None, None, -1), slice(None, None, -1), slice(None)]
    return arr[tuple(slices)].transpose(axes_order)


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
            array_tf_0(input_),
            array_tf_90(input_),
            array_tf_180(input_),
            array_tf_270(input_),
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
        self.inv_tf_funcs = [array_tf_0, array_tf_270, array_tf_180, array_tf_90]
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
        from functools import partial as _P
        if verbose:
            print('[model] Build model')
            print('[model]   * batch_size     = %r' % (batch_size,))
            print('[model]   * input_width    = %r' % (input_width,))
            print('[model]   * input_height   = %r' % (input_height,))
            print('[model]   * input_channels = %r' % (input_channels,))
            print('[model]   * output_dims    = %r' % (output_dims,))

        rlu_glorot = dict(nonlinearity=nonlinearities.rectify, W=init.GlorotUniform())

        network_layers = [
            layers.GaussianNoiseLayer,

            # Convolve + Max Pool 1
            _P(Conv2DLayer, num_filters=32, filter_size=(5, 5), **rlu_glorot),
            _P(MaxPool2DLayer_, ds=(2, 2), strides=(2, 2)),

            ## Convolve + Max Pool 2
            #_P(Conv2DLayer, num_filters=64, filter_size=(3, 3), **rlu_glorot),
            #_P(MaxPool2DLayer_, ds=(2, 2), strides=(2, 2),),

            ## Convolve + Max Pool 3
            #_P(Conv2DLayer, num_filters=128, filter_size=(3, 3), **rlu_glorot),
            #_P(MaxPool2DLayer_, ds=(2, 2), strides=(2, 2),),

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

        input_layer = layers.InputLayer(
            # variable batch size (None), channel, width, height
            shape=(None, input_channels, input_width, input_height)
        )

        # connect layers
        prev_layer = input_layer
        for layer_fn in network_layers:
            prev_layer = layer_fn(prev_layer)

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
            nonlinearity=nonlinearities.rectify,
            # nonlinearity=nonlinearities.LeakyRectify,
            W=init.GlorotUniform(),
        )

        l_pool1 = MaxPool2DLayer_(
            l_conv1,
            ds=(2, 2),
            strides=(2, 2),
        )

        l_conv2 = Conv2DLayer(
            l_pool1,
            num_filters=64,
            filter_size=(3, 3),
            nonlinearity=nonlinearities.rectify,
            # nonlinearity=nonlinearities.LeakyRectify,
            W=init.GlorotUniform(),
        )

        l_pool2 = MaxPool2DLayer_(
            l_conv2,
            ds=(2, 2),
            strides=(2, 2),
        )

        l_conv3 = Conv2DLayer(
            l_pool2,
            num_filters=128,
            filter_size=(3, 3),
            nonlinearity=nonlinearities.rectify,
            # nonlinearity=nonlinearities.LeakyRectify,
            W=init.GlorotUniform(),
        )

        l_pool3 = MaxPool2DLayer_(
            l_conv3,
            ds=(2, 2),
            strides=(2, 2),
        )

        l_hidden1 = layers.DenseLayer(
            l_pool3,
            num_units=1024,
            nonlinearity=nonlinearities.rectify,
            # nonlinearity=nonlinearities.LeakyRectify,
            W=init.GlorotUniform(),
        )

        l_hidden1_maxout = layers.FeaturePoolLayer(
            l_hidden1,
            ds=2,
        )

        l_hidden1_dropout = layers.DropoutLayer(l_hidden1_maxout, p=0.5)

        l_hidden2 = layers.DenseLayer(
            l_hidden1_dropout,
            num_units=1024,
            nonlinearity=nonlinearities.rectify,
            # nonlinearity=nonlinearities.LeakyRectify,
            W=init.GlorotUniform(),
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
            W=init.GlorotUniform(),
        )

        return l_out


class PZ_Model(object):
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
            num_filters=64,
            filter_size=(5, 5),
            nonlinearity=nonlinearities.rectify,
            # nonlinearity=nonlinearities.LeakyRectify,
            W=init.GlorotUniform(),
        )

        l_pool1 = MaxPool2DLayer_(
            l_conv1,
            ds=(2, 2),
            strides=(2, 2),
        )

        l_conv2 = Conv2DLayer(
            l_pool1,
            num_filters=128,
            filter_size=(3, 3),
            nonlinearity=nonlinearities.rectify,
            # nonlinearity=nonlinearities.LeakyRectify,
            W=init.GlorotUniform(),
        )

        l_pool2 = MaxPool2DLayer_(
            l_conv2,
            ds=(2, 2),
            strides=(2, 2),
        )

        l_conv3 = Conv2DLayer(
            l_pool2,
            num_filters=128,
            filter_size=(3, 3),
            nonlinearity=nonlinearities.rectify,
            # nonlinearity=nonlinearities.LeakyRectify,
            W=init.GlorotUniform(),
        )

        l_pool3 = MaxPool2DLayer_(
            l_conv3,
            ds=(2, 2),
            strides=(2, 2),
        )

        l_hidden1 = layers.DenseLayer(
            l_pool3,
            num_units=1024,
            nonlinearity=nonlinearities.rectify,
            # nonlinearity=nonlinearities.LeakyRectify,
            W=init.GlorotUniform(),
        )

        l_hidden1_maxout = layers.FeaturePoolLayer(
            l_hidden1,
            ds=2,
        )

        l_hidden1_dropout = layers.DropoutLayer(l_hidden1_maxout, p=0.5)

        l_hidden2 = layers.DenseLayer(
            l_hidden1_dropout,
            num_units=1024,
            nonlinearity=nonlinearities.rectify,
            # nonlinearity=nonlinearities.LeakyRectify,
            W=init.GlorotUniform(),
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
            W=init.GlorotUniform(),
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
