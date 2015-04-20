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
import random
import utool as ut
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

    def build_model(self, batch_size, input_width, input_height, input_channels, output_dims, verbose=ut.VERBOSE):
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
