# file model.py
# allows the definition of different models to be trained
# for initialization: Lasagne/lasagne/init.py
# for nonlinearities: Lasagne/lasagne/nonlinearities.py
# for layers: Lasagne/lasagne/layers/

from lasagne import layers
from lasagne.layers import cuda_convnet
from lasagne import nonlinearities
from lasagne import init

# use cuda_convnet for a speed improvement
# will not be available without a GPU
Conv2DLayer = cuda_convnet.Conv2DCCLayer
MaxPool2DLayer = cuda_convnet.MaxPool2DCCLayer


def build_model(batch_size, input_width, input_height, input_channels, output_dims):
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
        nonlinearity=nonlinearities.LeakyRectify,
        W=init.GlorotUniform(),
    )

    l_pool1 = MaxPool2DLayer(
        l_conv1,
        ds=(2, 2),
        strides=(2, 2),
    )

    l_conv2 = Conv2DLayer(
        l_pool1,
        num_filters=64,
        filter_size=(3, 3),
        # nonlinearity=nonlinearities.rectify,
        nonlinearity=nonlinearities.LeakyRectify,
        W=init.GlorotUniform(),
    )

    l_pool2 = MaxPool2DLayer(
        l_conv2,
        ds=(2, 2),
        strides=(2, 2),
    )

    l_conv3 = Conv2DLayer(
        l_pool2,
        num_filters=128,
        filter_size=(3, 3),
        # nonlinearity=nonlinearities.rectify,
        nonlinearity=nonlinearities.LeakyRectify,
        W=init.GlorotUniform(),
    )

    l_pool3 = MaxPool2DLayer(
        l_conv3,
        ds=(2, 2),
        strides=(2, 2),
    )

    l_hidden1 = layers.DenseLayer(
        l_pool3,
        num_units=1024,
        # nonlinearity=nonlinearities.rectify,
        nonlinearity=nonlinearities.LeakyRectify,
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
        # nonlinearity=nonlinearities.rectify,
        nonlinearity=nonlinearities.LeakyRectify,
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
