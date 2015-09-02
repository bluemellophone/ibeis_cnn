# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import lasagne  # NOQA
from lasagne import layers
from lasagne import nonlinearities
from lasagne import init
from ibeis_cnn import custom_layers
from ibeis_cnn.models import abstract_models
import six
import utool as ut
print, rrr, profile = ut.inject2(__name__, '[ibeis_cnn.models.background]')


Conv2DLayer = custom_layers.Conv2DLayer
MaxPool2DLayer = custom_layers.MaxPool2DLayer


@six.add_metaclass(ut.ReloadingMetaclass)
class BackgroundModel(abstract_models.AbstractCategoricalModel):
    def __init__(self):
        super(BackgroundModel, self).__init__()

    def learning_rate_update(self, x):
        return x / 2.0

    def learning_rate_shock(self, x):
        return x * 2.0

    def build_model(self, batch_size, input_width, input_height, input_channels, output_dims):
        _CaffeNet = abstract_models.PretrainedNetwork('caffenet')
        _leaky_relu = nonlinearities.LeakyRectify(leakiness=(1. / 10.))

        l_in = layers.InputLayer(
            # variable batch size (None), channel, width, height
            shape=(None, input_channels, input_width, input_height)
        )

        l_conv0 = Conv2DLayer(
            l_in,
            num_filters=32,
            filter_size=(11, 11),
            # nonlinearity=nonlinearities.rectify,
            nonlinearity=_leaky_relu,
            W=_CaffeNet.get_pretrained_layer(0),
        )

        l_pool0 = MaxPool2DLayer(
            l_conv0,
            pool_size=(2, 2),
            stride=(2, 2),
        )

        l_conv0_dropout = layers.DropoutLayer(l_pool0, p=0.1)

        l_conv1 = Conv2DLayer(
            l_conv0_dropout,
            num_filters=64,
            filter_size=(5, 5),
            # nonlinearity=nonlinearities.rectify,
            nonlinearity=_leaky_relu,
            W=_CaffeNet.get_pretrained_layer(2),
        )

        l_pool1 = MaxPool2DLayer(
            l_conv1,
            pool_size=(2, 2),
            stride=(2, 2),
        )

        l_conv1_dropout = layers.DropoutLayer(l_pool1, p=0.2)

        l_conv2 = Conv2DLayer(
            l_conv1_dropout,
            num_filters=128,
            filter_size=(3, 3),
            # nonlinearity=nonlinearities.rectify,
            nonlinearity=_leaky_relu,
            W=init.Orthogonal(),
        )

        l_conv2_dropout = layers.DropoutLayer(l_conv2, p=0.3)

        l_conv3 = Conv2DLayer(
            l_conv2_dropout,
            num_filters=128,
            filter_size=(3, 3),
            # nonlinearity=nonlinearities.rectify,
            nonlinearity=_leaky_relu,
            W=init.Orthogonal(),
        )

        l_pool3 = MaxPool2DLayer(
            l_conv3,
            pool_size=(2, 2),
            stride=(2, 2),
        )

        l_conv3_dropout = layers.DropoutLayer(l_pool3, p=0.5)

        l_conv4 = Conv2DLayer(
            l_conv3_dropout,
            num_filters=256,
            filter_size=(3, 3),
            # nonlinearity=nonlinearities.rectify,
            nonlinearity=_leaky_relu,
            W=init.Orthogonal(),
        )

        l_conv4_dropout = layers.DropoutLayer(l_conv4, p=0.5)

        l_conv5 = Conv2DLayer(
            l_conv4_dropout,
            num_filters=256,
            filter_size=(3, 3),
            # nonlinearity=nonlinearities.rectify,
            nonlinearity=_leaky_relu,
            W=init.Orthogonal(),
        )

        l_conv5_dropout = layers.DropoutLayer(l_conv5, p=0.5)

        l_hidden1 = layers.DenseLayer(
            l_conv5_dropout,
            num_units=1024,
            # nonlinearity=nonlinearities.rectify,
            nonlinearity=_leaky_relu,
            W=init.Orthogonal(),
        )

        l_hidden1_maxout = layers.FeaturePoolLayer(
            l_hidden1,
            pool_size=2,
        )

        l_hidden1_dropout = layers.DropoutLayer(l_hidden1_maxout, p=0.5)

        l_hidden2 = layers.DenseLayer(
            l_hidden1_dropout,
            num_units=1024,
            # nonlinearity=nonlinearities.rectify,
            nonlinearity=_leaky_relu,
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
        self.output_layer = l_out
        return l_out


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis_cnn.models.background
        python -m ibeis_cnn.models.background --allexamples
        python -m ibeis_cnn.models.background --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
