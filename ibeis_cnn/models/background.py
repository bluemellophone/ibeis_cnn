# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import lasagne  # NOQA
from lasagne import layers
from lasagne import nonlinearities
from ibeis_cnn import custom_layers
from ibeis_cnn.models import abstract_models
import functools
import six
import utool as ut
print, rrr, profile = ut.inject2(__name__, '[ibeis_cnn.models.background]')


Conv2DLayer = custom_layers.Conv2DLayer
MaxPool2DLayer = custom_layers.MaxPool2DLayer
DenseLayer = layers.DenseLayer


@six.add_metaclass(ut.ReloadingMetaclass)
class BackgroundModel(abstract_models.AbstractCategoricalModel):
    def __init__(model, autoinit=False, batch_size=128, data_shape=(96, 96, 3), arch_tag='background', **kwargs):
        super(BackgroundModel, model).__init__(batch_size=batch_size, data_shape=data_shape, arch_tag=arch_tag, **kwargs)

    def learning_rate_update(model, x):
        return x / 2.0

    def learning_rate_shock(model, x):
        return x * 2.0

    # def augment(model, Xb, yb=None):
    #     import random
    #     import cv2
    #     for index, X in enumerate(Xb):
    #         if random.uniform(0.0, 1.0) <= 0.5:
    #             Xb[index] = cv2.flip(X, 1)
    #     return Xb, yb

    def get_background_def(model, verbose=ut.VERBOSE, **kwargs):
        # _CaffeNet = abstract_models.PretrainedNetwork('caffenet')
        _P = functools.partial

        hidden_initkw = {
            'nonlinearity' : nonlinearities.LeakyRectify(leakiness=(1. / 10.))
        }

        network_layers_def = (
            [
                _P(layers.InputLayer, shape=model.input_shape),

                _P(Conv2DLayer, num_filters=16, filter_size=(11, 11), name='C0', **hidden_initkw),
                _P(MaxPool2DLayer, pool_size=(2, 2), stride=(2, 2), name='P0'),
                _P(layers.DropoutLayer, p=0.1, name='D0'),

                _P(Conv2DLayer, num_filters=32, filter_size=(5, 5), name='C1', **hidden_initkw),
                _P(MaxPool2DLayer, pool_size=(2, 2), stride=(2, 2), name='P1'),
                _P(layers.DropoutLayer, p=0.2, name='D1'),

                _P(Conv2DLayer, num_filters=64, filter_size=(3, 3), name='C2', **hidden_initkw),
                _P(layers.DropoutLayer, p=0.3, name='D2'),

                _P(Conv2DLayer, num_filters=64, filter_size=(3, 3), name='C3', **hidden_initkw),
                _P(MaxPool2DLayer, pool_size=(2, 2), stride=(2, 2), name='P2'),
                _P(layers.DropoutLayer, p=0.5, name='D3'),

                _P(Conv2DLayer, num_filters=128, filter_size=(3, 3), name='C4', **hidden_initkw),
                _P(layers.DropoutLayer, p=0.5, name='D4'),

                _P(Conv2DLayer, num_filters=128, filter_size=(3, 3), name='C5', **hidden_initkw),
                _P(layers.DropoutLayer, p=0.5, name='D5'),

                _P(DenseLayer, num_units=1024, name='F1', **hidden_initkw),
                _P(layers.FeaturePoolLayer, pool_size=2),
                _P(layers.DropoutLayer, p=0.5),

                _P(DenseLayer, num_units=1024, name='F2', **hidden_initkw),
                _P(layers.FeaturePoolLayer, pool_size=2),
                _P(layers.DropoutLayer, p=0.5),

                _P(DenseLayer, num_units=2, name='F3', nonlinearity=nonlinearities.softmax),
            ]
        )
        return network_layers_def

    def get_background_fcnn_def(model, verbose=ut.VERBOSE, **kwargs):
        # _CaffeNet = abstract_models.PretrainedNetwork('caffenet')
        _P = functools.partial

        hidden_initkw = {
            'nonlinearity' : nonlinearities.LeakyRectify(leakiness=(1. / 10.))
        }

        network_layers_def = (
            [
                _P(layers.InputLayer, shape=model.input_shape),

                _P(Conv2DLayer, num_filters=16, filter_size=(11, 11), name='C0', **hidden_initkw),
                _P(MaxPool2DLayer, pool_size=(2, 2), stride=(2, 2), name='P0'),
                _P(layers.DropoutLayer, p=0.1, name='D0'),

                _P(Conv2DLayer, num_filters=32, filter_size=(5, 5), name='C1', **hidden_initkw),
                _P(MaxPool2DLayer, pool_size=(2, 2), stride=(2, 2), name='P1'),
                _P(layers.DropoutLayer, p=0.2, name='D1'),

                _P(Conv2DLayer, num_filters=64, filter_size=(3, 3), name='C2', **hidden_initkw),
                _P(layers.DropoutLayer, p=0.3, name='D2'),

                _P(Conv2DLayer, num_filters=64, filter_size=(3, 3), name='C3', **hidden_initkw),
                _P(MaxPool2DLayer, pool_size=(2, 2), stride=(2, 2), name='P2'),
                _P(layers.DropoutLayer, p=0.5, name='D3'),

                _P(Conv2DLayer, num_filters=128, filter_size=(3, 3), name='C4', **hidden_initkw),
                _P(layers.DropoutLayer, p=0.5, name='D4'),

                _P(Conv2DLayer, num_filters=128, filter_size=(3, 3), name='C5', **hidden_initkw),
                _P(layers.DropoutLayer, p=0.5, name='D5'),

                _P(Conv2DLayer, num_filters=256, filter_size=(4, 4), name='F1', **hidden_initkw),
                _P(layers.DropoutLayer, p=0.5),

                _P(layers.NINLayer, num_units=1024, name='F2', **hidden_initkw),
                _P(layers.FeaturePoolLayer, pool_size=2),
                _P(layers.DropoutLayer, p=0.5),

                _P(layers.NINLayer, num_units=1024, name='F2', **hidden_initkw),
                _P(layers.FeaturePoolLayer, pool_size=2),
                _P(layers.DropoutLayer, p=0.5),

                _P(layers.NINLayer, num_units=2, name='F3', nonlinearity=nonlinearities.softmax),
            ]
        )
        return network_layers_def

    def initialize_architecture(model, verbose=ut.VERBOSE, fcnn=True, **kwargs):
        r"""
        """
        (_, input_channels, input_width, input_height) = model.input_shape
        if verbose:
            print('[model] Initialize center siamese l2 model architecture')
            print('[model]   * batch_size     = %r' % (model.batch_size,))
            print('[model]   * input_width    = %r' % (input_width,))
            print('[model]   * input_height   = %r' % (input_height,))
            print('[model]   * input_channels = %r' % (input_channels,))
            print('[model]   * output_dims    = %r' % (model.output_dims,))

        if fcnn:
            network_layers_def = model.get_background_fcnn_def(verbose=verbose, **kwargs)
        else:
            network_layers_def = model.get_background_def(verbose=verbose, **kwargs)
        # connect and record layers
        network_layers = abstract_models.evaluate_layer_list(network_layers_def, verbose=verbose)
        #model.network_layers = network_layers
        output_layer = network_layers[-1]
        model.output_layer = output_layer
        return output_layer

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
