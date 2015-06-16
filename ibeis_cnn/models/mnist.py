# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import lasagne  # NOQA
from lasagne import layers
from lasagne import nonlinearities
from lasagne import init
import functools
from ibeis_cnn.models import abstract_models
from ibeis_cnn import custom_layers
import utool as ut
print, rrr, profile = ut.inject2(__name__, '[ibeis_cnn.models.dummy]')


Conv2DLayer = custom_layers.Conv2DLayer
MaxPool2DLayer = custom_layers.MaxPool2DLayer


class MNISTModel(abstract_models.AbstractCategoricalModel):
    """
    Toy model for testing and playing with mnist
    """
    def __init__(self, **kwargs):
        super(MNISTModel, self).__init__(**kwargs)

    def get_mnist_model_def1(model, input_shape, output_dims):
        _P = functools.partial
        leaky = dict(nonlinearity=nonlinearities.LeakyRectify(leakiness=(1. / 10.)))
        #orthog = dict(W=init.Orthogonal())
        weight_initkw = dict()
        output_initkw = weight_initkw
        hidden_initkw = ut.merge_dicts(weight_initkw, leaky)

        network_layers_def = (
            [
                _P(layers.InputLayer, shape=input_shape),
                layers.GaussianNoiseLayer,

                _P(Conv2DLayer, num_filters=32, filter_size=(5, 5), stride=(1, 1), name='C0', **hidden_initkw),
                _P(layers.DropoutLayer, p=0.1),
                _P(MaxPool2DLayer, pool_size=(2, 2), stride=(2, 2), name='P0'),

                _P(Conv2DLayer, num_filters=32, filter_size=(5, 5), stride=(1, 1), name='C1', **hidden_initkw),
                _P(MaxPool2DLayer, pool_size=(2, 2), stride=(2, 2), name='P1'),

                _P(layers.DenseLayer, num_units=256, name='F1',  **hidden_initkw),
                _P(layers.FeaturePoolLayer, pool_size=2),  # maxout
                _P(layers.DropoutLayer, p=0.5),

                _P(layers.DenseLayer, num_units=output_dims,
                   nonlinearity=nonlinearities.softmax, **output_initkw)
            ]
        )
        return network_layers_def

    def get_mnist_model_def_failure(model, input_shape, output_dims):
        # causes failure in building model
        _P = functools.partial
        leaky = dict(nonlinearity=nonlinearities.LeakyRectify(leakiness=(1. / 10.)))
        orthog = dict(W=init.Orthogonal())
        hidden_initkw = ut.merge_dicts(orthog, leaky)
        initkw = hidden_initkw
        #initkw = {}

        # def to test failures
        network_layers_def = (
            [
                _P(layers.InputLayer, shape=input_shape),
                #layers.GaussianNoiseLayer,

                _P(Conv2DLayer, num_filters=16, filter_size=(7, 7), stride=(1, 1), name='C0', **initkw),
                _P(MaxPool2DLayer, pool_size=(3, 3), stride=(3, 3), name='P0'),
                _P(Conv2DLayer, num_filters=16, filter_size=(3, 3), stride=(1, 1), name='C1', **initkw),
                _P(Conv2DLayer, num_filters=16, filter_size=(3, 3), stride=(1, 1), name='C2', **initkw),
                _P(Conv2DLayer, num_filters=16, filter_size=(3, 3), stride=(1, 1), name='C3', **initkw),
                _P(Conv2DLayer, num_filters=16, filter_size=(2, 2), stride=(1, 1), name='C4', **initkw),
                _P(Conv2DLayer, num_filters=16, filter_size=(1, 1), stride=(1, 1), name='C5', **initkw),
                _P(Conv2DLayer, num_filters=16, filter_size=(2, 2), stride=(1, 1), name='C6', **initkw),

                _P(layers.DenseLayer, num_units=32, name='F1',  **initkw),
                _P(layers.DenseLayer, num_units=output_dims, nonlinearity=nonlinearities.softmax)
            ]
        )
        return network_layers_def

    def initialize_architecture(self):
        input_shape = self.input_shape
        output_dims = self.output_dims
        network_layers_def = self.get_mnist_model_def1(input_shape, output_dims)
        network_layers = abstract_models.evaluate_layer_list(network_layers_def)
        self.output_layer = network_layers[-1]
        return self.output_layer


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis_cnn.models.mnist
        python -m ibeis_cnn.models.mnist --allexamples
        python -m ibeis_cnn.models.mnist --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
