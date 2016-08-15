# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import functools
import six
import utool as ut
import ibeis_cnn.__LASAGNE__ as lasagne
from ibeis_cnn.__LASAGNE__ import layers
from ibeis_cnn.__LASAGNE__ import nonlinearities
from ibeis_cnn.__THEANO__ import tensor as T  # NOQA
from ibeis_cnn.models import abstract_models
print, rrr, profile = ut.inject2(__name__, '[ibeis_cnn.models.background]')


class NonlinearityLayerSpatial(lasagne.layers.NonlinearityLayer):
    def __init__(self, incoming, nonlinearity=nonlinearities.rectify,
                 **kwargs):
        """ The spatial version of a nonlinearity as applied accross all spatial
        dimensions of a network's output.
        """
        super(NonlinearityLayerSpatial, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)
        in_batch, in_channels, in_width, in_height = self.input_shape
        self.reshape_required = (in_width == 1 and in_height == 1)

    def get_output_for(self, input, **kwargs):
        old_shape = T.shape(input)
        if self.reshape_required:
            input = T.reshape(input, (-1, old_shape[1]))
            return self.nonlinearity(input)
        elif input.ndim == 4:
            input = input.dimshuffle((0, 3, 2, 1))
            temp = T.shape(input)
            input = T.reshape(input, (-1, old_shape[1]))
            activation = self.nonlinearity(input)
            activation = T.reshape(activation, temp)
            activation = activation.dimshuffle((0, 3, 2, 1))  # Transpose
            return activation
        else:
            _super = super(NonlinearityLayerSpatial, self)
            return _super.get_output_for(input, **kwargs)

    def get_output_shape_for(self, input_shape):
        if self.reshape_required:
            return input_shape[:2]
        else:
            _super = super(NonlinearityLayerSpatial, self)
            return _super.get_output_shape_for(input_shape)


@six.add_metaclass(ut.ReloadingMetaclass)
class BackgroundModel(abstract_models.AbstractCategoricalModel):
    def __init__(model, autoinit=False, batch_size=128, data_shape=(48, 48, 3), num_output=2, **kwargs):
        model.num_output = num_output
        super(BackgroundModel, model).__init__(batch_size=batch_size, data_shape=data_shape, name='background', **kwargs)

    def learning_rate_update(model, x):
        return x / 2.0

    def learning_rate_shock(model, x):
        return x * 2.0

    def augment(model, Xb, yb=None):
        import random
        import cv2
        for index, X in enumerate(Xb):
            if random.uniform(0.0, 1.0) <= 0.5:
                Xb[index] = cv2.flip(X, 1)
        return Xb, yb

    def get_background_def(model, verbose=ut.VERBOSE, **kwargs):
        # _CaffeNet = abstract_models.PretrainedNetwork('caffenet')
        _P = functools.partial

        hidden_initkw = {
            'nonlinearity' : nonlinearities.LeakyRectify(leakiness=(1. / 10.))
        }

        from ibeis_cnn import custom_layers

        Conv2DLayer = custom_layers.Conv2DLayer
        MaxPool2DLayer = custom_layers.MaxPool2DLayer
        #DenseLayer = layers.DenseLayer

        network_layers_def = (
            [
                _P(layers.InputLayer, shape=model.input_shape),

                _P(Conv2DLayer, num_filters=16, filter_size=(11, 11), name='C0', **hidden_initkw),
                _P(layers.DropoutLayer, p=0.1, name='D0'),
                _P(MaxPool2DLayer, pool_size=(2, 2), stride=(2, 2), name='P0'),

                _P(Conv2DLayer, num_filters=32, filter_size=(5, 5), name='C1', **hidden_initkw),
                _P(layers.DropoutLayer, p=0.2, name='D1'),
                _P(MaxPool2DLayer, pool_size=(2, 2), stride=(2, 2), name='P1'),

                _P(Conv2DLayer, num_filters=64, filter_size=(3, 3), name='C2', **hidden_initkw),
                _P(layers.DropoutLayer, p=0.3, name='D2'),
                _P(MaxPool2DLayer, pool_size=(2, 2), stride=(2, 2), name='P2'),

                _P(Conv2DLayer, num_filters=128, filter_size=(3, 3), name='C4', **hidden_initkw),
                _P(layers.DropoutLayer, p=0.4, name='D4'),

                _P(layers.NINLayer, num_units=model.num_output, name='F3', nonlinearity=None),

                _P(NonlinearityLayerSpatial, name='S0', nonlinearity=nonlinearities.softmax),
            ]
        )
        return network_layers_def

    def initialize_architecture(model, verbose=ut.VERBOSE, **kwargs):
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
