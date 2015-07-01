# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import lasagne  # NOQA
from lasagne import layers
from lasagne import nonlinearities
from lasagne import init
import functools
import six
import theano.tensor as T
from ibeis_cnn.models import abstract_models
from ibeis_cnn import custom_layers
import utool as ut
print, rrr, profile = ut.inject2(__name__, '[ibeis_cnn.models.dummy]')


Conv2DLayer = custom_layers.Conv2DLayer
MaxPool2DLayer = custom_layers.MaxPool2DLayer


@six.add_metaclass(ut.ReloadingMetaclass)
class DummyModel(abstract_models.AbstractCategoricalModel):
    def __init__(model, autoinit=False, batch_size=8, input_shape=None, data_shape=(4, 4, 1), **kwargs):
        #if data_shape is not None:
        #    input_shape = (batch_size, data_shape[2], data_shape[0], data_shape[1])
        #if input_shape is None:
        #    input_shape = (batch_size, data_shape[2], data_shape[0], data_shape[1])
        super(DummyModel, model).__init__(input_shape=input_shape, data_shape=data_shape, batch_size=batch_size, **kwargs)
        #model.network_layers = None
        #model.input_shape = input_shape
        model.output_dims = 5
        #model.batch_size = 8
        #model.learning_rate = .001
        #model.momentum = .9
        if autoinit:
            model.initialize_architecture()

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
        network_layers = abstract_models.evaluate_layer_list(network_layers_def)
        #model.network_layers = network_layers
        model.output_layer = network_layers[-1]
        if verbose:
            print('initialize_architecture')
        if ut.VERBOSE:
            model.print_architecture_str()
            model.print_layer_info()
        return model.output_layer


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis_cnn.models.dummy
        python -m ibeis_cnn.models.dummy --allexamples
        python -m ibeis_cnn.models.dummy --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
