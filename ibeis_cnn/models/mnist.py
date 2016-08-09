# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import functools
from ibeis_cnn.models import abstract_models
import utool as ut
print, rrr, profile = ut.inject2(__name__, '[ibeis_cnn.models.dummy]')


class MNISTModel(abstract_models.AbstractCategoricalModel):
    """
    Toy model for testing and playing with mnist

    CommandLine:
        python -m ibeis_cnn.models.mnist MNISTModel:0
        python -m ibeis_cnn.models.mnist MNISTModel:1

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_cnn.models.mnist import *  # NOQA
        >>> from ibeis_cnn import ingest_data
        >>> dataset = ingest_data.grab_mnist_category_dataset()
        >>> model = MNISTModel(batch_size=128, data_shape=dataset.data_shape,
        >>>                    output_dims=dataset.output_dims,
        >>>                    training_dpath=dataset.training_dpath)
        >>> output_layer = model.initialize_architecture()
        >>> model.print_dense_architecture_str()
        >>> model.mode = 'FAST_COMPILE'
        >>> model.build_backprop_func()

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_cnn.models.mnist import *  # NOQA
        >>> from ibeis_cnn import ingest_data
        >>> dataset = ingest_data.grab_mnist_category_dataset()
        >>> model = MNISTModel(batch_size=128, data_shape=dataset.data_shape,
        >>>                    output_dims=dataset.output_dims,
        >>>                    arch_tag='mnist_test1',
        >>>                    training_dpath=dataset.training_dpath)
        >>> model.encoder = None
        >>> model.train_config['monitor'] = True
        >>> model.learning_state['weight_decay'] = None
        >>> model.print_architecture_str()
        >>> model.learning_rate = .01
        >>> output_layer = model.initialize_architecture()
        >>> # parse training arguments
        >>> model.train_config.update(**ut.argparse_dict(dict(
        >>>     era_schedule=100,
        >>>     max_epochs=5,
        >>>     learning_rate_adjust=.8,
        >>> )))
        >>> X_train, y_train = dataset.load_subset('train')
        >>> model.fit(X_train, y_train)

    """
    def __init__(self, **kwargs):
        super(MNISTModel, self).__init__(**kwargs)

    def get_mnist_model_def1(model):
        """
        Follows https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py
        """
        from ibeis_cnn.__LASAGNE__ import init
        from ibeis_cnn.__LASAGNE__ import layers
        from ibeis_cnn.__LASAGNE__ import nonlinearities

        _P = functools.partial
        #leaky = dict(nonlinearity=nonlinearities.LeakyRectify(leakiness=(1. / 10.)))
        leaky = dict(nonlinearity=nonlinearities.rectify)
        #orthog = dict(W=init.Orthogonal())
        #weight_initkw = dict(W=init.GlorotUniform())
        weight_initkw = dict()
        output_initkw = weight_initkw
        hidden_initkw = ut.merge_dicts(weight_initkw, leaky)

        from ibeis_cnn import custom_layers

        Conv2DLayer = custom_layers.Conv2DLayer
        MaxPool2DLayer = custom_layers.MaxPool2DLayer

        network_layers_def = [
            _P(layers.InputLayer, shape=model.input_shape, name='I0'),
            #_P(layers.GaussianNoiseLayer, name='N0'),

            # Convolutional layer with 32 kernels of size 5x5 and 2x2 pooling
            _P(Conv2DLayer, num_filters=32, filter_size=(5, 5), stride=(1, 1),
               name='C1', W=init.GlorotUniform(),
               **hidden_initkw),
            _P(MaxPool2DLayer, pool_size=(2, 2), stride=(2, 2), name='P1'),

            # Another convolution with 32 5x5 kernels, and 2x2 pooling
            _P(Conv2DLayer, num_filters=32, filter_size=(5, 5), stride=(1, 1),
               name='C2', **hidden_initkw),
            _P(MaxPool2DLayer, pool_size=(2, 2), stride=(2, 2), name='P2'),

            # A fully-connected layer of 256 units with 50% dropout on its inputs
            _P(layers.DropoutLayer, p=0.5, name='D3'),
            _P(layers.DenseLayer, num_units=256, name='F3',  **hidden_initkw),

            # And, finally, the 10-unit output layer with 50% dropout on its inputs
            _P(layers.DropoutLayer, p=0.5, name='D4'),
            _P(layers.DenseLayer, num_units=model.output_dims,
               nonlinearity=nonlinearities.softmax, name='O4', **output_initkw),
        ]
        return network_layers_def

    def initialize_architecture(model):
        """

        CommandLine:
            python -m ibeis_cnn --tf  MNISTModel.initialize_architecture --verbcnn --show

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis_cnn.models.mnist import *  # NOQA
            >>> verbose = True
            >>> model = MNISTModel(batch_size=128, data_shape=(28, 28, 1), output_dims=9)
            >>> model.initialize_architecture()
            >>> model.print_dense_architecture_str()
            >>> print(model)
            >>> ut.quit_if_noshow()
            >>> model.show_architecture_image()
            >>> ut.show_if_requested()
        """
        print('[model] initialize_architecture')
        if True:
            print('[model] Initialize MNIST model architecture')
            print('[model]   * batch_size     = %r' % (model.batch_size,))
            print('[model]   * input_width    = %r' % (model.input_width,))
            print('[model]   * input_height   = %r' % (model.input_height,))
            print('[model]   * input_channels = %r' % (model.input_channels,))
            print('[model]   * output_dims    = %r' % (model.output_dims,))
        network_layers_def = model.get_mnist_model_def1()
        network_layers = abstract_models.evaluate_layer_list(network_layers_def)
        model.output_layer = network_layers[-1]
        return model.output_layer


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
