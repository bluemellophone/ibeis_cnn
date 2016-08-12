# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from ibeis_cnn.models import abstract_models
import utool as ut
print, rrr, profile = ut.inject2(__name__, '[ibeis_cnn.models.dummy]')


class MNISTModel(abstract_models.AbstractCategoricalModel):
    """
    Toy model for testing and playing with mnist

    CommandLine:
        python -m ibeis_cnn.models.mnist MNISTModel:0
        python -m ibeis_cnn.models.mnist MNISTModel:1

        python -m ibeis_cnn _ModelFitting.fit:0 --vd
        python -m ibeis_cnn _ModelFitting.fit:1 --vd

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_cnn.models.mnist import *  # NOQA
        >>> from ibeis_cnn import ingest_data
        >>> dataset = ingest_data.grab_mnist_category_dataset_float()
        >>> model = MNISTModel(batch_size=128, data_shape=dataset.data_shape,
        >>>                    output_dims=dataset.output_dims,
        >>>                    training_dpath=dataset.training_dpath)
        >>> output_layer = model.initialize_architecture()
        >>> model.print_model_info_str()
        >>> model.mode = 'FAST_COMPILE'
        >>> model.build_backprop_func()

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_cnn.models.mnist import *  # NOQA
        >>> from ibeis_cnn import ingest_data
        >>> dataset = ingest_data.grab_mnist_category_dataset_float()
        >>> model = MNISTModel(batch_size=128, data_shape=dataset.data_shape,
        >>>                    output_dims=len(dataset.unique_labels),
        >>>                    learning_rate=.01,
        >>>                    training_dpath=dataset.training_dpath)
        >>> model.encoder = None
        >>> model.train_config['monitor'] = True
        >>> model.learn_state['weight_decay'] = None
        >>> output_layer = model.initialize_architecture()
        >>> model.print_layer_info()
        >>> # parse training arguments
        >>> model.train_config.update(**ut.argparse_dict(dict(
        >>>     era_size=100,
        >>>     max_epochs=5,
        >>>     rate_decay=.8,
        >>> )))
        >>> X_train, y_train = dataset.load_subset('train')
        >>> model.fit(X_train, y_train)

    """
    def __init__(model, **kwargs):
        model.batch_norm = kwargs.pop('batch_norm', True)
        super(MNISTModel, model).__init__(**kwargs)

    def get_mnist_model_def1(model):
        """
        Follows https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py
        """
        import ibeis_cnn.__LASAGNE__ as lasange
        from ibeis_cnn import custom_layers
        batch_norm = model.batch_norm
        dropout = 0 if batch_norm else .5

        bundles = custom_layers.make_bundles(
            nonlinearity=lasange.nonlinearities.rectify,
            batch_norm=batch_norm,
        )
        InputBundle = bundles['InputBundle']
        ConvPoolBundle = bundles['ConvPoolBundle']
        FullyConnectedBundle = bundles['FullyConnectedBundle']
        SoftmaxBundle = bundles['SoftmaxBundle']

        network_layers_def = [
            InputBundle(shape=model.input_shape, noise=False),
            # Convolutional layer with 32 kernels of size 5x5 and 2x2 pooling
            ConvPoolBundle(num_filters=32, filter_size=(5, 5)),
            # Another convolution with 32 5x5 kernels, and 2x2 pooling
            # with 50% dropout on its outputs
            ConvPoolBundle(num_filters=32, filter_size=(5, 5), dropout=dropout),
            # A fully-connected layer of 256 units and 50% dropout of its outputs
            FullyConnectedBundle(num_units=256, dropout=dropout),
            # And, finally, the 10-unit output layer with 50% dropout on its inputs
            SoftmaxBundle(num_units=model.output_dims),
        ]
        return network_layers_def

    def initialize_architecture(model):
        """

        CommandLine:
            python -m ibeis_cnn --tf  MNISTModel.initialize_architecture --verbcnn
            python -m ibeis_cnn --tf  MNISTModel.initialize_architecture --verbcnn --show

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis_cnn.models.mnist import *  # NOQA
            >>> verbose = True
            >>> model = MNISTModel(batch_size=128, data_shape=(28, 28, 1), output_dims=9)
            >>> model.initialize_architecture()
            >>> model.print_model_info_str()
            >>> print(model)
            >>> ut.quit_if_noshow()
            >>> model.show_arch()
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
