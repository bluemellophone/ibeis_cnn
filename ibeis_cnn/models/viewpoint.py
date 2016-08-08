# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import six
import random
from ibeis_cnn.__LASAGNE__ import layers
from ibeis_cnn.__LASAGNE__ import nonlinearities
# from ibeis_cnn.__LASAGNE__ import init
from ibeis_cnn.models import abstract_models
import utool as ut
print, rrr, profile = ut.inject2(__name__, '[ibeis_cnn.models.viewpoint]')


@six.add_metaclass(ut.ReloadingMetaclass)
class ViewpointModel(abstract_models.AbstractCategoricalModel):
    def __init__(model, autoinit=False, batch_size=128, data_shape=(96, 96, 3), arch_tag='viewpoint', **kwargs):
        super(ViewpointModel, model).__init__(batch_size=batch_size, data_shape=data_shape, arch_tag=arch_tag, **kwargs)
        if autoinit:
            model.initialize_architecture()

    def augment(model, Xb, yb=None):
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
                if yb is not None:
                    yb[index] = _invert_label(yb[index])
        return Xb, yb

    def label_order_mapping(model, category_list):
        r"""
        Args:
            category_list (list):

        Returns:
            ?: category_mapping

        CommandLine:
            python -m ibeis_cnn.models.viewpoint --exec-label_order_mapping

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis_cnn.models.viewpoint import *  # NOQA
            >>> model = ViewpointModel()
            >>> category_list = ['LEFT', 'FRONT_LEFT', 'FRONT', 'FRONT_RIGHT', 'RIGHT', 'BACK_RIGHT', 'BACK', 'BACK_LEFT']
            >>> category_mapping = model.label_order_mapping(category_list)
            >>> result = ('category_mapping = %s' % (str(category_mapping),))
            >>> print(result)
        """
        if len(category_list) == 8:
            species_list = [
                'ZEBRA_PLAINS',
            ]
        else:
            species_list = [
                'ZEBRA_PLAINS',
                'ZEBRA_GREVYS',
                'ELEPHANT_SAVANNA',
                'GIRAFFE_RETICULATED',
                'GIRAFFE_MASAI',
            ]
        viewpoint_mapping = {
            'LEFT':        0,
            'FRONT_LEFT':  1,
            'FRONT':       2,
            'FRONT_RIGHT': 3,
            'RIGHT':       4,
            'BACK_RIGHT':  5,
            'BACK':        6,
            'BACK_LEFT':   7,
        }
        viewpoint_mapping = {
        }
        viewpoints = len(viewpoint_mapping.keys())
        category_mapping = {}
        for index, species in enumerate(species_list):
            for viewpoint, value in six.iteritems(viewpoint_mapping):
                key = '%s:%s' % (species, viewpoint, )
                base = viewpoints * index
                category_mapping[key] = base + value
        return category_mapping

    def learning_rate_update(model, x):
        return x / 2.0

    def learning_rate_shock(model, x):
        return x * 2.0

    #def build_model(model, batch_size, input_width, input_height, input_channels, output_dims):
    def initialize_architecture(model):

        from ibeis_cnn import custom_layers
        Conv2DLayer = custom_layers.Conv2DLayer
        MaxPool2DLayer = custom_layers.MaxPool2DLayer

        (_, input_channels, input_width, input_height) = model.input_shape
        output_dims = model.output_dims

        _CaffeNet = abstract_models.PretrainedNetwork('caffenet')

        l_in = layers.InputLayer(
            # variable batch size (None), channel, width, height
            #shape=(None, input_channels, input_width, input_height)
            shape=model.input_shape,
        )

        l_noise = layers.GaussianNoiseLayer(
            l_in,
        )

        l_conv0 = Conv2DLayer(
            l_noise,
            num_filters=32,
            filter_size=(11, 11),
            # nonlinearity=nonlinearities.rectify,
            nonlinearity=nonlinearities.LeakyRectify(leakiness=(1. / 10.)),
            W=_CaffeNet.get_pretrained_layer(0),
        )

        l_conv0_dropout = layers.DropoutLayer(l_conv0, p=0.10)

        l_conv1 = Conv2DLayer(
            l_conv0_dropout,
            num_filters=32,
            filter_size=(5, 5),
            # nonlinearity=nonlinearities.rectify,
            nonlinearity=nonlinearities.LeakyRectify(leakiness=(1. / 10.)),
            W=_CaffeNet.get_pretrained_layer(2),
        )

        l_pool1 = MaxPool2DLayer(
            l_conv1,
            pool_size=(2, 2),
            stride=(2, 2),
        )

        l_conv2_dropout = layers.DropoutLayer(l_pool1, p=0.10)

        l_conv2 = Conv2DLayer(
            l_conv2_dropout,
            num_filters=64,
            filter_size=(3, 3),
            # nonlinearity=nonlinearities.rectify,
            nonlinearity=nonlinearities.LeakyRectify(leakiness=(1. / 10.)),
            # W=init.Orthogonal(),
        )

        l_pool2 = MaxPool2DLayer(
            l_conv2,
            pool_size=(2, 2),
            stride=(2, 2),
        )

        l_conv3_dropout = layers.DropoutLayer(l_pool2, p=0.30)

        l_conv3 = Conv2DLayer(
            l_conv3_dropout,
            num_filters=128,
            filter_size=(3, 3),
            # nonlinearity=nonlinearities.rectify,
            nonlinearity=nonlinearities.LeakyRectify(leakiness=(1. / 10.)),
            # W=init.Orthogonal(),
        )

        l_pool3 = MaxPool2DLayer(
            l_conv3,
            pool_size=(2, 2),
            stride=(2, 2),
        )

        l_conv4_dropout = layers.DropoutLayer(l_pool3, p=0.30)

        l_conv4 = Conv2DLayer(
            l_conv4_dropout,
            num_filters=128,
            filter_size=(3, 3),
            # nonlinearity=nonlinearities.rectify,
            nonlinearity=nonlinearities.LeakyRectify(leakiness=(1. / 10.)),
            # W=init.Orthogonal(),
        )

        l_pool4 = MaxPool2DLayer(
            l_conv4,
            pool_size=(2, 2),
            stride=(2, 2),
        )

        l_hidden1 = layers.DenseLayer(
            l_pool4,
            num_units=512,
            # nonlinearity=nonlinearities.rectify,
            nonlinearity=nonlinearities.LeakyRectify(leakiness=(1. / 10.)),
            # W=init.Orthogonal(),
        )

        l_hidden1_maxout = layers.FeaturePoolLayer(
            l_hidden1,
            pool_size=2,
        )

        l_hidden1_dropout = layers.DropoutLayer(l_hidden1_maxout, p=0.5)

        l_hidden2 = layers.DenseLayer(
            l_hidden1_dropout,
            num_units=512,
            # nonlinearity=nonlinearities.rectify,
            nonlinearity=nonlinearities.LeakyRectify(leakiness=(1. / 10.)),
            # W=init.Orthogonal(),
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
            # W=init.Orthogonal(),
        )

        model.output_layer = l_out
        return l_out


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
