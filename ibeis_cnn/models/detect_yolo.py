# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import six
from ibeis_cnn.__LASAGNE__ import layers
from ibeis_cnn.__LASAGNE__ import nonlinearities
from ibeis_cnn import custom_layers
from ibeis_cnn.models import abstract_models
import utool as ut
print, rrr, profile = ut.inject2(__name__, '[ibeis_cnn.models.detect_yolo]')


Conv2DLayer = custom_layers.Conv2DLayer
MaxPool2DLayer = custom_layers.MaxPool2DLayer


class DetectionLayer(layers.Layer):
    def __init__(self, incoming, num, side, classes, coords=4, softmax=False,
                 sqrt=True, **kwargs):
        super(DetectionLayer, self).__init__(incoming, **kwargs)

    def get_output_for(self, input, **kwargs):
        return input

    def get_output_shape_for(self, input_shape):
        return input_shape


@six.add_metaclass(ut.ReloadingMetaclass)
class DetectYoloModel(abstract_models.AbstractCategoricalModel):
    def __init__(model, autoinit=True, batch_size=128, data_shape=(448, 448, 3), arch_tag='detect_yolo', **kwargs):
        super(DetectYoloModel, model).__init__(batch_size=batch_size, data_shape=data_shape, arch_tag=arch_tag, **kwargs)
        if autoinit:
            model.initialize_architecture()
        if model.preproc_kw is None:
            model.preproc_kw = {
                'center_mean' : 1.0,
                'center_std'  : 2.0,
            }

    def initialize_architecture(model):

        (_, input_channels, input_width, input_height) = model.input_shape

        _Yolo = abstract_models.PretrainedNetwork('detect_yolo')
        leakiness = 0.1

        l_in = layers.InputLayer(
            shape=model.input_shape,
        )

        l_conv0 = Conv2DLayer(
            l_in,
            num_filters=64,
            filter_size=(7, 7),
            stride=(2, 2),
            pad=(3, 3),
            nonlinearity=nonlinearities.LeakyRectify(leakiness=leakiness),
            W=_Yolo.get_pretrained_layer(0),
            b=_Yolo.get_pretrained_layer(1),
        )

        l_pool0 = MaxPool2DLayer(
            l_conv0,
            pool_size=(2, 2),
            stride=(2, 2),
        )

        l_conv1 = Conv2DLayer(
            l_pool0,
            num_filters=192,
            filter_size=(3, 3),
            pad=(1, 1),
            nonlinearity=nonlinearities.LeakyRectify(leakiness=leakiness),
            W=_Yolo.get_pretrained_layer(2),
            b=_Yolo.get_pretrained_layer(3),
        )

        l_pool1 = MaxPool2DLayer(
            l_conv1,
            pool_size=(2, 2),
            stride=(2, 2),
        )

        l_conv2 = Conv2DLayer(
            l_pool1,
            num_filters=128,
            filter_size=(1, 1),
            nonlinearity=nonlinearities.LeakyRectify(leakiness=leakiness),
            W=_Yolo.get_pretrained_layer(4),
            b=_Yolo.get_pretrained_layer(5),
        )

        l_conv3 = Conv2DLayer(
            l_conv2,
            num_filters=256,
            filter_size=(3, 3),
            pad=(1, 1),
            nonlinearity=nonlinearities.LeakyRectify(leakiness=leakiness),
            W=_Yolo.get_pretrained_layer(6),
            b=_Yolo.get_pretrained_layer(7),
        )

        l_conv4 = Conv2DLayer(
            l_conv3,
            num_filters=256,
            filter_size=(1, 1),
            nonlinearity=nonlinearities.LeakyRectify(leakiness=leakiness),
            W=_Yolo.get_pretrained_layer(8),
            b=_Yolo.get_pretrained_layer(9),
        )

        l_conv5 = Conv2DLayer(
            l_conv4,
            num_filters=512,
            filter_size=(3, 3),
            pad=(1, 1),
            nonlinearity=nonlinearities.LeakyRectify(leakiness=leakiness),
            W=_Yolo.get_pretrained_layer(10),
            b=_Yolo.get_pretrained_layer(11),
        )

        l_pool2 = MaxPool2DLayer(
            l_conv5,
            pool_size=(2, 2),
            stride=(2, 2),
        )

        l_conv6 = Conv2DLayer(
            l_pool2,
            num_filters=256,
            filter_size=(1, 1),
            nonlinearity=nonlinearities.LeakyRectify(leakiness=leakiness),
            W=_Yolo.get_pretrained_layer(12),
            b=_Yolo.get_pretrained_layer(13),
        )

        l_conv7 = Conv2DLayer(
            l_conv6,
            num_filters=512,
            filter_size=(3, 3),
            pad=(1, 1),
            nonlinearity=nonlinearities.LeakyRectify(leakiness=leakiness),
            W=_Yolo.get_pretrained_layer(14),
            b=_Yolo.get_pretrained_layer(15),
        )

        l_conv8 = Conv2DLayer(
            l_conv7,
            num_filters=256,
            filter_size=(1, 1),
            nonlinearity=nonlinearities.LeakyRectify(leakiness=leakiness),
            W=_Yolo.get_pretrained_layer(16),
            b=_Yolo.get_pretrained_layer(17),
        )

        l_conv9 = Conv2DLayer(
            l_conv8,
            num_filters=512,
            filter_size=(3, 3),
            pad=(1, 1),
            nonlinearity=nonlinearities.LeakyRectify(leakiness=leakiness),
            W=_Yolo.get_pretrained_layer(18),
            b=_Yolo.get_pretrained_layer(19),
        )

        l_conv10 = Conv2DLayer(
            l_conv9,
            num_filters=256,
            filter_size=(1, 1),
            nonlinearity=nonlinearities.LeakyRectify(leakiness=leakiness),
            W=_Yolo.get_pretrained_layer(20),
            b=_Yolo.get_pretrained_layer(21),
        )

        l_conv11 = Conv2DLayer(
            l_conv10,
            num_filters=512,
            filter_size=(3, 3),
            pad=(1, 1),
            nonlinearity=nonlinearities.LeakyRectify(leakiness=leakiness),
            W=_Yolo.get_pretrained_layer(22),
            b=_Yolo.get_pretrained_layer(23),
        )

        l_conv12 = Conv2DLayer(
            l_conv11,
            num_filters=256,
            filter_size=(1, 1),
            nonlinearity=nonlinearities.LeakyRectify(leakiness=leakiness),
            W=_Yolo.get_pretrained_layer(24),
            b=_Yolo.get_pretrained_layer(25),
        )

        l_conv13 = Conv2DLayer(
            l_conv12,
            num_filters=512,
            filter_size=(3, 3),
            pad=(1, 1),
            nonlinearity=nonlinearities.LeakyRectify(leakiness=leakiness),
            W=_Yolo.get_pretrained_layer(26),
            b=_Yolo.get_pretrained_layer(27),
        )

        l_conv14 = Conv2DLayer(
            l_conv13,
            num_filters=512,
            filter_size=(1, 1),
            nonlinearity=nonlinearities.LeakyRectify(leakiness=leakiness),
            W=_Yolo.get_pretrained_layer(28),
            b=_Yolo.get_pretrained_layer(29),
        )

        l_conv15 = Conv2DLayer(
            l_conv14,
            num_filters=1024,
            filter_size=(3, 3),
            pad=(1, 1),
            nonlinearity=nonlinearities.LeakyRectify(leakiness=leakiness),
            W=_Yolo.get_pretrained_layer(30),
            b=_Yolo.get_pretrained_layer(31),
        )

        l_pool3 = MaxPool2DLayer(
            l_conv15,
            pool_size=(2, 2),
            stride=(2, 2),
        )

        l_conv16 = Conv2DLayer(
            l_pool3,
            num_filters=512,
            filter_size=(1, 1),
            nonlinearity=nonlinearities.LeakyRectify(leakiness=leakiness),
            W=_Yolo.get_pretrained_layer(32),
            b=_Yolo.get_pretrained_layer(33),
        )

        l_conv17 = Conv2DLayer(
            l_conv16,
            num_filters=1024,
            filter_size=(3, 3),
            pad=(1, 1),
            nonlinearity=nonlinearities.LeakyRectify(leakiness=leakiness),
            W=_Yolo.get_pretrained_layer(34),
            b=_Yolo.get_pretrained_layer(35),
        )

        l_conv18 = Conv2DLayer(
            l_conv17,
            num_filters=512,
            filter_size=(1, 1),
            nonlinearity=nonlinearities.LeakyRectify(leakiness=leakiness),
            W=_Yolo.get_pretrained_layer(36),
            b=_Yolo.get_pretrained_layer(37),
        )

        l_conv19 = Conv2DLayer(
            l_conv18,
            num_filters=1024,
            filter_size=(3, 3),
            pad=(1, 1),
            nonlinearity=nonlinearities.LeakyRectify(leakiness=leakiness),
            W=_Yolo.get_pretrained_layer(38),
            b=_Yolo.get_pretrained_layer(39),
        )

        l_conv20 = Conv2DLayer(
            l_conv19,
            num_filters=1024,
            filter_size=(3, 3),
            pad=(1, 1),
            nonlinearity=nonlinearities.LeakyRectify(leakiness=leakiness),
            W=_Yolo.get_pretrained_layer(40),
            b=_Yolo.get_pretrained_layer(41),
        )

        l_conv21 = Conv2DLayer(
            l_conv20,
            num_filters=1024,
            filter_size=(3, 3),
            stride=(2, 2),
            pad=(1, 1),
            nonlinearity=nonlinearities.LeakyRectify(leakiness=leakiness),
            W=_Yolo.get_pretrained_layer(42),
            b=_Yolo.get_pretrained_layer(43),
        )

        l_conv22 = Conv2DLayer(
            l_conv21,
            num_filters=1024,
            filter_size=(3, 3),
            pad=(1, 1),
            nonlinearity=nonlinearities.LeakyRectify(leakiness=leakiness),
            W=_Yolo.get_pretrained_layer(44),
            b=_Yolo.get_pretrained_layer(45),
        )

        l_conv23 = Conv2DLayer(
            l_conv22,
            num_filters=1024,
            filter_size=(3, 3),
            pad=(1, 1),
            nonlinearity=nonlinearities.LeakyRectify(leakiness=leakiness),
            W=_Yolo.get_pretrained_layer(46),
            b=_Yolo.get_pretrained_layer(47),
        )

        l_hidden0 = layers.DenseLayer(
            l_conv23,
            num_units=4096,
            nonlinearity=nonlinearities.LeakyRectify(leakiness=leakiness),
            W=_Yolo.get_pretrained_layer(48),
            b=_Yolo.get_pretrained_layer(49),
        )

        l_dropout0 = layers.DropoutLayer(
            l_hidden0,
            p=0.5,
        )

        l_hidden1 = layers.DenseLayer(
            l_dropout0,
            num_units=735,
            nonlinearity=nonlinearities.linear,
            W=_Yolo.get_pretrained_layer(50),
            b=_Yolo.get_pretrained_layer(51),
        )

        l_out = DetectionLayer(
            l_hidden1,
            num=2,
            side=7,
            classes=5,
        )

        model.output_layer = l_out
        return l_out


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis_cnn.models.detect_yolo
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
