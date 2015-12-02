# -*- coding: utf-8 -*-
"""
Siamese based models

References:
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    https://github.com/BVLC/caffe/pull/959
    http://yann.lecun.com/exdb/publis/pdf/chopra-05.pdf
    http://www.commendo.at/references/files/paperCVWW08.pdf
    https://tspace.library.utoronto.ca/bitstream/1807/43097/3/Liu_Chen_201311_MASc_thesis.pdf
    http://arxiv.org/pdf/1412.6622.pdf
    http://papers.nips.cc/paper/4314-extracting-speaker-specific-information-with-a-regularized-siamese-deep-network.pdf
    http://machinelearning.wustl.edu/mlpapers/paper_files/NIPS2005_265.pdf
    http://vision.ia.ac.cn/zh/senimar/reports/Siamese-Network-Architecture-and-Applications-in-Computer-Vision.pdf

    https://groups.google.com/forum/#!topic/caffe-users/D-7sRDw9v8c
    http://caffe.berkeleyvision.org/gathered/examples/siamese.html
    https://groups.google.com/forum/#!topic/lasagne-users/N9zDNvNkyWY
    http://www.cs.nyu.edu/~sumit/research/research.html
    https://github.com/Lasagne/Lasagne/issues/168
    https://groups.google.com/forum/#!topic/lasagne-users/7JX_8zKfDI0
"""
from __future__ import absolute_import, division, print_function
import lasagne  # NOQA
from lasagne import layers
from lasagne import nonlinearities
from lasagne import init
import functools
import six
#import theano.tensor as T
from ibeis_cnn.__THEANO__ import tensor as T
import numpy as np
from ibeis_cnn.models import abstract_models
from ibeis_cnn import custom_layers
import utool as ut
from ibeis_cnn import augment
print, rrr, profile = ut.inject2(__name__, '[ibeis_cnn.models]')


Conv2DLayer = custom_layers.Conv2DLayer
MaxPool2DLayer = custom_layers.MaxPool2DLayer


@six.add_metaclass(ut.ReloadingMetaclass)
class AbstractSiameseModel(abstract_models.BaseModel):
    def __init__(model, *args, **kwargs):
        super(AbstractSiameseModel, model).__init__(*args, **kwargs)
        # bad name, says that this network will take
        # 2*N images in a batch and N labels that map to
        # two images a piece
        model.data_per_label_input = 2
        model.data_per_label_output = 2

    def augment(model, Xb, yb=None):
        Xb_, yb_ = augment.augment_siamese_patches2(Xb, yb)
        return Xb_, yb_


@six.add_metaclass(ut.ReloadingMetaclass)
class SiameseL2(AbstractSiameseModel):
    """
    Model for individual identification
    """
    def __init__(model, autoinit=False, batch_size=128, data_shape=(64, 64, 3),
                 arch_tag='siaml2', **kwargs):
        #if data_shape is not None:
        #    input_shape = (batch_size, data_shape[2], data_shape[0], data_shape[1])
        #if input_shape is None:
        #    (batch_size, 3, 64, 64)
        super(SiameseL2, model).__init__(batch_size=batch_size,
                                         data_shape=data_shape,
                                         arch_tag=arch_tag, **kwargs)
        #model.network_layers = None
        #model.batch_size = batch_size
        model.output_dims = 1
        model.name = arch_tag
        # bad name, says that this network will take
        # 2*N images in a batch and N labels that map to
        # two images a piece
        model.data_per_label_input = 2
        model.data_per_label_output = 2
        #model.arch_tag = arch_tag
        if autoinit:
            model.initialize_architecture()

    def get_siaml2_def(model, verbose=True, **kwargs):
        """
        Notes:
            (ix) siam-2stream-l2 consists of one central and one surround
                branch of siam-2stream.

                C0(96, 7, 3) - ReLU - P0(2, 2) - C1(192, 5, 1) - ReLU - P1(2, 2) - C2(256, 3, 1)
        """
        _P = functools.partial

        leaky_kw = dict(nonlinearity=nonlinearities.LeakyRectify(leakiness=(1. / 10.)))
        #orthog_kw = dict(W=init.Orthogonal())
        #hidden_initkw = ut.merge_dicts(orthog_kw, leaky_kw)
        hidden_initkw = leaky_kw

        #ReshapeLayer = layers.ReshapeLayer

        network_layers_def = (
            [
                _P(layers.InputLayer, shape=model.input_shape),
                # TODO: Stack Inputs by making a 2 Channel Layer
                #caffenet.get_conv2d_layer(0, trainable=False, **leaky),
                _P(Conv2DLayer, num_filters=96, filter_size=(7, 7), stride=(3, 3), name='C0', **hidden_initkw),
                _P(layers.DropoutLayer, p=0.3),
                _P(MaxPool2DLayer, pool_size=(2, 2), stride=(2, 2), name='P0'),
                _P(Conv2DLayer, num_filters=192, filter_size=(5, 5), name='C1', **hidden_initkw),
                _P(layers.DropoutLayer, p=0.3),
                _P(MaxPool2DLayer, pool_size=(2, 2), stride=(2, 2), name='P1'),
                _P(Conv2DLayer, num_filters=256, filter_size=(3, 3), name='C2', **hidden_initkw),
                #_P(custom_layers.SiameseConcatLayer, axis=1, data_per_label=2, name='concat'),  # 2 when CenterSurroundIsOn but two channel network
                _P(layers.FlattenLayer, outdim=2, name='flatten'),
                #_P(custom_layers.L2NormalizeLayer, axis=2),
                # TODO: L2 distance layer
                #_P(custom_layers.SiameseConcatLayer, data_per_label=2),
            ]
        )
        #raise NotImplementedError('The 2-channel part is not yet implemented')
        return network_layers_def

    def get_siaml2_128_def(model, verbose=True, **kwargs):
        """
        Notes:
            (ix) siam-2stream-l2 consists of one central and one surround
                branch of siam-2stream.

                C0(96, 7, 3) - ReLU - P0(2, 2) - C1(192, 5, 1) - ReLU - P1(2, 2) - C2(256, 3, 1)
        """
        _P = functools.partial

        leaky_kw = dict(nonlinearity=nonlinearities.LeakyRectify(leakiness=(1. / 10.)))
        #orthog_kw = dict(W=init.Orthogonal())
        #hidden_initkw = ut.merge_dicts(orthog_kw, leaky_kw)
        hidden_initkw = leaky_kw

        #ReshapeLayer = layers.ReshapeLayer

        network_layers_def = (
            [
                _P(layers.InputLayer, shape=model.input_shape),
                # TODO: Stack Inputs by making a 2 Channel Layer
                #caffenet.get_conv2d_layer(0, trainable=False, **leaky),
                _P(Conv2DLayer, num_filters=96, filter_size=(7, 7), stride=(3, 3), name='C0', **hidden_initkw),
                _P(layers.DropoutLayer, p=0.3),
                _P(MaxPool2DLayer, pool_size=(2, 2), stride=(2, 2), name='P0'),
                _P(Conv2DLayer, num_filters=192, filter_size=(5, 5), name='C1', **hidden_initkw),
                _P(layers.DropoutLayer, p=0.3),
                _P(MaxPool2DLayer, pool_size=(2, 2), stride=(2, 2), name='P1'),
                _P(Conv2DLayer, num_filters=128, filter_size=(3, 3), name='C2_128', **hidden_initkw),
                #_P(custom_layers.SiameseConcatLayer, axis=1, data_per_label=2, name='concat'),  # 2 when CenterSurroundIsOn but two channel network
                _P(layers.FlattenLayer, outdim=2, name='flatten128'),
                #_P(custom_layers.L2NormalizeLayer, axis=2),
                # TODO: L2 distance layer
                #_P(custom_layers.SiameseConcatLayer, data_per_label=2),
            ]
        )
        #raise NotImplementedError('The 2-channel part is not yet implemented')
        return network_layers_def

    def get_siam_deepfaceish_def(model, verbose=True, **kwargs):
        """
        CommandLine:
            python -m ibeis_cnn --tf  SiameseL2.initialize_architecture --archtag siam_deepfaceish --datashape=128,256,1 --verbose  --show
            python -m ibeis_cnn --tf  SiameseL2.initialize_architecture --archtag siam_deepface --datashape=152,152,3 --verbose  --show
        """
        _P = functools.partial

        leaky_kw = dict(nonlinearity=nonlinearities.LeakyRectify(leakiness=(1. / 10.)))
        #orthog_kw = dict(W=init.Orthogonal())
        #hidden_initkw = ut.merge_dicts(orthog_kw, leaky_kw)
        hidden_initkw = leaky_kw

        #ReshapeLayer = layers.ReshapeLayer

        _tmp = [1]

        def CDP_layer(num_filters=32,
                              conv_size=(5, 5), conv_stride=(3, 3),
                              pool_size=(2, 2), pool_stride=(2, 2),
                              drop_p=0.3):
            num = _tmp[0]
            _tmp[0] += 1
            return [
                _P(Conv2DLayer, num_filters=num_filters, filter_size=conv_size,
                   stride=conv_stride, name='C' + str(num), **hidden_initkw),
                _P(layers.DropoutLayer, p=drop_p, name='D' + str(num)),
                _P(MaxPool2DLayer, pool_size=pool_size, stride=pool_stride, name='P' + str(num)),
            ]

        def CD_layer(num_filters=32,
                     conv_size=(5, 5), conv_stride=(3, 3),
                     drop_p=0.3):
            num = _tmp[0]
            _tmp[0] += 1
            return [
                _P(Conv2DLayer, num_filters=num_filters, filter_size=conv_size,
                   stride=conv_stride, name='C' + str(num), **hidden_initkw),
                _P(layers.DropoutLayer, p=drop_p, name='D' + str(num)),
            ]

        network_layers_def = (
            [
                _P(layers.InputLayer, shape=model.input_shape)
            ]  +
            CDP_layer( 32, (11, 11), (1, 1), (3, 3), (2, 2)) +
            CD_layer( 16, ( 9,  9), (1, 2)) +
            CD_layer( 16, ( 9,  9), (1, 1)) +
            CD_layer( 16, ( 9,  9), (1, 1)) +
            CD_layer( 16, ( 9,  9), (2, 2)) +
            CD_layer( 16, ( 9,  9), (2, 2)) +
            [
                _P(layers.DenseLayer, num_units=128, name='F1',  **hidden_initkw),
                _P(layers.DenseLayer, num_units=64, name='F2',  **hidden_initkw),
                #_P(layers.DenseLayer, num_units=64, name='F3',  **hidden_initkw),
            ] +
            #CD_layer(128, (3, 3), (2, 2)) +
            #CD_layer(96, (2, 2), (1, 1)) +
            #CD_layer(64, (2, 2), (1, 1)) +
            #CD_layer(64, (1, 1), (1, 1)) +
            #CD_layer(64, (2, 1), (2, 2)) +
            [
                #_P(Conv2DLayer, num_filters=128, filter_size=(3, 3), name='C3_128', **hidden_initkw),
                #_P(Conv2DLayer, num_filters=64, filter_size=(3, 3), name='C4_128', **hidden_initkw),
                #_P(Conv2DLayer, num_filters=64, filter_size=(2, 1), stride=(2, 2), name='C4_128', **hidden_initkw),
                #_P(layers.FlattenLayer, outdim=2, name='flatten128'),
                _P(custom_layers.L2NormalizeLayer, axis=2),
            ]
        )
        #raise NotImplementedError('The 2-channel part is not yet implemented')
        return network_layers_def

    def get_siaml2_partmatch_def(model, verbose=True, **kwargs):
        """
        CommandLine:
            python -m ibeis_cnn --tf  SiameseL2.initialize_architecture --archtag siaml2_partmatch --datashape=128,256,1 --verbose  --show
        """
        _P = functools.partial

        leaky_kw = dict(nonlinearity=nonlinearities.LeakyRectify(leakiness=(1. / 10.)))
        #orthog_kw = dict(W=init.Orthogonal())
        #hidden_initkw = ut.merge_dicts(orthog_kw, leaky_kw)
        hidden_initkw = leaky_kw

        #ReshapeLayer = layers.ReshapeLayer

        _tmp = [1]

        def CDP_layer(num_filters=32,
                              conv_size=(5, 5), conv_stride=(3, 3),
                              pool_size=(2, 2), pool_stride=(2, 2),
                              drop_p=0.3):
            num = _tmp[0]
            _tmp[0] += 1
            return [
                _P(Conv2DLayer, num_filters=num_filters, filter_size=conv_size,
                   stride=conv_stride, name='C' + str(num), **hidden_initkw),
                _P(layers.DropoutLayer, p=drop_p, name='D' + str(num)),
                _P(MaxPool2DLayer, pool_size=pool_size, stride=pool_stride, name='P' + str(num)),
            ]

        def CD_layer(num_filters=32,
                     conv_size=(5, 5), conv_stride=(3, 3),
                     drop_p=0.3):
            num = _tmp[0]
            _tmp[0] += 1
            return [
                _P(Conv2DLayer, num_filters=num_filters, filter_size=conv_size,
                   stride=conv_stride, name='C' + str(num), **hidden_initkw),
                _P(layers.DropoutLayer, p=drop_p, name='D' + str(num)),
            ]

        network_layers_def = (
            [
                _P(layers.InputLayer, shape=model.input_shape)
            ]  +
            CDP_layer( 96, (3, 3), (2, 4), (2, 2), (2, 2), .1) +
            CDP_layer(192, (3, 3), (2, 2), (2, 2), (1, 1), .1) +
            CD_layer(128, (3, 3), (2, 2)) +
            CD_layer(96, (2, 2), (1, 1)) +
            CD_layer(64, (2, 2), (1, 1)) +
            CD_layer(64, (1, 1), (1, 1)) +
            #CD_layer(64, (2, 1), (2, 2)) +
            [
                #_P(Conv2DLayer, num_filters=128, filter_size=(3, 3), name='C3_128', **hidden_initkw),
                #_P(Conv2DLayer, num_filters=64, filter_size=(3, 3), name='C4_128', **hidden_initkw),
                #_P(Conv2DLayer, num_filters=64, filter_size=(2, 1), stride=(2, 2), name='C4_128', **hidden_initkw),
                _P(layers.FlattenLayer, outdim=2, name='flatten128'),
                _P(custom_layers.L2NormalizeLayer, axis=2),
            ]
        )
        #raise NotImplementedError('The 2-channel part is not yet implemented')
        return network_layers_def

    def get_siam2streaml2_def(model, verbose=True, **kwargs):
        """
        Notes:
            (ix) siam-2stream-l2 consists of one central and one surround
                branch of siam-2stream.

                C0(96, 7, 3) - ReLU - P0(2, 2) - C1(192, 5, 1) - ReLU - P1(2, 2) - C2(256, 3, 1)

        CommandLine:
            python -m ibeis_cnn --tf  SiameseL2.initialize_architecture --archtag siam2streaml2 --datashape=64,64,1 --verbose  --show
        """
        _P = functools.partial

        leaky_kw = dict(nonlinearity=nonlinearities.LeakyRectify(leakiness=(1. / 10.)))
        #orthog_kw = dict(W=init.Orthogonal())
        #hidden_initkw = ut.merge_dicts(orthog_kw, leaky_kw)
        hidden_initkw = leaky_kw

        #ReshapeLayer = layers.ReshapeLayer

        network_layers_def = (
            [
                _P(layers.InputLayer, shape=model.input_shape),
                # TODO: Stack Inputs by making a 2 Channel Layer
                #caffenet.get_conv2d_layer(0, trainable=False, **leaky),
                _P(custom_layers.CenterSurroundLayer, name='CentSuround'),
                _P(Conv2DLayer, num_filters=96, filter_size=(5, 5), stride=(1, 1), name='C0', **hidden_initkw),
                _P(layers.DropoutLayer, p=0.3),
                _P(MaxPool2DLayer, pool_size=(2, 2), stride=(2, 2), name='P0'),
                _P(Conv2DLayer, num_filters=192, filter_size=(3, 3), name='C1', **hidden_initkw),
                _P(layers.DropoutLayer, p=0.3),
                _P(MaxPool2DLayer, pool_size=(2, 2), stride=(2, 2), name='P0'),
                _P(Conv2DLayer, num_filters=256, filter_size=(3, 3), name='C2', **hidden_initkw),
                _P(layers.DropoutLayer, p=0.3),
                _P(MaxPool2DLayer, pool_size=(2, 2), stride=(1, 1), name='P0'),
                _P(Conv2DLayer, num_filters=256, filter_size=(3, 3), name='C3', **hidden_initkw),
                _P(custom_layers.SiameseConcatLayer, axis=1, data_per_label=2, name='concat'),  # 2 when CenterSurroundIsOn but two channel network
                _P(layers.FlattenLayer, outdim=2, name='flatten'),
                #_P(custom_layers.L2NormalizeLayer, axis=2),
                # TODO: L2 distance layer
                #_P(custom_layers.SiameseConcatLayer, data_per_label=2),
            ]
        )
        #raise NotImplementedError('The 2-channel part is not yet implemented')
        return network_layers_def

    def get_mnist_siaml2_def(model, verbose=True, **kwargs):
        """
        python -m ibeis_cnn --tf  SiameseL2.initialize_architecture --archtag mnist_siaml2 --datashape=28,28,1 --verbose  --show

        """
        _P = functools.partial

        leaky_kw = dict(nonlinearity=nonlinearities.LeakyRectify(leakiness=(1. / 10.)))
        #orthog_kw = dict(W=init.Orthogonal())
        #hidden_initkw = ut.merge_dicts(orthog_kw, leaky_kw)
        hidden_initkw = leaky_kw

        network_layers_def = (
            [
                #_P(layers.InputLayer, shape=model.input_shape),
                #_P(Conv2DLayer, num_filters=96, filter_size=(7, 7), stride=(1, 1), name='C0', **hidden_initkw),
                #_P(MaxPool2DLayer, pool_size=(2, 2), stride=(2, 2), name='P0'),
                #_P(Conv2DLayer, num_filters=192, filter_size=(5, 5), name='C1', **hidden_initkw),
                #_P(MaxPool2DLayer, pool_size=(2, 2), stride=(2, 2), name='P1'),
                #_P(Conv2DLayer, num_filters=256, filter_size=(4, 4), name='C2', **hidden_initkw),
                _P(layers.InputLayer, shape=model.input_shape),
                _P(Conv2DLayer, num_filters=96, filter_size=(5, 5), stride=(1, 1), name='C0', **hidden_initkw),
                _P(MaxPool2DLayer, pool_size=(2, 2), stride=(2, 2), name='P0'),
                _P(Conv2DLayer, num_filters=192, filter_size=(3, 3), name='C1', **hidden_initkw),
                _P(MaxPool2DLayer, pool_size=(2, 2), stride=(2, 2), name='P2'),
                _P(Conv2DLayer, num_filters=128, filter_size=(3, 3), name='C2', **hidden_initkw),
                _P(MaxPool2DLayer, pool_size=(2, 2), stride=(1, 1), name='P3'),
                _P(Conv2DLayer, num_filters=128, filter_size=(1, 1), name='C2', **hidden_initkw),
                _P(MaxPool2DLayer, pool_size=(2, 2), stride=(1, 1), name='P3'),
                #_P(layers.ReshapeLayer, shape=(-1, 128))
                _P(layers.FlattenLayer, outdim=2)
                #_P(Conv2DLayer, num_filters=256, filter_size=(2, 2), name='C3', **hidden_initkw),
            ]
        )
        return network_layers_def

    def initialize_architecture(model, verbose=ut.VERBOSE, **kwargs):
        r"""
        Notes:
            http://arxiv.org/pdf/1504.03641.pdf

        CommandLine:
            python -m ibeis_cnn.models.siam --test-SiameseL2.initialize_architecture --verbcnn --show
            python -m ibeis_cnn --tf  SiameseL2.initialize_architecture --verbcnn --show

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis_cnn.models.siam import *  # NOQA
            >>> verbose = True
            >>> arch_tag = ut.get_argval('--archtag', default='siaml2')
            >>> data_shape = tuple(ut.get_argval('--datashape', type_=list, default=(64, 64, 3)))
            >>> model = SiameseL2(batch_size=128, data_shape=data_shape, arch_tag=arch_tag)
            >>> output_layer = model.initialize_architecture()
            >>> model.print_dense_architecture_str()
            >>> ut.quit_if_noshow()
            >>> model.show_architecture_image()
            >>> ut.show_if_requested()
        """
        # TODO: remove output dims
        #_P = functools.partial
        print('[model] initialize_architecture')
        #(_, input_channels, input_width, input_height) = model.input_shape
        (_, input_channels, input_height, input_width) = model.input_shape
        if verbose:
            print('[model] Initialize center siamese l2 model architecture')
            print('[model]   * batch_size     = %r' % (model.batch_size,))
            print('[model]   * input_width    = %r' % (input_width,))
            print('[model]   * input_height   = %r' % (input_height,))
            print('[model]   * input_channels = %r' % (input_channels,))
            print('[model]   * output_dims    = %r' % (model.output_dims,))

        #network_layers_def = model.get_mnist_siaml2_def(verbose=verbose, **kwargs)
        network_layers_def = getattr(model, 'get_' + model.arch_tag + '_def')(verbose=verbose, **kwargs)
        #if model.arch_tag == 'siam2streaml2':
        #    network_layers_def = model.get_siam2streaml2_def(verbose=verbose, **kwargs)
        #elif model.arch_tag == 'siaml2':
        #    network_layers_def = model.get_siaml2_def(verbose=verbose, **kwargs)
        #elif model.arch_tag == 'siaml2_128':
        #    network_layers_def = model.get_siaml2_128_def(verbose=verbose, **kwargs)
        #elif model.arch_tag == 'mnist_siaml2':
        #    network_layers_def = model.get_mnist_siaml2_def(verbose=verbose, **kwargs)
        # connect and record layers
        network_layers = abstract_models.evaluate_layer_list(network_layers_def, verbose=verbose)
        #model.network_layers = network_layers
        output_layer = network_layers[-1]
        model.output_layer = output_layer
        return output_layer

    def loss_function(model, network_output, labels, T=T, verbose=True):
        """
        Implements the contrastive loss term from (Hasdel, Chopra, LeCun 06)

        CommandLine:
            python -m ibeis_cnn.models.siam --test-SiameseL2.loss_function --show

        Example1:
            >>> # ENABLE_DOCTEST
            >>> from ibeis_cnn.models import *  # NOQA
            >>> network_output, labels = testdata_siam_desc()
            >>> verbose = False
            >>> T = np
            >>> func = SiameseL2.loss_function
            >>> loss, dist_l2 = ut.exec_func_src(func, globals(), locals(), ['loss', 'dist_l2'])
            >>> ut.quit_if_noshow()
            >>> dist0_l2 = dist_l2[labels]
            >>> dist1_l2 = dist_l2[~labels]
            >>> loss0 = loss[labels]
            >>> loss1 = loss[~labels]
            >>> import plottool as pt
            >>> pt.plot2(dist0_l2, loss0, 'x', color=pt.TRUE_BLUE, label='imposter_loss', y_label='loss')
            >>> pt.plot2(dist1_l2, loss1, 'x', color=pt.FALSE_RED, label='genuine_loss', y_label='loss')
            >>> pt.legend()
            >>> ut.show_if_requested()
        """
        if verbose:
            print('[model] Build SiameseL2 loss function')
        vecs1 = network_output[0::2]
        vecs2 = network_output[1::2]
        margin = 1.0
        dist_l2 = T.sqrt(((vecs1 - vecs2) ** 2).sum(axis=1))
        loss = constrastive_loss(dist_l2, labels, margin, T=T)
        # Ignore the hardest cases
        #num_ignore = 3
        #loss = ignore_hardest_cases(loss, labels, num_ignore=num_ignore, T=T)
        return loss

    def learn_encoder(model, labels, scores, **kwargs):
        import vtool as vt
        encoder = vt.ScoreNormalizer(**kwargs)
        encoder.fit(scores, labels)
        print('[model] learned encoder accuracy = %r' % (encoder.get_accuracy(scores, labels)))
        model.encoder = encoder
        return encoder


def ignore_hardest_cases(loss, labels, num_ignore=3, T=T):
    r"""
    Args:
        loss (theano.Tensor):
        labels (theano.Tensor):
        num_ignore (int): (default = 3)
        T (module): (default = theano.tensor)

    Returns:
        theano.Tensor: loss

    CommandLine:
        python -m ibeis_cnn.models.siam --test-ignore_hardest_cases:0
        python -m ibeis_cnn.models.siam --test-ignore_hardest_cases:1
        python -m ibeis_cnn.models.siam --test-ignore_hardest_cases:2

    Example0:
        >>> # ENABLE_DOCTEST
        >>> # Test numpy version
        >>> from ibeis_cnn.models.siam import *  # NOQA
        >>> loss_arr   = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
        >>> labels_arr = np.array([1, 0, 0, 1, 1, 1, 1, 1, 0], dtype=np.int32)
        >>> loss   = loss_arr
        >>> labels = labels_arr
        >>> num_ignore = 2
        >>> T = np
        >>> ignored_loss_arr = ignore_hardest_cases(loss, labels, num_ignore, T)
        >>> result = ('ignored_loss_arr = %s' % (ut.numpy_str(ignored_loss_arr),))
        >>> print(result)
        ignored_loss = np.array([0, 1, 0, 3, 4, 5, 0, 0, 0], dtype=np.int32)

     Example1:
        >>> # ENABLE_DOCTEST
        >>> # Test theano version
        >>> from ibeis_cnn.models.siam import *  # NOQA
        >>> import theano.tensor
        >>> loss_arr   = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
        >>> labels_arr = np.array([1, 0, 0, 1, 1, 1, 1, 1, 0], dtype=np.int32)
        >>> T = theano.tensor
        >>> loss = T.ivector(name='loss')
        >>> labels = T.ivector(name='labels')
        >>> num_ignore = 2
        >>> ignored_loss = ignore_hardest_cases(loss, labels, num_ignore, T)
        >>> ignored_loss_arr = ignored_loss.eval({loss: loss_arr, labels: labels_arr})
        >>> result = ('ignored_loss = %s' % (ut.numpy_str(ignored_loss_arr),))
        >>> print(result)
        ignored_loss = np.array([0, 1, 0, 3, 4, 5, 0, 0, 0], dtype=np.int32)

    Example2:
        >>> # ENABLE_DOCTEST
        >>> # Test version compatiblity
        >>> from ibeis_cnn.models.siam import *  # NOQA
        >>> import ibeis_cnn.theano_ext as theano_ext
        >>> import theano.tensor
        >>> loss_arr   = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
        >>> labels_arr = np.array([1, 0, 0, 1, 1, 1, 1, 1, 0], dtype=np.int32)
        >>> loss = T.ivector(name='loss')
        >>> labels = T.ivector(name='labels')
        >>> num_ignore = 2
        >>> # build numpy targets
        >>> numpy_locals = {'np': np, 'T': np, 'loss': loss_arr, 'labels': labels_arr, 'num_ignore': num_ignore}
        >>> func = ignore_hardest_cases
        >>> numpy_vars = ut.exec_func_src(func, {}, numpy_locals, None)
        >>> numpy_targets = ut.delete_dict_keys(numpy_vars, ['__doc__', 'T', 'np', 'num_ignore'])
        >>> # build theano functions
        >>> theano_locals = {'np': np, 'T': theano.tensor, 'loss': loss, 'labels': labels, 'num_ignore': num_ignore}
        >>> func = ignore_hardest_cases
        >>> theano_vars = ut.exec_func_src(func, {}, theano_locals, None)
        >>> theano_symbols = ut.delete_dict_keys(theano_vars, ['__doc__', 'T', 'np', 'num_ignore'])
        >>> inputs_to_value = {loss: loss_arr, labels: labels_arr}
        >>> # Evalute and test consistency
        >>> key_order = sorted(list(theano_symbols.keys()))
        >>> theano_values = {}
        >>> noerror = True
        >>> for key in key_order:
        ...     symbol = theano_symbols[key]
        ...     print('key=%r' % (key,))
        ...     theano_value = theano_ext.eval_symbol(symbol, inputs_to_value)
        ...     theano_values[key] = theano_value
        ...     prefix = '  * '
        ...     if not np.all(theano_values[key] == numpy_targets[key]):
        ...         prefix = ' !!! '
        ...         noerror = False
        ...     # Cast to compatible dtype
        ...     numpy_value = numpy_targets[key]
        ...     result_dtype = np.result_type(numpy_value, theano_value)
        ...     numpy_value = numpy_value.astype(result_dtype)
        ...     theano_value = theano_value.astype(result_dtype)
        ...     numpy_targets[key] = numpy_value
        ...     theano_values[key] = theano_value
        ...     print(prefix + 'numpy_value  = %r' % (numpy_value,))
        ...     print(prefix + 'theano_value = %r' % (theano_value,))
        >>> print('numpy_targets = ' + ut.dict_str(numpy_targets, align=True))
        >>> print('theano_values = ' + ut.dict_str(theano_values, align=True))
        >>> assert noerror, 'There was an error'

    """
    if T is np:
        T.eq = np.equal
        T.le = np.less_equal

    hardest_sortx_ = loss.argsort()
    hardest_sortx  = hardest_sortx_[::-1]

    invert_sortx = hardest_sortx.argsort()

    hardest_labels = labels[hardest_sortx]

    hardest_istrue = T.eq(hardest_labels, 1)
    hardest_isfalse = T.eq(hardest_labels, 0)

    cumsum_istrue = T.cumsum(hardest_istrue)
    cumsum_isfalse = T.cumsum(hardest_isfalse)

    inrange_true  = T.le(cumsum_istrue, num_ignore)
    inrange_false = T.le(cumsum_isfalse, num_ignore)

    hardest_false_mask = inrange_false * hardest_isfalse
    hardest_true_mask = inrange_true * hardest_istrue
    true_mask = hardest_true_mask[invert_sortx]
    false_mask = hardest_false_mask[invert_sortx]

    keep_mask = 1 - (true_mask + false_mask)

    ignored_loss = keep_mask * loss

    #CHECK = False
    #if CHECK:
    #    hardest_trues  = T.nonzero(hardest_labels)[0][0:num_ignore]
    #    hardest_falses = T.nonzero(1 - hardest_labels)[0][0:num_ignore]
    #    hardest_true_sortx  = hardest_sortx[hardest_trues]
    #    hardest_false_sortx = hardest_sortx[hardest_falses]
    #    hardest_true_sortx.sort()
    #    hardest_false_sortx.sort()
    #    assert np.all(np.where(false_mask)[0] == hardest_false_sortx)
    #    assert np.all(np.where(true_mask)[0] == hardest_true_sortx)
    #    ignored_loss_ = loss.copy()
    #    ignored_loss_[hardest_true_sortx] = 0
    #    ignored_loss_[hardest_false_sortx] = 0

    if T is not np:
        ignored_loss.name = 'ignored_' + loss.name
    return ignored_loss


def constrastive_loss(dist_l2, labels, margin, T=T):
    r"""

    LaTeX:
        $(y E)^2 + ((1 - y) max(m - E, 0)^2)$

    Args:
        dist_l2 (ndarray): energy of a training example (l2 distance of descriptor pairs)
        labels (ndarray): 1 if genuine pair, 0 if imposter pair
        margin (float): positive number
        T (module): (default = theano.tensor)


    Returns:
        ndarray: loss

    Notes:
        Carefull, you need to pass the the euclidean distance in here here, NOT
        the squared euclidean distance otherwise you end up with
        T.maximum(0, (m ** 2 - 2 * m * d + d ** 2)),
        which still requires the square root operation

    CommandLine:
        python -m ibeis_cnn.models.siam --test-constrastive_loss --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.models.siam import *  # NOQA
        >>> dist_l2 = np.linspace(0, 2.5, 200)
        >>> labels = np.tile([True, False], 100)
        >>> # margin, T = 1.25, np
        >>> margin, T = 1.25, np
        >>> loss = constrastive_loss(dist_l2, labels, margin, T)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> xdat_genuine, ydat_genuine = dist_l2[labels], loss[labels] * 2.0
        >>> xdat_imposter, ydat_imposter = dist_l2[~labels], loss[~labels] * 2.0
        >>> #pt.presetup_axes(x_label='Energy (D_w)', y_label='Loss (L)', equal_aspect=False)
        >>> pt.presetup_axes(x_label='Energy (E)', y_label='Loss (L)', equal_aspect=False)
        >>> pt.plot(xdat_genuine, ydat_genuine, '--', lw=2, color=pt.TRUE, label='Correct training pairs')
        >>> pt.plot(xdat_imposter, ydat_imposter, '-', lw=2, color=pt.FALSE,  label='Incorrect training pairs')
        >>> pt.pad_axes(.03, ylim=(0, 3.5))
        >>> pt.postsetup_axes()
        >>> ut.show_if_requested()
    """
    #if __debug__:
    #    assert margin > 0
    #    assert set(labels).issubset({0, 1})
    loss_genuine = (labels * dist_l2) ** 2
    loss_imposter = (1 - labels) * T.maximum(margin - dist_l2, 0) ** 2
    loss = (loss_genuine + loss_imposter) / 2.0
    if T is not np:
        loss.name = 'contrastive_loss'
    return loss


@six.add_metaclass(ut.ReloadingMetaclass)
class SiameseCenterSurroundModel(AbstractSiameseModel):
    """
    Model for individual identification
    """
    def __init__(model, autoinit=False, batch_size=128, input_shape=None,
                 data_shape=(64, 64, 3), **kwargs):
        if data_shape is not None:
            input_shape = (batch_size, data_shape[2], data_shape[0], data_shape[1])
        if input_shape is None:
            (batch_size, 3, 64, 64)
        super(SiameseCenterSurroundModel,
              model).__init__(input_shape=input_shape, batch_size=batch_size,
                              **kwargs)
        #model.network_layers = None
        model.input_shape = input_shape
        model.batch_size = batch_size
        model.output_dims = 1
        # bad name, says that this network will take
        # 2*N images in a batch and N labels that map to
        # two images a piece
        model.data_per_label_input  = 2
        model.data_per_label_output = 1
        if autoinit:
            model.initialize_architecture()

    def initialize_architecture(model, verbose=True, **kwargs):
        r"""
        Notes:
            http://arxiv.org/pdf/1504.03641.pdf

        CommandLine:
            python -m ibeis_cnn.models.siam --test-SiameseCenterSurroundModel.initialize_architecture
            python -m ibeis_cnn.models.siam --test-SiameseCenterSurroundModel.initialize_architecture --verbcnn
            python -m ibeis_cnn.models.siam --test-SiameseCenterSurroundModel.initialize_architecture --verbcnn --show
            python -m ibeis_cnn.train --test-pz_patchmatch --vtd --max-examples=5 --batch_size=128 --learning_rate .0000001 --verbcnn
            python -m ibeis_cnn.train --test-pz_patchmatch --vtd

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis_cnn.models import *  # NOQA
            >>> # build test data
            >>> batch_size = 128
            >>> input_shape = (batch_size, 3, 64, 64)
            >>> verbose = True
            >>> model = SiameseCenterSurroundModel(batch_size=batch_size, input_shape=input_shape)
            >>> # execute function
            >>> output_layer = model.initialize_architecture()
            >>> model.print_dense_architecture_str()
            >>> result = str(output_layer)
            >>> print(result)
            >>> ut.quit_if_noshow()
            >>> import plottool as pt
            >>> model.show_architecture_image()
            >>> ut.show_if_requested()
        """
        print('[model] initialize_architecture')
        (_, input_channels, input_width, input_height) = model.input_shape
        if verbose:
            print('[model] Initialize center surround siamese model architecture')
            print('[model]   * batch_size     = %r' % (model.batch_size,))
            print('[model]   * input_width    = %r' % (input_width,))
            print('[model]   * input_height   = %r' % (input_height,))
            print('[model]   * input_channels = %r' % (input_channels,))
            print('[model]   * output_dims    = %r' % (model.output_dims,))
        network_layers_def = model.get_siam2stream_def(verbose=verbose, **kwargs)
        # connect and record layers
        network_layers = abstract_models.evaluate_layer_list(network_layers_def)
        #model.network_layers = network_layers
        output_layer = network_layers[-1]
        model.output_layer = output_layer
        return output_layer

    def loss_function(model, network_output, Y, T=T, verbose=True):
        """

        CommandLine:
            python -m ibeis_cnn.models.siam --test-loss_function
            python -m ibeis_cnn.models.siam --test-loss_function:1 --show

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis_cnn.models import *  # NOQA
            >>> from ibeis_cnn import ingest_data
            >>> from ibeis_cnn import batch_processing as batch
            >>> data, labels = ingest_data.testdata_patchmatch()
            >>> model = SiameseCenterSurroundModel(autoinit=True, input_shape=(128,) + (data.shape[1:]))
            >>> theano_forward = batch.create_unbuffered_network_output_func(model)
            >>> batch_size = model.batch_size
            >>> Xb, yb = data[0:batch_size * model.data_per_label_input], labels[0:batch_size]
            >>> network_output = theano_forward(Xb)[0]
            >>> network_output = network_output
            >>> Y = yb
            >>> T = np
            >>> # execute function
            >>> verbose = True
            >>> avg_loss = model.loss_function(network_output, Y, T=T)
            >>> result = str(avg_loss)
            >>> print(result)

        Example1:
            >>> # ENABLE_DOCTEST
            >>> from ibeis_cnn.models import *  # NOQA
            >>> network_output = np.linspace(-2, 2, 128)
            >>> Y0 = np.zeros(len(network_output), np.float32)
            >>> Y1 = np.ones(len(network_output), np.float32)
            >>> verbose = False
            >>> T = np
            >>> Y = Y0
            >>> func = SiameseCenterSurroundModel.loss_function
            >>> loss0, Y0_ = ut.exec_func_src(func, globals(), locals(), ['loss', 'Y_'])
            >>> Y = Y1
            >>> loss1, Y1_ = ut.exec_func_src(func, globals(), locals(), ['loss', 'Y_'])
            >>> assert np.all(Y1 == 1) and np.all(Y1_ == 1), 'bad label mapping'
            >>> assert np.all(Y0 == 0) and np.all(Y0_ == -1), 'bad label mapping'
            >>> ut.quit_if_noshow()
            >>> import plottool as pt
            >>> pt.plot2(network_output, loss0, '-', color=pt.TRUE_BLUE, label='imposter_loss', y_label='network output')
            >>> pt.plot2(network_output, loss1, '-', color=pt.FALSE_RED, label='genuine_loss', y_label='network output')
            >>> pt.legend()
            >>> ut.show_if_requested()
        """
        if verbose:
            print('[model] Build SiameseCenterSurroundModel loss function')
        # make y_i in {-1, 1} where -1 denotes non-matching and +1 denotes matching
        #Y_ = (1 - (2 * Y))
        Y_ = ((2 * Y) - 1)
        # Hinge-loss objective from Zagoruyko and Komodakis
        loss = T.maximum(0, 1 - (Y_ * network_output.T))
        avg_loss = T.mean(loss)
        if T is not np:
            loss.name = 'loss'
            avg_loss.name = 'avg_loss'
        return avg_loss

    def learn_encoder(model, labels, scores, **kwargs):
        import vtool as vt
        encoder = vt.ScoreNormalizer(**kwargs)
        encoder.fit(scores, labels)
        print('[model] learned encoder accuracy = %r' % (encoder.get_accuracy(scores, labels)))
        model.encoder = encoder
        return encoder

    def get_2ch2stream_def(model, verbose=True, **kwargs):
        """
        Notes:
            (i) 2ch-2stream consists of two branches
                C(95, 5, 1)- ReLU- P(2, 2)- C(96, 3, 1)- ReLU- P(2, 2)- C(192, 3, 1)-
                  ReLU- C(192, 3, 1)- ReLU,
                one for central and one for surround parts, followed by
                F(768)- ReLU- F(1)
        """
        raise NotImplementedError('The 2-channel part is not yet implemented')
        _P = functools.partial
        leaky_kw = dict(nonlinearity=nonlinearities.LeakyRectify(leakiness=(1. / 10.)))
        orthog_kw = dict(W=init.Orthogonal())
        hidden_initkw = ut.merge_dicts(orthog_kw, leaky_kw)

        network_layers_def = (
            [
                _P(layers.InputLayer, shape=model.input_shape),
                # TODO: Stack Inputs by making a 2 Channel Layer
                _P(custom_layers.CenterSurroundLayer, name='CentSur'),

                #layers.GaussianNoiseLayer,
                #caffenet.get_conv2d_layer(0, trainable=False, **leaky),
                #lasange_ext.freeze_params,
                _P(Conv2DLayer, num_filters=96, filter_size=(5, 5), stride=(1, 1), name='C0', **hidden_initkw),
                _P(MaxPool2DLayer, pool_size=(2, 2), stride=(2, 2), name='P0'),
                _P(Conv2DLayer, num_filters=96, filter_size=(3, 3), name='C1', **hidden_initkw),
                _P(MaxPool2DLayer, pool_size=(2, 2), stride=(2, 2), name='P1'),
                _P(Conv2DLayer, num_filters=192, filter_size=(3, 3), name='C2', **hidden_initkw),
                _P(Conv2DLayer, num_filters=192, filter_size=(3, 3), name='C3', **hidden_initkw),
                #_P(custom_layers.L2NormalizeLayer, axis=2),
                _P(custom_layers.SiameseConcatLayer, axis=1, data_per_label=4, name='Concat'),  # 4 when CenterSurroundIsOn
                #_P(custom_layers.SiameseConcatLayer, data_per_label=2),
                _P(layers.DenseLayer, num_units=768, name='F1',  **hidden_initkw),
                _P(layers.DropoutLayer, p=0.5),
                _P(layers.DenseLayer, num_units=1, name='F2', **hidden_initkw),
            ]
        )
        return network_layers_def

    def get_siam2stream_def(model, verbose=True, **kwargs):
        """
        Notes:
            (viii) siam-2stream has 4 branches
                C(96, 4, 2)- ReLU- P(2, 2)- C(192, 3, 1)- ReLU- C(256, 3, 1)- ReLU- C(256, 3, 1)-
                  ReLU
                (coupled in pairs for central and surround streams, and decision layer)
                F(512)-ReLU- F(1)
        """
        _P = functools.partial

        leaky_kw = dict(nonlinearity=nonlinearities.LeakyRectify(leakiness=(1. / 10.)))
        orthog_kw = dict(W=init.Orthogonal())
        hidden_initkw = ut.merge_dicts(orthog_kw, leaky_kw)
        if not kwargs.get('fresh_model', False):
            # FIXME: figure out better way of encoding that Orthogonal
            # Initialization doesnt need to happen. Or do initialization of
            # weights after the network has been built.
            # don't do fancy initializating unless training from scratch
            del hidden_initkw['W']

        network_layers_def = (
            [
                _P(layers.InputLayer, shape=model.input_shape),
                # TODO: Stack Inputs by making a 2 Channel Layer
                _P(custom_layers.CenterSurroundLayer),

                layers.GaussianNoiseLayer,
                #caffenet.get_conv2d_layer(0, trainable=False, **leaky),
                _P(Conv2DLayer, num_filters=96, filter_size=(4, 4), name='C0', **hidden_initkw),
                _P(MaxPool2DLayer, pool_size=(2, 2), stride=(2, 2), name='P0'),
                _P(Conv2DLayer, num_filters=192, filter_size=(3, 3), name='C2', **hidden_initkw),
                _P(Conv2DLayer, num_filters=256, filter_size=(3, 3), name='C3', **hidden_initkw),
                _P(Conv2DLayer, num_filters=256, filter_size=(3, 3), name='C3', **hidden_initkw),
                #_P(custom_layers.L2NormalizeLayer, axis=2),
                _P(custom_layers.SiameseConcatLayer, axis=1, data_per_label=4),  # 4 when CenterSurroundIsOn
                #_P(custom_layers.SiameseConcatLayer, data_per_label=2),
                _P(layers.DenseLayer, num_units=512, name='F1',  **hidden_initkw),
                _P(layers.DropoutLayer, p=0.5),
                _P(layers.DenseLayer, num_units=1, name='F2', **hidden_initkw),
            ]
        )
        #raise NotImplementedError('The 2-channel part is not yet implemented')
        return network_layers_def

    def get_siam2stream_l2_def(model, verbose=True, **kwargs):
        """
        Notes:
            (ix) siam-2stream-l2 consists of one central and one surround
                branch of siam-2stream.
        """
        raise NotImplementedError('Need to implement L2 distance layer')
        _P = functools.partial

        leaky_kw = dict(nonlinearity=nonlinearities.LeakyRectify(leakiness=(1. / 10.)))
        orthog_kw = dict(W=init.Orthogonal())
        hidden_initkw = ut.merge_dicts(orthog_kw, leaky_kw)
        if not kwargs.get('fresh_model', False):
            # FIXME: figure out better way of encoding that Orthogonal
            # Initialization doesnt need to happen. Or do initialization of
            # weights after the network has been built.
            # don't do fancy initializating unless training from scratch
            del hidden_initkw['W']

        network_layers_def = (
            [
                _P(layers.InputLayer, shape=model.input_shape),
                # TODO: Stack Inputs by making a 2 Channel Layer
                _P(custom_layers.CenterSurroundLayer),

                layers.GaussianNoiseLayer,
                #caffenet.get_conv2d_layer(0, trainable=False, **leaky),
                _P(Conv2DLayer, num_filters=96, filter_size=(4, 4), name='C0', **hidden_initkw),
                _P(MaxPool2DLayer, pool_size=(2, 2), stride=(2, 2), name='P0'),
                _P(Conv2DLayer, num_filters=192, filter_size=(3, 3), name='C2', **hidden_initkw),
                _P(Conv2DLayer, num_filters=256, filter_size=(3, 3), name='C3', **hidden_initkw),
                _P(Conv2DLayer, num_filters=256, filter_size=(3, 3), name='C3', **hidden_initkw),
                #_P(custom_layers.L2NormalizeLayer, axis=2),
                _P(custom_layers.SiameseConcatLayer, axis=1, data_per_label=4),  # 4 when CenterSurroundIsOn
                # TODO: L2 distance layer
                #_P(custom_layers.SiameseConcatLayer, data_per_label=2),
            ]
        )
        #raise NotImplementedError('The 2-channel part is not yet implemented')
        return network_layers_def


def predict():
    pass


def testdata_siam_desc(num_data=128, desc_dim=8):
    import vtool as vt
    rng = np.random.RandomState(0)
    network_output = vt.normalize_rows(rng.rand(num_data, desc_dim))
    vecs1 = network_output[0::2]
    vecs2 = network_output[1::2]
    # roll vecs2 so it is essentially translated
    vecs2 = np.roll(vecs1, 1, axis=1)
    network_output[1::2] = vecs2
    # Every other pair is an imposter match
    network_output[::4, :] = vt.normalize_rows(rng.rand(32, desc_dim))
    #data_per_label = 2

    vecs1 = network_output[0::2]
    vecs2 = network_output[1::2]
    def true_dist_metric(vecs1, vecs2):
        g1_ = np.roll(vecs1, 1, axis=1)
        dist = vt.L2(g1_, vecs2)
        return dist
    #l2dist = vt.L2(vecs1, vecs2)
    true_dist = true_dist_metric(vecs1, vecs2)

    labels = true_dist > 0
    return network_output, labels


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis_cnn.models.siam
        python -m ibeis_cnn.models.siam --allexamples
        python -m ibeis_cnn.models.siam --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
