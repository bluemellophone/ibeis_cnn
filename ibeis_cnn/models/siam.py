# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import lasagne  # NOQA
from lasagne import layers
from lasagne import nonlinearities
from lasagne import init
import functools
import six
import theano.tensor as T
import numpy as np
from ibeis_cnn.models import abstract_models
from ibeis_cnn import custom_layers
import utool as ut
print, rrr, profile = ut.inject2(__name__, '[ibeis_cnn.models]')


Conv2DLayer = custom_layers.Conv2DLayer
MaxPool2DLayer = custom_layers.MaxPool2DLayer


@six.add_metaclass(ut.ReloadingMetaclass)
class SiameseCenterSurroundModel(abstract_models.AbstractSiameseModel):
    """
    Model for individual identification
    """
    def __init__(model, autoinit=False, batch_size=128, input_shape=None, data_shape=(64, 64, 3), **kwargs):
        if data_shape is not None:
            input_shape = (batch_size, data_shape[2], data_shape[0], data_shape[1])
        if input_shape is None:
            (batch_size, 3, 64, 64)
        super(SiameseCenterSurroundModel, model).__init__(input_shape=input_shape, batch_size=batch_size, **kwargs)
        #model.network_layers = None
        model.input_shape = input_shape
        model.batch_size = batch_size
        model.output_dims = 1
        # bad name, says that this network will take
        # 2*N images in a batch and N labels that map to
        # two images a piece
        model.data_per_label = 2
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
            python -m ibeis_cnn.train --test-train_patchmatch_pz --vtd --max-examples=5 --batch_size=128 --learning_rate .0000001 --verbcnn
            python -m ibeis_cnn.train --test-train_patchmatch_pz --vtd

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
            >>> img = model.make_architecture_image()
            >>> pt.imshow(img)
            >>> ut.show_if_requested()

        """
        # TODO: remove output dims
        #_P = functools.partial
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
            >>> Xb, yb = data[0:batch_size * model.data_per_label], labels[0:batch_size]
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
                C(95, 5, 1)- ReLU- P(2, 2)- C(96, 3, 1)- ReLU- P(2, 2)- C(192, 3, 1)- ReLU- C(192, 3, 1)- ReLU,
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
                _P(custom_layers.CenterSurroundLayer),

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
                _P(custom_layers.SiameseConcatLayer, axis=1, data_per_label=4),  # 4 when CenterSurroundIsOn
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
                C(96, 4, 2)- ReLU- P(2, 2)- C(192, 3, 1)- ReLU- C(256, 3, 1)- ReLU- C(256, 3, 1)- ReLU
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


@six.add_metaclass(ut.ReloadingMetaclass)
class SiameseL2(abstract_models.AbstractSiameseModel):
    """
    Model for individual identification
    """
    def __init__(model, autoinit=False, batch_size=128, input_shape=None, data_shape=None, **kwargs):
        if data_shape is not None:
            input_shape = (batch_size, data_shape[2], data_shape[0], data_shape[1])
        if input_shape is None:
            (batch_size, 3, 64, 64)
        super(SiameseL2, model).__init__(input_shape=input_shape, batch_size=batch_size, **kwargs)
        #model.network_layers = None
        model.input_shape = input_shape
        model.batch_size = batch_size
        model.output_dims = 1
        model.name = 'siaml2'
        # bad name, says that this network will take
        # 2*N images in a batch and N labels that map to
        # two images a piece
        model.data_per_label = 2
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

        network_layers_def = (
            [
                _P(layers.InputLayer, shape=model.input_shape),
                # TODO: Stack Inputs by making a 2 Channel Layer
                #caffenet.get_conv2d_layer(0, trainable=False, **leaky),
                _P(Conv2DLayer, num_filters=96, filter_size=(7, 7), stride=(3, 3), name='C0', **hidden_initkw),
                _P(MaxPool2DLayer, pool_size=(2, 2), stride=(2, 2), name='P0'),
                _P(Conv2DLayer, num_filters=192, filter_size=(5, 5), name='C1', **hidden_initkw),
                _P(MaxPool2DLayer, pool_size=(2, 2), stride=(2, 2), name='P1'),
                _P(Conv2DLayer, num_filters=256, filter_size=(3, 3), name='C2', **hidden_initkw),
                #_P(custom_layers.L2NormalizeLayer, axis=2),
                # TODO: L2 distance layer
                #_P(custom_layers.SiameseConcatLayer, data_per_label=2),
            ]
        )
        #raise NotImplementedError('The 2-channel part is not yet implemented')
        return network_layers_def

    def initialize_architecture(model, verbose=True, **kwargs):
        r"""
        Notes:
            http://arxiv.org/pdf/1504.03641.pdf

        CommandLine:
            python -m ibeis_cnn.models.siam --test-SiameseL2.initialize_architecture --verbcnn --show

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis_cnn.models.siam import *  # NOQA
            >>> # build test data
            >>> batch_size = 128
            >>> input_shape = (batch_size, 3, 64, 64)
            >>> verbose = True
            >>> model = SiameseL2(batch_size=batch_size, input_shape=input_shape)
            >>> # execute function
            >>> output_layer = model.initialize_architecture()
            >>> model.print_dense_architecture_str()
            >>> # verify results
            >>> result = str(output_layer)
            >>> print(result)
            >>> ut.quit_if_noshow()
            >>> import plottool as pt
            >>> img = model.make_architecture_image()
            >>> pt.imshow(img)
            >>> ut.show_if_requested()

        """
        # TODO: remove output dims
        #_P = functools.partial
        (_, input_channels, input_width, input_height) = model.input_shape
        if verbose:
            print('[model] Initialize center surround siamese model architecture')
            print('[model]   * batch_size     = %r' % (model.batch_size,))
            print('[model]   * input_width    = %r' % (input_width,))
            print('[model]   * input_height   = %r' % (input_height,))
            print('[model]   * input_channels = %r' % (input_channels,))
            print('[model]   * output_dims    = %r' % (model.output_dims,))

        network_layers_def = model.get_siaml2_def(verbose=verbose, **kwargs)
        # connect and record layers
        network_layers = abstract_models.evaluate_layer_list(network_layers_def)
        #model.network_layers = network_layers
        output_layer = network_layers[-1]
        model.output_layer = output_layer
        return output_layer

    def loss_function(model, network_output, Y, T=T, verbose=True):
        """
        References:
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

        CommandLine:
            python -m ibeis_cnn.models.siam --test-SiameseL2.loss_function
            python -m ibeis_cnn.models.siam --test-SiameseL2.loss_function:1 --show

        Example1:
            >>> # ENABLE_DOCTEST
            >>> from ibeis_cnn.models import *  # NOQA
            >>> network_output = np.random.rand(128, 256)
            >>> network_output /= np.linalg.norm(network_output, axis=-1)[:, None]
            >>> Y = np.zeros(64, np.float32)
            >>> Y[::2] += 1
            >>> verbose = False
            >>> T = np
            >>> Y = Y0
            >>> func = SiameseL2.loss_function
            >>> loss0, Y0_ = ut.exec_func_src(func, globals(), locals(), ['loss', 'Y_'])
            >>> Y = Y1
            >>> loss1, Y1_ = ut.exec_func_src(func, globals(), locals(), ['loss', 'Y_'])
            >>> assert np.all(Y1 == 1), 'bad label mapping'
            >>> assert np.all(Y0 == 0), 'bad label mapping'
            >>> assert np.all(Y0_ == -1), 'bad label mapping'
            >>> assert np.all(Y1_ == 1), 'bad label mapping'
            >>> ut.quit_if_noshow()
            >>> import plottool as pt
            >>> pt.plot2(network_output, loss0, '-', color=pt.TRUE_BLUE, label='imposter_loss', y_label='network output')
            >>> pt.plot2(network_output, loss1, '-', color=pt.FALSE_RED, label='genuine_loss', y_label='network output')
            >>> pt.legend()
            >>> ut.show_if_requested()
        """
        if verbose:
            print('[model] Build SiameseL2 loss function')
        # make y_i in {-1, 1} where -1 denotes non-matching and +1 denotes matching
        #Y_ = (1 - (2 * Y))
        vecs1 = network_output[0::2]
        vecs2 = network_output[1::2]
        l2_dist = ((vecs1 - vecs2) ** 2).sum(axis=-1)
        aug_l2_dist = (l2_dist - .5) * 2
        Y_ = ((2 * Y) - 1)
        # Hinge-loss objective from Zagoruyko and Komodakis
        loss = T.maximum(0, 1 - (Y_ * aug_l2_dist.T))
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
