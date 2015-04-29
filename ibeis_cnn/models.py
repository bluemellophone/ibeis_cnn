"""
file model.py
allows the definition of different models to be trained
for initialization: Lasagne/lasagne/init.py
for nonlinearities: Lasagne/lasagne/nonlinearities.py
for layers: Lasagne/lasagne/layers/
"""
from __future__ import absolute_import, division, print_function
from lasagne import layers
from lasagne import nonlinearities
from lasagne import init
import random
import utool as ut
import six
import theano.tensor as T

FORCE_CPU = False  # ut.get_argflag('--force-cpu')
try:
    if FORCE_CPU:
        raise ImportError('GPU is forced off')
    # use cuda_convnet for a speed improvement
    # will not be available without a GPU
    import lasagne.layers.cuda_convnet as convnet
    Conv2DLayer = convnet.Conv2DCCLayer
    MaxPool2DLayer = convnet.MaxPool2DCCLayer
    USING_GPU = True
except ImportError as ex:
    ut.printex(ex, 'WARNING: GPU seems unavailable')
    Conv2DLayer = layers.Conv2DLayer
    MaxPool2DLayer = layers.MaxPool2DLayer
    USING_GPU = False


def MaxPool2DLayer_(*args, **kwargs):
    """ wrapper for gpu / cpu compatibility """
    # if not USING_GPU and 'strides' in kwargs:
    #     # cpu does not have stride kwarg. :(
    #     del kwargs['strides']
    return MaxPool2DLayer(*args, **kwargs)


class SiameseModel(object):
    """
    Model for individual identification
    """
    def __init__(self):
        self.network_layers = None
        # bad name, says that this network will take
        # 2*N images in a batch and N labels that map to
        # two images a piece
        self.data_per_label = 2

    def learning_rate_update(self, x):
        return x / 10.0

    def learning_rate_shock(self, x):
        return x * 10.0

    def draw_architecture(self):
        from ibeis_cnn import draw_net
        filename = 'tmp.png'
        draw_net.draw_to_file(self.network_layers, filename)
        ut.start_file(filename)

    def loss_function(self, x, t):
        """
        Args:
            x : network output
            t : target groundtruth labels

        References:
            https://www.cs.nyu.edu/~sumit/research/assets/cvpr05.pdf
            https://github.com/Lasagne/Lasagne/issues/168
        """
        if True:
            print('[model] Build siamese loss function')

        # Mark same genuine pairs as 0 and imposter pairs as 1
        y = (1 - t[::2])
        # x is the output we get from all images in the batch
        gw = x
        gw1, gw2 = gw[0::2], gw[1::2]
        # Energy of training pairs
        ew = T.sum(((gw1 - gw2) ** 2), axis=1)

        margin = 10
        loss = T.mean(y * ew + (1 - y) * T.maximum(margin - ew, 0))
        return loss

    def build_model(self, batch_size, input_width, input_height,
                    input_channels, output_dims, verbose=True):
        from functools import partial as _P  # NOQA
        if verbose:
            print('[model] Build siamese model')
            print('[model]   * batch_size     = %r' % (batch_size,))
            print('[model]   * input_width    = %r' % (input_width,))
            print('[model]   * input_height   = %r' % (input_height,))
            print('[model]   * input_channels = %r' % (input_channels,))
            print('[model]   * output_dims    = %r' % (output_dims,))

        # JON, ADD THIS INSTEAD W=init.Orthogonal  -- Jason

        #rlu_glorot = dict(nonlinearity=nonlinearities.rectify, W=init.GlorotUniform())
        rlu_orthog = dict(nonlinearity=nonlinearities.rectify, W=init.Orthogonal())
        # variable batch size (None), channel, width, height
        #input_shape = (batch_size * self.data_per_label, input_channels, input_width, input_height)
        input_shape = (None, input_channels, input_width, input_height)

        network_layers_def = [
            _P(layers.InputLayer, shape=input_shape),

            #layers.GaussianNoiseLayer,

            # Convolve + Max Pool + Dropout 1
            _P(Conv2DLayer, num_filters=32, filter_size=(3, 3), **rlu_orthog),
            _P(MaxPool2DLayer_, pool_size=(2, 2), stride=(2, 2)),
            #_P(layers.DropoutLayer,  p=0.10),

            # Convolve + Max Pool + Dropout 2
            _P(Conv2DLayer, num_filters=128, filter_size=(3, 3), **rlu_orthog),
            _P(MaxPool2DLayer_, pool_size=(2, 2), stride=(2, 2),),
            #_P(Conv2DLayer, num_filters=64, filter_size=(4, 4), **rlu_orthog),
            #_P(MaxPool2DLayer_, pool_size=(2, 2), stride=(2, 2),),
            #_P(layers.DropoutLayer,  p=0.30),

            # Convolve + Max Pool + Dropout 3
            _P(Conv2DLayer, num_filters=128, filter_size=(3, 3), **rlu_orthog),
            _P(MaxPool2DLayer_, pool_size=(2, 2), stride=(2, 2),),
            #_P(layers.DropoutLayer,  p=0.30),

            # Convolve + Max Pool + Dropout 4
            _P(Conv2DLayer, num_filters=64, filter_size=(3, 3), **rlu_orthog),
            _P(MaxPool2DLayer_, pool_size=(2, 2), stride=(2, 2),),
            _P(layers.DropoutLayer,  p=0.30),

            # Dense Layer + Feature Pool + Dropout 1
            _P(layers.DenseLayer, num_units=1024, **rlu_orthog),
            #_P(layers.FeaturePoolLayer, pool_size=2),
            #_P(layers.DropoutLayer, p=0.5),

            ## Dense Layer + Feature Pool + Dropout 2
            #_P(layers.DenseLayer, num_units=1024, **rlu_orthog),
            #_P(layers.FeaturePoolLayer, pool_size=2,),
            #_P(layers.DropoutLayer, p=0.5),

            # Softmax output
            #_P(layers.DenseLayer, num_units=output_dims,
            #   nonlinearity=nonlinearities.softmax,
            #   W=init.Orthogonal(),),
        ]

        # connect and record layers
        network_layers = []
        layer_fn_iter = iter(network_layers_def)
        prev_layer = six.next(layer_fn_iter)()
        network_layers.append(prev_layer)
        for layer_fn in layer_fn_iter:
            prev_layer = layer_fn(prev_layer)
            network_layers.append(prev_layer)

        self.network_layers = network_layers
        output_layer = prev_layer
        return output_layer


class PZ_GIRM_LARGE_Model(object):
    def __init__(self):
        pass

    def augment(self, Xb, yb=None):
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

    def label_order_mapping(self, category_list):
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
        viewpoints = len(viewpoint_mapping.keys())
        category_mapping = {}
        for index, species in enumerate(species_list):
            for viewpoint, value in viewpoint_mapping.iteritems():
                key = '%s:%s' % (species, viewpoint, )
                base = viewpoints * index
                category_mapping[key] = base + value
        return category_mapping

    def learning_rate_update(self, x):
        return x / 2.0

    def learning_rate_shock(self, x):
        return x * 2.0

    def build_model(self, batch_size, input_width, input_height, input_channels, output_dims):
        l_in = layers.InputLayer(
            # variable batch size (None), channel, width, height
            shape=(None, input_channels, input_width, input_height)
        )

        l_noise = layers.GaussianNoiseLayer(
            l_in,
        )

        l_conv1 = Conv2DLayer(
            l_noise,
            num_filters=32,
            filter_size=(5, 5),
            # nonlinearity=nonlinearities.rectify,
            nonlinearity=nonlinearities.LeakyRectify(leakiness=(1. / 10.)),
            W=init.Orthogonal(),
        )

        l_pool1 = MaxPool2DLayer_(
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
            W=init.Orthogonal(),
        )

        l_pool2 = MaxPool2DLayer_(
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
            W=init.Orthogonal(),
        )

        l_pool3 = MaxPool2DLayer_(
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
            W=init.Orthogonal(),
        )

        l_pool4 = MaxPool2DLayer_(
            l_conv4,
            pool_size=(2, 2),
            stride=(2, 2),
        )

        l_hidden1 = layers.DenseLayer(
            l_pool4,
            num_units=512,
            # nonlinearity=nonlinearities.rectify,
            nonlinearity=nonlinearities.LeakyRectify(leakiness=(1. / 10.)),
            W=init.Orthogonal(),
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
            W=init.Orthogonal(),
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
            W=init.Orthogonal(),
        )

        return l_out


class PZ_GIRM_LARGE_DEEP_Model(object):
    def __init__(self):
        pass

    def augment(self, Xb, yb=None):
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

    def label_order_mapping(self, category_list):
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
        viewpoints = len(viewpoint_mapping.keys())
        category_mapping = {}
        for index, species in enumerate(species_list):
            for viewpoint, value in viewpoint_mapping.iteritems():
                key = '%s:%s' % (species, viewpoint, )
                base = viewpoints * index
                category_mapping[key] = base + value
        return category_mapping

    def learning_rate_update(self, x):
        return x / 2.0

    def learning_rate_shock(self, x):
        return x * 2.0

    def build_model(self, batch_size, input_width, input_height, input_channels, output_dims):
        l_in = layers.InputLayer(
            # variable batch size (None), channel, width, height
            shape=(None, input_channels, input_width, input_height)
        )

        l_conv1 = Conv2DLayer(
            l_in,
            num_filters=32,
            filter_size=(3, 3),
            # nonlinearity=nonlinearities.rectify,
            nonlinearity=nonlinearities.LeakyRectify(leakiness=(1. / 10.)),
            W=init.Orthogonal(),
        )

        l_conv1_dropout = layers.DropoutLayer(l_conv1, p=0.10)

        l_conv2 = Conv2DLayer(
            l_conv1_dropout,
            num_filters=32,
            filter_size=(3, 3),
            # nonlinearity=nonlinearities.rectify,
            nonlinearity=nonlinearities.LeakyRectify(leakiness=(1. / 10.)),
            W=init.Orthogonal(),
        )

        l_conv2_dropout = layers.DropoutLayer(l_conv2, p=0.10)

        l_pool1 = MaxPool2DLayer_(
            l_conv2_dropout,
            pool_size=(2, 2),
            stride=(2, 2),
        )

        l_conv3 = Conv2DLayer(
            l_pool1,
            num_filters=64,
            filter_size=(3, 3),
            # nonlinearity=nonlinearities.rectify,
            nonlinearity=nonlinearities.LeakyRectify(leakiness=(1. / 10.)),
            W=init.Orthogonal(),
        )

        l_conv3_dropout = layers.DropoutLayer(l_conv3, p=0.30)

        l_conv4 = Conv2DLayer(
            l_conv3_dropout,
            num_filters=64,
            filter_size=(3, 3),
            # nonlinearity=nonlinearities.rectify,
            nonlinearity=nonlinearities.LeakyRectify(leakiness=(1. / 10.)),
            W=init.Orthogonal(),
        )

        l_conv4_dropout = layers.DropoutLayer(l_conv4, p=0.30)

        l_conv5 = Conv2DLayer(
            l_conv4_dropout,
            num_filters=64,
            filter_size=(3, 3),
            # nonlinearity=nonlinearities.rectify,
            nonlinearity=nonlinearities.LeakyRectify(leakiness=(1. / 10.)),
            W=init.Orthogonal(),
        )

        l_conv5_dropout = layers.DropoutLayer(l_conv5, p=0.30)

        l_pool2 = MaxPool2DLayer_(
            l_conv5_dropout,
            pool_size=(2, 2),
            stride=(2, 2),
        )

        l_conv6 = Conv2DLayer(
            l_pool2,
            num_filters=128,
            filter_size=(3, 3),
            # nonlinearity=nonlinearities.rectify,
            nonlinearity=nonlinearities.LeakyRectify(leakiness=(1. / 10.)),
            W=init.Orthogonal(),
        )

        l_conv6_dropout = layers.DropoutLayer(l_conv6, p=0.50)

        l_conv7 = Conv2DLayer(
            l_conv6_dropout,
            num_filters=128,
            filter_size=(3, 3),
            # nonlinearity=nonlinearities.rectify,
            nonlinearity=nonlinearities.LeakyRectify(leakiness=(1. / 10.)),
            W=init.Orthogonal(),
        )

        l_conv7_dropout = layers.DropoutLayer(l_conv7, p=0.50)

        l_conv8 = Conv2DLayer(
            l_conv7_dropout,
            num_filters=128,
            filter_size=(3, 3),
            # nonlinearity=nonlinearities.rectify,
            nonlinearity=nonlinearities.LeakyRectify(leakiness=(1. / 10.)),
            W=init.Orthogonal(),
        )

        l_conv8_dropout = layers.DropoutLayer(l_conv8, p=0.50)

        l_conv9 = Conv2DLayer(
            l_conv8_dropout,
            num_filters=128,
            filter_size=(3, 3),
            # nonlinearity=nonlinearities.rectify,
            nonlinearity=nonlinearities.LeakyRectify(leakiness=(1. / 10.)),
            W=init.Orthogonal(),
        )

        l_conv9_dropout = layers.DropoutLayer(l_conv9, p=0.50)

        l_pool3 = MaxPool2DLayer_(
            l_conv9_dropout,
            pool_size=(2, 2),
            stride=(2, 2),
        )

        l_hidden1 = layers.DenseLayer(
            l_pool3,
            num_units=512,
            # nonlinearity=nonlinearities.rectify,
            nonlinearity=nonlinearities.LeakyRectify(leakiness=(1. / 10.)),
            W=init.Orthogonal(),
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
            W=init.Orthogonal(),
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
            W=init.Orthogonal(),
        )

        return l_out


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis_cnn.models
        python -m ibeis_cnn.models --allexamples
        python -m ibeis_cnn.models --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
