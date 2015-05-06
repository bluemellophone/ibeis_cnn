from __future__ import absolute_import, division, print_function
from lasagne import layers
import warnings  # NOQA
import theano.tensor as T


def l1(layer, include_biases=False):
    """ custom should move to regulariztion.lasange.l1 """
    with warnings.catch_warnings():
        #warnings.simplefilter("ignore")
        warnings.filterwarnings('ignore', '.*topo.*')
        if include_biases:
            all_params = layers.get_all_params(layer)
        else:
            all_params = layers.get_all_non_bias_params(layer)

    return sum(T.sum(T.abs_(p)) for p in all_params)
