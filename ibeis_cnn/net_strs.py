# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import warnings
import functools
import utool as ut
print, rrr, profile = ut.inject2(__name__, '[ibeis_cnn.net_strs]')


def make_layer_str(layer):
    r"""
    Args:
        layer (lasange.Layer): a network layer
    """
    # filter_size is a scalar in Conv2DLayers because non-uniform shapes is not supported
    common_attrs = ['shape', 'num_filters', 'num_units', 'ds', 'filter_shape', 'stride', 'strides', 'p', 'axis']
    ignore_attrs = ['nonlinearity', 'b', 'W']
    #func_attrs = ['get_output_shape']
    func_attrs = []
    layer_attrs_dict = {
        'DropoutLayer': ['p'],
        'InputLayer': ['shape'],
        'Conv2DCCLayer': ['num_filters', 'filter_size', 'stride'],
        'MaxPool2DCCLayer': ['stride', 'pool_size'],
        'DenseLayer': ['num_units'],
        'L2NormalizeLayer': ['axis'],
    }
    layer_type = '{0}'.format(layer.__class__.__name__)
    request_attrs = sorted(list(set(layer_attrs_dict.get(layer_type, []) + common_attrs)))
    isvalid_list = [hasattr(layer, attr) for attr in request_attrs]
    attr_key_list = ut.compress(request_attrs, isvalid_list)

    DEBUG = False
    if DEBUG:
        #if layer_type == 'Conv2DCCLayer':
        #    ut.embed()
        print('---')
        print(' * ' + layer_type)
        print(' * does not have keys: %r' % (ut.filterfalse_items(request_attrs, isvalid_list),))
        print(' * missing keys: %r' % ((set(layer.__dict__.keys()) - set(ignore_attrs)) - set(attr_key_list),))
        print(' * has keys: %r' % (attr_key_list,))

    attr_val_list = [getattr(layer, attr) for attr in attr_key_list]
    attr_str_list = ['%s=%r' % item for item in zip(attr_key_list, attr_val_list)]

    for func_attr in func_attrs:
        if hasattr(layer, func_attr):
            attr_str_list.append('%s=%r' % (func_attr, getattr(layer, func_attr).__call__()))

    layer_name = getattr(layer, 'name', None)
    if layer_name is not None:
        attr_str_list = ['name=%s' % (layer_name,)] + attr_str_list

    if hasattr(layer, 'nonlinearity'):
        try:
            nonlinearity = layer.nonlinearity.__name__
        except AttributeError:
            nonlinearity = layer.nonlinearity.__class__.__name__
        attr_str_list.append('nonlinearity={0}'.format(nonlinearity))
    attr_str = ','.join(attr_str_list)

    layer_str = layer_type + '(' + attr_str + ')'
    return layer_str


def print_pretrained_weights(pretrained_weights, lbl=''):
    r"""
    Args:
        pretrained_weights (list of ndarrays): represents layer weights
        lbl (str): label
    """
    print('Initialization network: %r' % (lbl))
    print('Total memory: %s' % (ut.get_object_size_str(pretrained_weights)))
    for index, layer_ in enumerate(pretrained_weights):
        print(' layer {:2}: shape={:<18}, memory={}'.format(index, layer_.shape, ut.get_object_size_str(layer_)))


def print_layer_info(output_layer):
    r"""
    Args:
        output_layer (lasange.layers.Layer):

    CommandLine:
        python -m ibeis_cnn.net_strs --test-print_layer_info

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.net_strs import *  # NOQA
        >>> from ibeis_cnn import models
        >>> model = models.DummyModel(data_shape=(24, 24, 3), autoinit=True)
        >>> output_layer = model.output_layer
        >>> result = print_layer_info(output_layer)
        >>> print(result)
    """
    import operator
    import lasagne
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', '.*topo.*')
        nn_layers = lasagne.layers.get_all_layers(output_layer)
        print('\n[info] Network Structure:')

        columns_ = ut.ddict(list)

        for index, layer in enumerate(nn_layers):
            #print()
            #params = lasagne.layers.get_all_params(layer)
            #layer.get_params()

            def surround(str_, b='{}'):
                return b[0] + str_ + b[1]

            def tagstr(tags):
                return surround(','.join([t[0] for t in tags]), '{}')

            def shapestr(shape):
                return ','.join(map(str, shape))

            def paramstr(p, tags):
                tag_str =  ', ' + tagstr(tags)
                return p.name + surround(shapestr(p.get_value().shape) + tag_str , '()')
                return p.name + surround(shapestr(p.get_value().shape), '()')

            param_str = surround(', '.join([paramstr(p, tags)
                                            for p, tags in layer.params.items()]), '[]')

            param_type_str = surround(', '.join([repr(p.type)
                                                 for p, tags in layer.params.items()]), '[]')

            output_shape = lasagne.layers.get_output_shape(layer)  # .get_output_shape()

            num_outputs = functools.reduce(operator.mul, output_shape[1:])

            columns_['index'].append(index)
            columns_['name'].append(layer.name)
            #columns_['type'].append(getattr(layer, 'type', None))
            columns_['class'].append(layer.__class__.__name__)
            columns_['num_outputs'].append('{:,}'.format(int(num_outputs)))
            columns_['shape'].append(str(output_shape))
            columns_['params'].append(param_str)
            columns_['param_type'].append(param_type_str)
            #ut.embed()

        header_nice = {
            'index'       : 'index',
            'name'        : 'Name',
            'class'       : 'Layer',
            'type'        : 'Type',
            'num_outputs' : 'Outputs',
            'shape'       : 'OutShape',
            'params'      : 'Params',
            'param_type'  : 'ParamType',
        }

        header_align = {
            'index'       : '<',
            'params'      : '<',
        }

        def get_col_maxval(key):
            header_len = len(header_nice[key])
            val_len = max(list(map(len, map(str, columns_[key]))))
            return max(val_len, header_len)

        header_order = ['index', 'name', 'class', 'num_outputs', 'shape', 'params' ]
        #'param_type']

        import six
        max_len = {key: str(get_col_maxval(key) + 1) for key, col in six.iteritems(columns_)}

        fmtstr = '[info]  ' + ' '.join(
            [
                '{:' + align + len_ + '}'
                for align, len_ in zip(ut.dict_take(header_align, header_order, '<'),
                                       ut.dict_take(max_len, header_order))
            ]
        )
        #           )
        #        '{:>' + max_len['index'] + '}',
        #        '{:<' + max_len['class'] + '}'
        #        '{:<' + max_len['num_outputs'] + '}'
        #        '{:<' + max_len['shape'] + '}',
        #        '{:>' + max_len['params'] + '}',
        #    ]
        #)

        print(fmtstr.format(*ut.dict_take(header_nice, header_order)))

        row_list = zip(*ut.dict_take(columns_, header_order))
        for row in row_list:
            str_ = fmtstr.format(*row)
            print(str_)

        print('[info] ...this model has {:,} learnable parameters\n'.format(
            lasagne.layers.count_params(output_layer)
        ))


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis_cnn.net_strs
        python -m ibeis_cnn.net_strs --allexamples
        python -m ibeis_cnn.net_strs --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
