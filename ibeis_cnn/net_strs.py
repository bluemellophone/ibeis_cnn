# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import warnings
import six
import functools
import utool as ut
print, rrr, profile = ut.inject2(__name__, '[ibeis_cnn.net_strs]')


def make_layer_str(layer):
    r"""
    Args:
        layer (lasange.Layer): a network layer
    """
    layer_info = get_layer_info(layer)
    layer_type = layer_info['classname']
    # filter_size is a scalar in Conv2DLayers because non-uniform shapes is not supported
    #common_attrs = ['shape', 'num_filters', 'num_units', 'ds', 'filter_shape',
    #                'stride', 'strides', 'p', 'axis']
    #ignore_attrs = ['nonlinearity', 'b', 'W']
    #func_attrs = ['get_output_shape']
    #func_attrs = []
    #layer_attrs_dict = {
    #    'DropoutLayer': ['p'],
    #    'InputLayer': ['shape'],
    #    'Conv2DCCLayer': ['num_filters', 'filter_size', 'stride'],
    #    'MaxPool2DCCLayer': ['stride', 'pool_size'],
    #    'DenseLayer': ['num_units'],
    #    'L2NormalizeLayer': ['axis'],
    #}
    #request_attrs = sorted(list(set(layer_attrs_dict.get(layer_type, []) + common_attrs)))
    #isvalid_list = [hasattr(layer, attr) for attr in request_attrs]
    #attr_key_list = ut.compress(request_attrs, isvalid_list)
    attr_key_list = layer_info['layer_attrs']

    #DEBUG = False
    #if DEBUG:
    #    #if layer_type == 'Conv2DCCLayer':
    #    #    ut.embed()
    #    print('---')
    #    print(' * ' + layer_type)
    #    print(' * does not have keys: %r' % (ut.filterfalse_items(request_attrs, isvalid_list),))
    #    print(' * missing keys: %r' % ((set(layer.__dict__.keys()) - set(ignore_attrs)) - set(attr_key_list),))
    #    print(' * has keys: %r' % (attr_key_list,))

    attr_val_list = [getattr(layer, attr) for attr in attr_key_list]
    attr_str_list = ['%s=%r' % item for item in zip(attr_key_list, attr_val_list)]

    #for func_attr in func_attrs:
    #    if hasattr(layer, func_attr):
    #        attr_str_list.append('%s=%r' % (func_attr, getattr(layer, func_attr).__call__()))

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


def make_layer_json_dict(layer):
    layer_info = get_layer_info(layer)
    #layer_type = layer_info['classname']
    #attr_key_list = layer_info['layer_attrs']
    json_dict = ut.odict([])
    json_dict['type'] = layer_info['classname']
    json_dict.update(**layer_info['layer_attrs'])
    nonlin = layer_info.get('nonlinearity', None)
    if nonlin is not None:
        json_dict['nonlinearity'] = nonlin
    return json_dict


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


def count_bytes(output_layer):
    import lasagne
    layers = lasagne.layers.get_all_layers(output_layer)
    info_list = [get_layer_info(layer) for layer in layers]
    total_bytes = sum([info['total_bytes'] for info in info_list])
    #print('total_bytes = %s' % (ut.byte_str2(total_bytes),))
    #print(ut.repr2(info_list, nl=2, hack_liststr=True))
    return total_bytes


def surround(str_, b='{}'):
    return b[0] + str_ + b[1]


def tagstr(tags):
    #return surround(','.join([t[0] for t in tags]), '{}')
    return ','.join([t[0] for t in tags])


def shapestr(shape):
    return ','.join(map(str, shape))


def paramstr(layer, p, tags):
    pname = p.name
    if layer.name is not None:
        if pname.startswith(layer.name + '.'):
            pname = pname[len(layer.name) + 1:]
    inner_parts = [shapestr(p.get_value().shape)]
    if tags:
        inner_parts.append(tagstr(tags))
    return pname + surround('; '.join(inner_parts), '()')


def get_layer_info(layer):
    r"""
    Args:
        layer (?):

    Returns:
        ?: layer_info

    CommandLine:
        python -m ibeis_cnn.net_strs get_layer_info --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.net_strs import *  # NOQA
        >>> from ibeis_cnn import models
        >>> model = models.mnist.MNISTModel(batch_size=8, data_shape=(24, 24, 1), output_dims=10)
        >>> model.initialize_architecture()
        >>> nn_layers = model.get_all_layers()
        >>> for layer in nn_layers:
        >>>     layer_info = get_layer_info(layer)
        >>>     print(ut.repr3(layer_info, nl=1))
    """
    import operator
    import ibeis_cnn.__LASAGNE__ as lasagne
    # Information that contributes to RAM usage
    import numpy as np
    # Get basic layer infos
    output_shape = lasagne.layers.get_output_shape(layer)
    input_shape = getattr(layer, 'input_shape', [])
    # Get number of outputs ignoring the batch size
    num_outputs = functools.reduce(operator.mul, output_shape[1:])
    if len(input_shape):
        num_inputs = functools.reduce(operator.mul, input_shape[1:])
    else:
        num_inputs = 0
    # TODO: if we can ever support non float32 calculations this must change
    #layer_type = 'float32'
    layer_dtype = np.dtype('float32')

    # Get individual param infos
    param_infos = []
    for param in layer.params.keys():
        value = param.get_value()
        param_info = ut.odict([
            ('name', param.name),
            ('shape', value.shape),
            ('size', value.size),
            ('itemsize', value.dtype.itemsize),
            ('dtype', str(value.dtype)),
            ('bytes', value.size * value.dtype.itemsize),
        ])
        param_infos.append(param_info)
    # Combine param infos
    param_str = surround(', '.join(
        [paramstr(layer, p, tags) for p, tags in layer.params.items()]), '[]')
    param_type_str = surround(', '.join(
        [repr(p.type) for p, tags in layer.params.items()]), '[]')
    num_params = sum([info['size'] for info in param_infos])

    classalias_map = {
        'Conv2DCCLayer'    : 'Conv2D',
        'Conv2DDNNLayer'   : 'Conv2D',
        'Conv2DLayer'   : 'Conv2D',
        'MaxPool2DLayer': 'MaxPool2D',
        'MaxPool2DCCLayer' : 'MaxPool2D',
        'MaxPool2DDNNLayer' : 'MaxPool2D',
        'LeakyRectify'     : 'LRU',
        'InputLayer'       : 'Input',
        'DropoutLayer'     : 'Dropout',
        'DenseLayer'       : 'Dense',
        'FlattenLayer'     : 'Flatten',
        'L2NormalizeLayer' : 'L2Norm',
        'BatchNormLayer'   : 'BatchNorm',
    }
    layer_attrs_ignore_dict = {
        'MaxPool2D'  : ['mode', 'ignore_border'],
        'Dropout'  : ['rescale'],
        'Conv2D'   : ['convolution'],
        'BatchNorm': ['epsilon', 'mean', 'inv_std', 'axes', 'beta', 'gamma']
    }
    layer_attrs_dict = {
        'Input'     : ['shape'],
        'Dropout'   : ['p'],
        'Conv2D'    : ['num_filters', 'filter_size', 'stride'],
        'MaxPool2D' : ['stride', 'pool_size'],  # 'mode'],
        'Dense'     : ['num_units'],
        'SoftMax'   : ['num_units'],
        'L2Norm'    : ['axis'],
        'BatchNorm' : ['alpha']
    }
    classname = layer.__class__.__name__
    classalias = classalias_map.get(classname, classname)
    if classalias == 'FeaturePoolLayer' and ut.get_funcname(layer.pool_function) == 'max':
        classalias = 'MaxOut'
    if classalias == 'Dense' and ut.get_funcname(layer.nonlinearity) == 'softmax':
        classalias = 'SoftMax'

    layer_attrs = ut.odict([(key, getattr(layer, key))
                            for key in layer_attrs_dict.get(classalias, [])])
    attr_key_list = layer_attrs
    ignore_attrs = ['nonlinearity', 'b', 'W', 'get_output_kwargs', 'name',
                    'input_shape', 'input_layer', 'input_var',
                    'untie_biases', 'flip_filters', 'pad', 'params', 'n']
    ignore_attrs += layer_attrs_ignore_dict.get(classalias, [])
    missing_keys = ((set(layer.__dict__.keys()) - set(ignore_attrs)) - set(attr_key_list))
    missing_keys = [k for k in missing_keys if not k.startswith('_')]
    #if layer_type == 'Conv2DCCLayer':
    #    ut.embed()
    DEBUG = True
    if DEBUG and len(missing_keys) > 0:
        print('---')
        print(' * ' + classname)
        print(' * missing keys: %r' % (missing_keys,))
        print(' * has keys: %r' % (attr_key_list,))
        if True:
            raise AssertionError('MISSING KEYS')

    layer_info = ut.odict([
        ('name', layer.name),
        ('classname', classname),
        ('classalias', classalias),
        ('output_shape', output_shape),
        ('input_shape', input_shape),
        ('num_outputs', num_outputs),
        ('num_inputs', num_inputs),
        ('size', np.prod(output_shape)),
        ('itemsize', layer_dtype.itemsize),
        ('dtype', str(layer_dtype)),
        ('num_params', num_params),
        ('param_infos', param_infos),
        ('param_str', param_str),
        ('param_type_str', param_type_str),
        ('layer_attrs', layer_attrs),
        ('nonlinearity', None),
    ])

    if hasattr(layer, 'nonlinearity'):
        try:
            nonlinearity = layer.nonlinearity.__name__
        except AttributeError:
            nonlinearity = layer.nonlinearity.__class__.__name__
        layer_info['nonlinearity'] = ut.odict([])
        layer_info['nonlinearity']['type'] = nonlinearity
        layer_info['nonlinearity'].update(layer.nonlinearity.__dict__)
        #attr_str_list.append('nonlinearity={0}'.format(nonlinearity))

    param_bytes = sum([info['bytes'] for info in param_infos])
    layer_bytes = layer_info['size'] * layer_info['itemsize']
    layer_info['bytes'] = layer_bytes
    layer_info['param_bytes'] = param_bytes
    layer_info['total_bytes'] = layer_bytes + param_bytes
    layer_info['total_memory'] = ut.byte_str2(layer_info['total_bytes'])
    return layer_info


def get_layer_info_str(output_layer):
    r"""
    Args:
        output_layer (lasange.layers.Layer):

    CommandLine:
        python -m ibeis_cnn.net_strs --test-get_layer_info_str:0
        python -m ibeis_cnn.net_strs --test-get_layer_info_str:1

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.net_strs import *  # NOQA
        >>> from ibeis_cnn import models
        >>> model = models.DummyModel(data_shape=(24, 24, 3), autoinit=True)
        >>> output_layer = model.output_layer
        >>> result = get_layer_info_str(output_layer)
        >>> result = '\n'.join([x.rstrip() for x in result.split('\n')])
        >>> print(result)
        Network Structure:
         index  Layer    Outputs    Bytes OutShape         Params
         0      Input      1,728   55,296 (8, 3, 24, 24)   []
         1      Conv2D     7,744  249,600 (8, 16, 22, 22)  [W(16,3,3,3, {t,r}), b(16, {t})]
         2      Conv2D     7,056  229,952 (8, 16, 21, 21)  [W(16,16,2,2, {t,r}), b(16, {t})]
         3      Dense          8  226,080 (8, 8)           [W(7056,8, {t,r}), b(8, {t})]
         4      Dense          5      340 (8, 5)           [W(8,5, {t,r}), b(5, {t})]
        ...this model has 57,989 learnable parameters
        ...this model will use 761,268 bytes = 743.43 KB

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.net_strs import *  # NOQA
        >>> from ibeis_cnn import models
        >>> model = models.mnist.MNISTModel(batch_size=128, output_dims=10,
        >>>                                 data_shape=(24, 24, 3))
        >>> model.initialize_architecture()
        >>> output_layer = model.output_layer
        >>> result = get_layer_info_str(output_layer)
        >>> result = '\n'.join([x.rstrip() for x in result.split('\n')])
        >>> print(result)
    """
    import ibeis_cnn.__LASAGNE__ as lasagne
    info_lines = []
    _print = info_lines.append
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', '.*topo.*')
        nn_layers = lasagne.layers.get_all_layers(output_layer)
        _print('Network Structure:')

        columns_ = ut.ddict(list)

        for index, layer in enumerate(nn_layers):

            layer_info = get_layer_info(layer)

            columns_['index'].append(index)
            columns_['name'].append(layer_info['name'])
            #columns_['type'].append(getattr(layer, 'type', None))
            #columns_['layer'].append(layer_info['classname'])
            columns_['layer'].append(layer_info['classalias'])
            columns_['num_outputs'].append('{:,}'.format(int(layer_info['num_outputs'])))
            columns_['output_shape'].append(str(layer_info['output_shape'] ))
            columns_['params'].append(layer_info['param_str'])
            columns_['param_type'].append(layer_info['param_type_str'])
            columns_['mem'].append(layer_info['total_memory'])
            columns_['bytes'].append('{:,}'.format(int(layer_info['total_bytes'])))
            #ut.embed()

        header_nice = {
            'index'        : 'index',
            'name'         : 'Name',
            'layer'        : 'Layer',
            'type'         : 'Type',
            'num_outputs'  : 'Outputs',
            'output_shape' : 'OutShape',
            'params'       : 'Params',
            'param_type'   : 'ParamType',
            'mem'          : 'Mem',
            'bytes'        : 'Bytes',
        }

        header_align = {
            'index'       : '<',
            'params'      : '<',
            'bytes'       : '>',
            'num_outputs' : '>',
        }

        def get_col_maxval(key):
            header_len = len(header_nice[key])
            val_len = max(list(map(len, map(str, columns_[key]))))
            return max(val_len, header_len)

        header_order = ['index']
        if len(ut.filter_Nones(columns_['name'])) > 0:
            header_order += ['name']
        header_order += ['layer', 'num_outputs']
        #header_order += ['mem']
        header_order += ['bytes']
        header_order += ['output_shape', 'params' ]
        #'param_type']

        max_len = {key: str(get_col_maxval(key) + 1) for key, col in six.iteritems(columns_)}

        fmtstr = ' ' + ' '.join(
            [
                '{:' + align + len_ + '}'
                for align, len_ in zip(ut.dict_take(header_align, header_order, '<'),
                                       ut.dict_take(max_len, header_order))
            ]
        )
        _print(fmtstr.format(*ut.dict_take(header_nice, header_order)))

        row_list = zip(*ut.dict_take(columns_, header_order))
        for row in row_list:
            str_ = fmtstr.format(*row)
            _print(str_)

        total_bytes = count_bytes(output_layer)
        num_params = lasagne.layers.count_params(output_layer)

        _print('...this model has {:,} learnable parameters'.format(num_params))
        _print('...this model will use {:,} bytes = {}'.format(
            total_bytes, ut.byte_str2(total_bytes)))
    info_str = '\n'.join(info_lines)
    return info_str


def print_layer_info(output_layer):
    str_ = get_layer_info_str(output_layer)
    str_ = ut.indent('[info] ' + str_)
    print('\n' + str_)


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
