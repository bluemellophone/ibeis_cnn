"""
Functions to create network diagrams from a list of Layers.

References:
    https://github.com/ebenolson/Lasagne/blob/master/examples/draw_net.py

Examples:
    Draw a minimal diagram to a pdf file:
        layers = lasagne.layers.get_all_layers(output_layer)
        draw_to_file(layers, 'network.pdf', output_shape=False)
    Draw a verbose diagram in an IPython notebook:
        from IPython.display import Image #needed to render in notebook
        layers = lasagne.layers.get_all_layers(output_layer)
        dot = get_pydot_graph(layers, verbose=True)
        return Image(dot.create_png())
"""
from __future__ import absolute_import, division, print_function
import warnings
from operator import itemgetter
import numpy as np
import cv2
from os.path import join, exists
import utool as ut
from lasagne import layers
from ibeis_cnn import utils
from ibeis_cnn import net_strs
print, rrr, profile = ut.inject2(__name__, '[ibeis_cnn.draw_net]')


def draw_theano_symbolic_expression(thean_expr):
    import theano
    graph_dpath = '.'
    graph_fname = 'symbolic_graph.png'
    graph_fpath = ut.unixjoin(graph_dpath, graph_fname)
    ut.ensuredir(graph_dpath)
    theano.printing.pydotprint(thean_expr, outfile=graph_fpath, var_with_name_simple=True)
    ut.startfile(graph_fpath)
    return graph_fpath


def get_pydot_graph(layers, output_shape=True, fullinfo=True):
    """
    Creates a PyDot graph of the network defined by the given layers.

    Args:
        layers (list): List of the layers, as obtained from lasange.layers.get_all_layers
        output_shape (bool): If `True`, the output shape of each layer will be displayed.
            (default `True`)
        fullinfo (bool): If `True`, layer attributes like filter shape, stride, etc.  will
            be displayed.  (default `True`)

    Returns:
        PyDot : pydot_graph  object containing the graph

    CommandLine:
        python -m ibeis_cnn.draw_net --test-get_pydot_graph --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_cnn.draw_net import *  # NOQA
        >>> from ibeis_cnn import models
        >>> # build test data
        >>> #model = models.DummyModel(input_shape=(128, 1, 4, 4), autoinit=True)
        >>> model = models.SiameseL2(input_shape=(128, 1, 4, 4), autoinit=True)
        >>> layers = model.get_all_layers()
        >>> output_shape = True
        >>> fullinfo = True
        >>> # execute function
        >>> pydot_graph = get_pydot_graph(layers, output_shape, fullinfo)
        >>> # verify results
        >>> result = str(pydot_graph)
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> img = pydot_to_image(pydot_graph)
        >>> pt.imshow(img)
        >>> ut.show_if_requested()
    """
    import pydot
    pydot_graph = pydot.Dot('Network', graph_type='digraph')
    node_dict = {}
    edge_list = []

    def get_hex_color(layer_type):
        """
        Determines the hex color for a layer. Some classes are given
        default values, all others are calculated pseudorandomly
        from their name.
        :parameters:
            - layer_type : string
                Class name of the layer
        :returns:
            - color : string containing a hex color.
        :usage:
            >>> color = get_hex_color('MaxPool2DDNN')
            '#9D9DD2'
        """
        if 'Input' in layer_type:
            return '#A2CECE'
        if 'Conv' in layer_type:
            return '#7C9ABB'
        if 'Dense' in layer_type:
            return '#6CCF8D'
        if 'Pool' in layer_type:
            return '#9D9DD2'
        else:
            return '#{0:x}'.format(hash(layer_type) % 2 ** 24)

    for i, layer in enumerate(layers):
        layer_type = '{0}'.format(layer.__class__.__name__)
        key = repr(layer)
        color = get_hex_color(layer_type)
        # Make label
        label = layer_type
        if fullinfo:
            for attr in ['num_filters', 'num_units', 'ds', 'axis'
                         'filter_shape', 'stride', 'strides', 'p', 'shape', 'name']:
                if hasattr(layer, attr):
                    label += '\n' + \
                        '{0}: {1}'.format(attr, getattr(layer, attr))
            if hasattr(layer, 'nonlinearity'):
                try:
                    nonlinearity = layer.nonlinearity.__name__
                except AttributeError:
                    nonlinearity = layer.nonlinearity.__class__.__name__
                label += '\n' + 'nonlinearity: {0}'.format(nonlinearity)

        if output_shape:
            label += '\n' + \
                'Output shape: {0}'.format(layer.output_shape)
        # append node

        node_dict[key] = pydot.Node(
            key, label=label, shape='record',
            fillcolor=color, style='filled',)

        if hasattr(layer, 'input_layers'):
            for input_layer in layer.input_layers:
                edge_list.append([repr(input_layer), key])

        if hasattr(layer, 'input_layer'):
            edge_list.append([repr(layer.input_layer), key])

    for node in node_dict.values():
        pydot_graph.add_node(node)

    for edge in edge_list:
        pydot_graph.add_edge(
            pydot.Edge(node_dict[edge[0]], node_dict[edge[1]]))
    return pydot_graph


def pydot_to_image(dot):
    from PIL import Image
    from cStringIO import StringIO
    png_str = dot.create_png(prog='dot')
    sio = StringIO()
    sio.write(png_str)
    sio.seek(0)
    pil_img = Image.open(sio)
    img = np.asarray(pil_img.convert('RGB'))
    img = img[..., ::-1]  # to bgr
    pil_img.close()
    sio.close()
    return img


def make_architecture_image(layers, **kwargs):
    """
    Args:
        layers (list): List of the layers, as obtained from lasange.layers.get_all_layers

    Kwargs:
        see docstring of get_pydot_graph for other options

    References:
        http://stackoverflow.com/questions/4596962/display-graph-without-saving-using-pydot

    CommandLine:
        python -m ibeis_cnn.draw_net --test-make_architecture_image --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_cnn.draw_net import *  # NOQA
        >>> from ibeis_cnn import models
        >>> model = models.SiameseCenterSurroundModel(autoinit=True)
        >>> #model = models.DummyModel(autoinit=True)
        >>> layers = model.get_all_layers()
        >>> # execute function
        >>> kwargs = {}
        >>> img = make_architecture_image(layers, **kwargs)
        >>> print(img.shape)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> pt.imshow(img)
        >>> ut.show_if_requested()
    """
    #from IPython.display import Image  # needed to render in notebook
    dot = get_pydot_graph(layers, **kwargs)
    img = pydot_to_image(dot)
    return img


def draw_to_file(layers, filename, **kwargs):
    """
    Draws a network diagram to a file

    Args:
        - layers : list
            List of the layers, as obtained from lasange.layers.get_all_layers
        - filename: string
            The filename to save output to.
        - **kwargs: see docstring of get_pydot_graph for other options

    CommandLine:
        python -m ibeis_cnn.draw_net --test-draw_to_file --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_cnn.draw_net import *  # NOQA
        >>> from ibeis_cnn import models
        >>> #model = models.DummyModel(autoinit=True)
        >>> model = models.SiameseCenterSurroundModel(autoinit=True)
        >>> layers = model.get_all_layers()
        >>> filename = ut.unixjoin(ut.ensure_app_resource_dir('ibeis_cnn'), 'tmp.png')
        >>> # execute function
        >>> draw_to_file(layers, filename)
        >>> ut.quit_if_noshow()
        >>> ut.startfile(filename)
    """
    dot = get_pydot_graph(layers, **kwargs)

    ext = filename[filename.rfind('.') + 1:]
    with open(filename, 'w') as fid:
        fid.write(dot.create(format=ext))


def show_confusion_matrix(correct_y, predict_y, category_list, results_path,
                          mapping_fn=None, data_x=None):
    """
    Given the correct and predict labels, show the confusion matrix

    Args:
        correct_y (list of int): the list of correct labels
        predict_y (list of int): the list of predict assigned labels
        category_list (list of str): the category list of all categories

    Displays:
        matplotlib: graph of the confusion matrix

    Returns:
        None
    """
    import matplotlib.pyplot as plt
    confused_examples = join(results_path, 'confused')
    if data_x is not None:
        if exists(confused_examples):
            ut.remove_dirs(confused_examples, quiet=True)
        ut.ensuredir(confused_examples)
    size = len(category_list)

    if mapping_fn is None:
        # Identity
        category_mapping = { key: index for index, key in enumerate(category_list) }
        category_list_ = category_list
    else:
        category_mapping = mapping_fn(category_list)
        assert all([ category in category_mapping.keys() for category in category_list ]), 'Not all categories are mapped'
        values = list(category_mapping.values())
        assert len(list(set(values))) == len(values), 'Mapped categories have a duplicate assignment'
        assert 0 in values, 'Mapped categories must have a 0 index'
        temp = list(category_mapping.iteritems())
        temp = sorted(temp, key=itemgetter(1))
        category_list_ = [ t[0] for t in temp ]

    confidences = np.zeros((size, size))
    counters = {}
    for index, (correct, predict) in enumerate(zip(correct_y, predict_y)):
        # Ensure type
        correct = int(correct)
        predict = int(predict)
        # Get the "text" label
        example_correct_label = category_list[correct]
        example_predict_label = category_list[predict]
        # Perform any mapping that needs to be done
        correct_ = category_mapping[example_correct_label]
        predict_ = category_mapping[example_predict_label]
        # Add to the confidence matrix
        confidences[correct_][predict_] += 1
        if data_x is not None and correct_ != predict_:
            example = data_x[index]
            example_name = '%s^SEEN_INCORRECTLY_AS^%s' % (example_correct_label, example_predict_label, )
            if example_name not in counters.keys():
                counters[example_name] = 0
            counter = counters[example_name]
            counters[example_name] += 1
            example_name = '%s^%d.png' % (example_name, counter)
            example_path = join(confused_examples, example_name)
            cv2.imwrite(example_path, example)

    row_sums = np.sum(confidences, axis=1)
    norm_conf = (confidences.T / row_sums).T

    fig = plt.figure(1)
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet,
                    interpolation='nearest')

    for x in range(size):
        for y in range(size):
            ax.annotate(str(int(confidences[x][y])), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')

    cb = fig.colorbar(res)  # NOQA
    plt.xticks(np.arange(size), category_list_[0:size], rotation=90)
    plt.yticks(np.arange(size), category_list_[0:size])
    margin_small = 0.1
    margin_large = 0.9
    plt.subplots_adjust(left=margin_small, right=margin_large, bottom=margin_small, top=margin_large)
    plt.xlabel('Predicted')
    plt.ylabel('Correct')
    output_fpath = join(results_path, 'confusion.png')
    plt.savefig(output_fpath)
    return output_fpath


def show_convolutional_layers(output_layer, results_path, use_color=None,
                              limit=144, target=None, epoch=None,
                              verbose=ut.VERYVERBOSE):
    r"""
    CommandLine:
        python -m ibeis_cnn.draw_net --test-show_convolutional_layers --show
        python -m ibeis_cnn.draw_net --test-show_convolutional_layers

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_cnn.draw_net import *  # NOQA
        >>> from ibeis_cnn import models
        >>> model = models.DummyModel(autoinit=True, input_shape=(128, 3, 64, 64))
        >>> output_layer = model.get_output_layer()
        >>> results_path = ut.ensure_app_resource_dir('ibeis_cnn', 'testing', 'figs')
        >>> # ut.vd(results_path)
        >>> use_color = True
        >>> limit = 150
        >>> target = 0
        >>> epoch = 0
        >>> result = show_convolutional_layers(output_layer, results_path, use_color, limit, target, epoch)
        >>> print('result = %r' % (result,))
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> pt.imshow(result[0])
        >>> ut.show_if_requested()
    """
    import matplotlib.pyplot as plt
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', '.*topo.*')
        nn_layers = layers.get_all_layers(output_layer)
    # TODO: be able to visualize all layers
    #cnn_layers = [layer for layer in nn_layers if 'Conv2DCCLayer' in str(type(layer))]
    cnn_layers = [layer for layer in nn_layers if hasattr(layer, 'W')]
    #weights_list = [layer.W.get_value() for layer in cnn_layers]

    # hacky to maintain functionality
    #print('target = %r' % (target,))
    #print('len(cnn_layers) = %r' % (len(cnn_layers),))
    #print('len(nn_layers) = %r' % (len(nn_layers),))
    if isinstance(target, list):
        cnn_layers = ut.list_take(cnn_layers, target)
    elif target is not None:
        cnn_layers = [cnn_layers[target]]
    #print('len(nn_layers) = %r' % (len(nn_layers),))

    output_fpath_list = []
    for index, layer in enumerate(cnn_layers):
        #print('index = %r' % (index,))
        all_weights = layer.W.get_value()
        #ut.embed()
        layername = net_strs.make_layer_str(layer) + '_%d' % (index,)
        #layername = 'conv%d' % (index,)
        #print(layername)
        #print('layername = %r' % (layername,))
        fig = show_convolutional_weights(all_weights, use_color=use_color, limit=limit)
        fig
        #print('...NEXT\n')

        # Save the figure
        color_str = 'color' if use_color else 'gray'
        if epoch is None:
            epoch = 'X'
        output_fname = 'features_%s_epoch_%s_%s.png' % (layername, epoch, color_str)
        fig_dpath = join(results_path, 'layer_features_%s' % (layername,))
        ut.ensuredir(fig_dpath)
        output_fpath = join(fig_dpath, output_fname)
        plt.savefig(output_fpath, bbox_inches='tight')

        output2_fname = 'features_%s_%s.png' % (layername, color_str)
        output2_fpath = join(results_path, output2_fname)
        ut.copy(output_fpath, output2_fpath, overwrite=True, verbose=ut.VERYVERBOSE)
        #plt.savefig(output_fpath, bbox_inches='tight')
        output_fpath_list.append(output_fpath)
    return output_fpath_list


def show_model_layer_weights(model, target, **kwargs):
    nn_layers = model.get_all_layers()
    cnn_layers = [layer_ for layer_ in nn_layers if hasattr(layer_, 'W')]
    layer = cnn_layers[target]
    all_weights = layer.W.get_value()
    layername = net_strs.make_layer_str(layer) + '_%d' % (target,)
    #layername = 'conv%d' % (index,)
    #print('layername = %r' % (layername,))
    import plottool as pt
    fig = show_convolutional_weights(all_weights, **kwargs)
    pt.set_figtitle(layername, subtitle='shape=%r' % (all_weights.shape,))
    return fig


def dump_model_layer_weights_img(model, target, dpath=None, **kwargs):
    import plottool as pt
    if dpath is None:
        dpath = model.get_epoch_diagnostic_dpath()
    fig = show_model_layer_weights(model, target, **kwargs)
    output_fpath = pt.save_figure(fig=fig, dpath=dpath)
    return output_fpath


def show_convolutional_weights(all_weights, use_color=None, limit=144, fnum=None, pnum=(1, 1, 1)):
    r"""
    Args:
        all_weights (?):
        use_color (bool):
        limit (int):

    CommandLine:
        python -m ibeis_cnn.draw_net --test-show_convolutional_weights --show
        python -m ibeis_cnn.draw_net --test-show_convolutional_weights --show --index=1

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_cnn.draw_net import *  # NOQA
        >>> from ibeis_cnn.draw_net import *  # NOQA
        >>> from ibeis_cnn import models
        >>> model = models.SiameseCenterSurroundModel(autoinit=True)
        >>> output_layer = model.get_output_layer()
        >>> nn_layers = layers.get_all_layers(output_layer)
        >>> weighted_layers = [layer for layer in nn_layers if hasattr(layer, 'W')]
        >>> index = ut.get_argval('--index', type_=int, default=0)
        >>> all_weights = weighted_layers[index].W.get_value()
        >>> print('all_weights.shape = %r' % (all_weights.shape,))
        >>> use_color = None
        >>> limit = 64
        >>> fig = show_convolutional_weights(all_weights, use_color, limit)
        >>> ut.show_if_requested()
    """
    import plottool as pt
    if fnum is None:
        fnum = pt.next_fnum()
    fig = pt.figure(fnum=fnum, pnum=pnum)
    # TODO: draw dense layers
    #import matplotlib.pyplot as plt
    #from mpl_toolkits.axes_grid1 import ImageGrid
    #import matplotlib.cm as cm
    # re-use the samtargete figure to save memory
    #fig = plt.figure(1)
    #ax1 = plt.axes(frameon=False)
    #ax1.get_xaxis().set_visible(False)
    #ax1.get_yaxis().set_visible(False)
    # Get shape of weights
    #print('all_weights.shape = %r' % (all_weights.shape,))
    num, channels, height, width = all_weights.shape
    #print('all_weights.shape = %r' % (all_weights.shape,))
    if use_color is None:
        # Try to infer if use_color should be shown
        use_color = (channels == 3)
    #print('use_color = %r' % (use_color,))

    stacked_img = make_conv_weight_image(all_weights, limit)
    #ax = fig.add_subplot(111)
    if len(stacked_img.shape) == 3 and stacked_img.shape[-1] == 1:
        stacked_img = stacked_img.reshape(stacked_img.shape[:-1])
    pt.imshow(stacked_img)
    #if use_color:
    #    #ax.imshow(stacked_img, interpolation='nearest')
    #else:
    #    #ax.imshow(stacked_img, cmap=cm.Greys_r, interpolation='nearest')
    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)
    return fig


def make_conv_weight_image(all_weights, limit=144):
    """ just makes the image ndarray of the weights """
    # Try to infer if use_color should be shown
    num, channels, height, width = all_weights.shape
    # Try to infer if use_color should be shown
    use_color = (channels == 3)
    # non-use_color features need to be flattened
    if not use_color:
        all_weights_ = all_weights.reshape(num * channels, height, width, 1)
    else:
        # convert from theano to cv2 BGR
        all_weights_ = utils.convert_theano_images_to_cv2_images(all_weights)
        # convert from BGR to RGB
        all_weights_ = all_weights_[..., ::-1]
        #cv2.cvtColor(all_weights_[-1], cv2.COLOR_BGR2RGB)

    # Limit all_weights_
    #num = all_weights_.shape[0]
    num, height, width, channels = all_weights_.shape
    if limit is not None and num > limit:
        all_weights_ = all_weights_[:limit]
        num = all_weights_.shape[0]

    # Convert weight values to image values
    all_max = utils.multiaxis_reduce(np.amax, all_weights_, startaxis=1)
    all_min = utils.multiaxis_reduce(np.amin, all_weights_, startaxis=1)
    all_domain = all_max - all_min
    broadcaster = (slice(None),) + (None,) * (len(all_weights_.shape) - 1)
    all_features = ((all_weights_ - all_min[broadcaster]) * (255.0 / all_domain[broadcaster])).astype(np.uint8)
    #import scipy.misc
    # resize feature, give them a border, and stack them together
    new_height, new_width = max(32, height), max(32, width)
    nbp_ = 1  # num border pixels
    _resized_features = np.array([
        cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        for img in all_features
    ])
    resized_features = _resized_features.reshape(num, new_height, new_width, channels)
    border_shape = (num, new_height + (nbp_ * 2), new_width + (nbp_ * 2), channels)
    bordered_features = np.zeros(border_shape, dtype=resized_features.dtype)
    bordered_features[:, nbp_:-nbp_, nbp_:-nbp_, :] = resized_features
    #img_list = bordered_features
    import plottool as pt
    stacked_img = pt.stack_square_images(bordered_features)
    return stacked_img
    #pt.imshow(stacked_img)


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis_cnn.draw_net
        python -m ibeis_cnn.draw_net --allexamples
        python -m ibeis_cnn.draw_net --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
