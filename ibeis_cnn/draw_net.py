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
from operator import itemgetter
import numpy as np
import cv2
from os.path import join, exists
import utool as ut
from lasagne import layers


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


def draw_theano_symbolic_expression(thean_expr):
    import theano
    graph_dpath = '.'
    graph_fname = 'symbolic_graph.png'
    graph_fpath = ut.unixjoin(graph_dpath, graph_fname)
    ut.ensuredir(graph_dpath)
    theano.printing.pydotprint(thean_expr, outfile=graph_fpath, var_with_name_simple=True)
    ut.startfile(graph_fpath)
    return graph_fpath


def get_pydot_graph(layers, output_shape=True, verbose=False):
    """
    Creates a PyDot graph of the network defined by the given layers.

    Args:
        - layers : list
            List of the layers, as obtained from lasange.layers.get_all_layers
        - output_shape: (default `True`)
            If `True`, the output shape of each layer will be displayed.
        - verbose: (default `False`)
            If `True`, layer attributes like filter shape, stride, etc.
            will be displayed.
        - verbose:

    Returns:
        - pydot_graph : PyDot object containing the graph

    CommandLine:
        python -m ibeis_cnn.draw_net --test-get_pydot_graph

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.draw_net import *  # NOQA
        >>> from ibeis_cnn import models
        >>> # build test data
        >>> self = models.IdentificationModel()
        >>> output_layer = self.build_model(8, 256, 256, 3, 3)
        >>> layers = self.network_layers
        >>> output_shape = True
        >>> verbose = False
        >>> # execute function
        >>> pydot_graph = get_pydot_graph(layers, output_shape, verbose)
        >>> # verify results
        >>> result = str(pydot_graph)
        >>> print(result)
    """
    import pydot
    pydot_graph = pydot.Dot('Network', graph_type='digraph')
    pydot_nodes = {}
    pydot_edges = []
    for i, layer in enumerate(layers):
        layer_type = '{0}'.format(layer.__class__.__name__)
        key = repr(layer)
        label = layer_type
        color = get_hex_color(layer_type)
        if verbose:
            for attr in ['num_filters', 'num_units', 'ds',
                         'filter_shape', 'stride', 'strides', 'p']:
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
                'Output shape: {0}'.format(layer.get_output_shape())
        pydot_nodes[key] = pydot.Node(key,
                                      label=label,
                                      shape='record',
                                      fillcolor=color,
                                      style='filled',
                                      )

        if hasattr(layer, 'input_layers'):
            for input_layer in layer.input_layers:
                pydot_edges.append([repr(input_layer), key])

        if hasattr(layer, 'input_layer'):
            pydot_edges.append([repr(layer.input_layer), key])

    for node in pydot_nodes.values():
        pydot_graph.add_node(node)
    for edge in pydot_edges:
        pydot_graph.add_edge(
            pydot.Edge(pydot_nodes[edge[0]], pydot_nodes[edge[1]]))
    return pydot_graph


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
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.draw_net import *  # NOQA
        >>> from ibeis_cnn import models
        >>> self = models.IdentificationModel()
        >>> output_layer = self.build_model(8, 256, 256, 3, 3)
        >>> layers = layers = self.network_layers
        >>> filename = 'tmp.png'
        >>> # execute function
        >>> draw_to_file(layers, filename)
        >>> ut.quit_if_noshow()
        >>> ut.start_file(filename)
    """
    dot = get_pydot_graph(layers, **kwargs)

    ext = filename[filename.rfind('.') + 1:]
    with open(filename, 'w') as fid:
        fid.write(dot.create(format=ext))


def draw_to_notebook(layers, **kwargs):
    """
    Draws a network diagram in an IPython notebook
    :parameters:
        - layers : list
            List of the layers, as obtained from lasange.layers.get_all_layers
        - **kwargs: see docstring of get_pydot_graph for other options
    """
    from IPython.display import Image  # needed to render in notebook

    dot = get_pydot_graph(layers, **kwargs)
    return Image(dot.create_png())


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
    plt.savefig(join(results_path, 'confusion.png'))


def show_convolutional_layers(output_layer, results_path, color=False, limit=150, target=None, epoch=None):
    nn_layers = layers.get_all_layers(output_layer)
    cnn_layers = []
    for layer in nn_layers:
        layer_type = str(type(layer))
        # Only print convolutional layers
        if 'Conv2DCCLayer' not in layer_type:
            continue
        cnn_layers.append(layer)

    weights_list = [layer.W.get_value() for layer in cnn_layers]
    show_convolutional_features(weights_list, results_path, color=color, limit=limit, target=target, epoch=epoch)


def show_convolutional_features(weights_list, results_path, color=False, limit=150, target=None, epoch=None):
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid
    import matplotlib.cm as cm
    for index, all_weights in enumerate(weights_list):
        if target is not None and target != index:
            continue
        # re-use the same figure to save memory
        fig = plt.figure(1)
        ax1 = plt.axes(frameon=False)
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        # Get shape of weights
        num, channels, height, width = all_weights.shape
        # non-color features need to be flattened
        if not color:
            all_weights = all_weights.reshape(num * channels, height, width)
            num, height, width = all_weights.shape
        # Limit all_weights
        if limit is not None and num > limit:
            all_weights = all_weights[:limit]
            if color:
                num, channels, height, width = all_weights.shape
            else:
                num, height, width = all_weights.shape
        # Find how many features and build grid
        dim = int(np.ceil(np.sqrt(num)))
        grid = ImageGrid(fig, 111, nrows_ncols=(dim, dim))

        # Build grid
        for f, feature in enumerate(all_weights):
            # get all the weights and scale them to dimensions that can be shown
            if color:
                feature = feature[::-1]  # Rotate BGR to RGB
                feature = cv2.merge(feature)
            fmin, fmax = np.min(feature), np.max(feature)
            domain = fmax - fmin
            feature = (feature - fmin) * (255. / domain)
            feature = feature.astype(np.uint8)
            if color:
                grid[f].imshow(feature, interpolation='nearest')
            else:
                grid[f].imshow(feature, cmap=cm.Greys_r, interpolation='nearest')

        for j in range(dim * dim):
            grid[j].get_xaxis().set_visible(False)
            grid[j].get_yaxis().set_visible(False)

        color_str = 'color' if color else 'gray'
        if epoch is None:
            epoch = 'X'
        output_fname = 'features_conv%d_epoch_%s_%s.png' % (index, epoch, color_str)
        fig_dpath = join(results_path, 'feature_figures')
        ut.ensuredir(fig_dpath)
        output_fpath = join(fig_dpath, output_fname)
        plt.savefig(output_fpath, bbox_inches='tight')

        output_fname = 'features_conv%d_%s.png' % (index, color_str)
        output_fpath = join(results_path, output_fname)
        plt.savefig(output_fpath, bbox_inches='tight')


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
