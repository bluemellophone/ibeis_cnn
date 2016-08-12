# -*- coding: utf-8 -*-
"""
Functions to create network diagrams from a list of Layers.

References:
    # Adapted from
    https://github.com/ebenolson/Lasagne/blob/master/examples/draw_net.py

    # TODO:
    https://github.com/dnouri/nolearn/blob/master/nolearn/lasagne/visualize.py
"""
from __future__ import absolute_import, division, print_function
from operator import itemgetter
from os.path import join, exists
import numpy as np
import cv2
import utool as ut
from ibeis_cnn import utils
print, rrr, profile = ut.inject2(__name__, '[ibeis_cnn.draw_net]')


def imwrite_theano_symbolic_graph(thean_expr):
    import theano
    graph_dpath = '.'
    graph_fname = 'symbolic_graph.png'
    graph_fpath = ut.unixjoin(graph_dpath, graph_fname)
    ut.ensuredir(graph_dpath)
    theano.printing.pydotprint(
        thean_expr, outfile=graph_fpath, var_with_name_simple=True)
    ut.startfile(graph_fpath)
    return graph_fpath


def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
    """

    References:
        # Taken from
        https://gist.github.com/craffel/2d727968c3aaebd10359

    Draw a neural network cartoon using matplotilb.

    Example:
        >>> fig = plt.figure(figsize=(12, 12))
        >>> draw_neural_net(fig.gca(), .1, .9, .1, .9, [4, 7, 2])

    :parameters:
        - ax : matplotlib.axes.AxesSubplot
            The axes on which to plot the cartoon (get e.g. by plt.gca())
        - left : float
            The center of the leftmost node(s) will be placed here
        - right : float
            The center of the rightmost node(s) will be placed here
        - bottom : float
            The center of the bottommost node(s) will be placed here
        - top : float
            The center of the topmost node(s) will be placed here
        - layer_sizes : list of int
            List of layer sizes, including input and output dimensionality
    """
    import matplotlib.pyplot as plt
    #n_layers = len(layer_sizes)
    v_spacing = (top - bottom) / float(max(layer_sizes))
    h_spacing = (right - left) / float(len(layer_sizes) - 1)
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2. + (top + bottom) / 2.
        for m in xrange(layer_size):
            circle = plt.Circle((n * h_spacing + left, layer_top - m * v_spacing), v_spacing / 4.,
                                color='w', ec='k', zorder=4)
            ax.add_artist(circle)
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing * (layer_size_a - 1) / 2. + (top + bottom) / 2.
        layer_top_b = v_spacing * (layer_size_b - 1) / 2. + (top + bottom) / 2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n * h_spacing + left, (n + 1) * h_spacing + left],
                                  [layer_top_a - m * v_spacing, layer_top_b - o * v_spacing], c='k')
                ax.add_artist(line)


def show_arch_nx_graph(layers, fnum=None, fullinfo=True):
    r"""

    CommandLine:
        python -m ibeis_cnn.draw_net show_arch_nx_graph:0 --show
        python -m ibeis_cnn.draw_net show_arch_nx_graph:1 --show

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_cnn.draw_net import *  # NOQA
        >>> from ibeis_cnn import models
        >>> model = models.mnist.MNISTModel(batch_size=128, output_dims=10,
        >>>                                 data_shape=(24, 24, 3))
        >>> model.initialize_architecture()
        >>> layers = model.get_all_layers()
        >>> show_arch_nx_graph(layers)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> ut.show_if_requested()

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_cnn.draw_net import *  # NOQA
        >>> from ibeis_cnn import models
        >>> model = models.SiameseCenterSurroundModel(autoinit=True)
        >>> layers = model.get_all_layers()
        >>> show_arch_nx_graph(layers)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> ut.show_if_requested()

    """
    import networkx as nx
    import plottool as pt
    import ibeis_cnn.__LASAGNE__ as lasange
    #from matplotlib import offsetbox
    #import matplotlib as mpl

    REMOVE_BATCH_SIZE = True
    from ibeis_cnn import net_strs

    def get_hex_color(layer_type):
        if 'Input' in layer_type:
            return '#A2CECE'
        if 'Conv2D' in layer_type:
            return '#7C9ABB'
        if 'Dense' in layer_type:
            return '#6CCF8D'
        if 'Pool' in layer_type:
            return '#9D9DD2'
        if 'SoftMax' in layer_type:
            return '#7E9FD9'
        else:
            return '#{0:x}'.format(hash(layer_type + 'salt') % 2 ** 24)

    node_dict = {}
    edge_list = []
    edge_attrs = ut.ddict(dict)

    # Make layer ids (ensure no duplicates)
    layer_to_id = {
        l: repr(l) if l.name is None else l.name
        for l in set(layers)
    }
    keys_ = layer_to_id.keys()
    dups = ut.find_duplicate_items(layer_to_id.values())
    for dupval, dupidxs in dups.items():
        newval_fmt = dupval + '_%d'
        for layer in ut.take(keys_, dupidxs):
            newid = ut.get_nonconflicting_string(newval_fmt, layer_to_id.values())
            layer_to_id[layer] = newid

    def layerid(layer):
        return layer_to_id[layer]

    for i, layer in enumerate(layers):
        layer_info = net_strs.get_layer_info(layer)
        layer_type = layer_info['classalias']

        key = layerid(layer)

        color = get_hex_color(layer_info['classalias'])
        # Make label
        lines = []
        if layer_info['name'] is not None:
            lines.append(layer_info['name'])
        lines.append(layer_info['classalias'])
        if fullinfo:
            for attr, val in layer_info['layer_attrs'].items():
                if attr == 'shape' and REMOVE_BATCH_SIZE:
                    val = val[1:]
                lines.append('{0}: {1}'.format(attr, val))

            nonlinearity = layer_info.get('nonlinearity')
            if nonlinearity is not None:
                alias_map = {
                    'LeakyRectify': 'LRU',
                }
                val = layer_info['nonlinearity']['type']
                val = alias_map.get(val, val)
                lines.append('nonlinearity:\n{0}'.format(val))

        label = '\n'.join(lines)

        # append node
        is_main_layer = len(layer.params) > 0
        #is_main_layer = len(lasange.layers.get_all_params(layer, trainable=True)) > 0
        if layer_info['classname'] in lasange.layers.normalization.__all__:
            is_main_layer = False
        if layer_info['classname'] in lasange.layers.special.__all__:
            is_main_layer = False

        if layer_type == 'Input':
            is_main_layer = True

        node_attr = dict(name=key, label=label, color=color,
                         fillcolor=color, style='filled',
                         is_main_layer=is_main_layer)

        node_attr['is_main_layer'] = is_main_layer
        node_attr['classalias'] = layer_info['classalias']

        if is_main_layer:
            if hasattr(layer, 'shape'):
                if len(layer.shape) == 3:
                    node_attr['out_size'] = (layer.shape[2],
                                             layer.shape[1])
                    node_attr['depth'] = layer.output_shape[0]
            if hasattr(layer, 'output_shape'):
                if len(layer.output_shape) == 4:
                    depth = layer.output_shape[1]
                    width, height = (layer.output_shape[3],
                                     layer.output_shape[2])
                    xshift = -width * (.1 / (depth ** (1 / 3))) / 3
                    yshift = height * (.1 / (depth ** (1 / 3))) / 2
                    node_attr['depth'] = depth
                    node_attr['xshift'] = xshift
                    node_attr['yshift'] = yshift
                    node_attr['out_size'] = (width, height)

                if len(layer.output_shape) == 2:
                    node_attr['out_size'] = (1,
                                             layer.output_shape[1])

        node_dict[key] = node_attr

        _input_layers = []
        if hasattr(layer, 'input_layers'):
            _input_layers += layer.input_layers
        if hasattr(layer, 'input_layer'):
            _input_layers += [layer.input_layer]

        for input_layer in _input_layers:
            parent_key = layerid(input_layer)
            edge = (parent_key, key)
            edge_list.append(edge)

    main_size_ = np.array((100, 100)) * 2
    sub_size = np.array((75, 50)) * 2

    # Setup scaled width and heights
    out_size_list = [v['out_size'] for v in node_dict.values() if 'out_size' in v]
    area_arr = np.prod(out_size_list, axis=1)
    main_outsize = np.array(out_size_list[area_arr.argmax()])
    scale = main_size_ / main_outsize

    scale_dense_max = .25
    scale_dense_min = 8

    for k, v in node_dict.items():
        if v['is_main_layer']:
            if 'out_size' in v:
                # Make dense layers more visible
                if v['classalias'] == 'Dense':
                    v['shape'] = 'rect'
                    v['width'] = scale_dense_min
                    if v['out_size'][1] > main_outsize[1]:
                        v['height'] =  v['out_size'][1] * scale[1] * scale_dense_max
                    elif v['out_size'][1] < scale_dense_min:
                        v['height'] = scale_dense_min * v['out_size'][1]
                else:
                    v['shape'] = 'stack'
                    #v['shape'] = 'rect'
                    v['width'] = v['out_size'][0] * scale[0]
                    v['height'] =  v['out_size'][1] * scale[1]
            else:
                v['shape'] = 'rect'
                v['width'] = main_size_[0]
                v['height'] = main_size_[1]
        else:
            #v['shape'] = 'ellipse'
            v['shape'] = 'rect'
            v['style'] = 'rounded'
            v['width'] = sub_size[0]
            v['height'] = sub_size[1]

    key_order = ut.take(layer_to_id, layers)
    node_dict = ut.dict_subset(node_dict, key_order)

    print(ut.repr3(node_dict))

    G = nx.DiGraph()
    #G.add_nodes_from(list(node_dict.keys()))
    G.add_nodes_from(node_dict.items())
    G.add_edges_from(edge_list)
    for key, val in edge_attrs.items():
        nx.set_edge_attributes(G, key, val)

    for n1, n2 in list(G.edges()):
        #if node_dict[n1]['is_main_layer']:
        #    G.edge[n1][n2]['constraint'] = 'true'
        if node_dict[n2]['is_main_layer']:
            G.edge[n1][n2]['constraint'] = 'true'
            #G.edge[n1][n2]['constraint'] = 'false'
        else:
            G.edge[n1][n2]['constraint'] = 'false'

    # Add invisible structure
    main_nodes = [key for key, val in
                  nx.get_node_attributes(G, 'is_main_layer').items() if val]

    main_children = {}

    between_edges = []
    for n1 in main_nodes:
        main_children[n1] = []
        #descendants = nx.descendants(G, n1)
        # Main nodes only place constraints on
        # nodes in the next main group. Not their own
        between = []
        next_main = None
        G.node[n1]['group'] = n1
        for (_, n2) in nx.bfs_edges(G, n1):
            if next_main is None:
                if n2 in main_nodes:
                    between = []
                    next_main = n2
                else:
                    G.node[n2]['group'] = n1
                    main_children[n1].append(n2)
            elif next_main is not None and n2 in main_nodes:
                #between.append(n2)
                break
            between.append(n2)
        between_edges.append((n1, between))

    # Custom position
    x = 0
    y = 1000
    #x_step = main_size_[0] * 1.5
    print('main_nodes = %s' % (ut.repr2(main_children),))

    main_nodes = ut.isect(list(nx.topological_sort(G)), main_nodes)
    xpad = main_size_[0] * .3
    ypad = main_size_[1] * .3

    # Draw each main node, and then put its children under it
    # Then move to the left and draw the next main node.
    cumwidth = 0
    for n1 in main_nodes:
        cumheight = 0

        maxwidth = G.node[n1]['width']
        for n2 in main_children[n1]:
            maxwidth = max(maxwidth, G.node[n2]['width'])

        cumwidth += xpad
        cumwidth += maxwidth / 2

        pos = np.array([x + cumwidth, y - cumheight])
        G.node[n1]['pos'] = pos
        G.node[n1]['pin'] = 'true'

        height = G.node[n1]['height']
        cumheight += height / 2

        for n2 in main_children[n1]:
            height = G.node[n2]['height']
            cumheight += ypad
            cumheight += height / 2
            pos = np.array([x + cumwidth, y - cumheight])
            G.node[n2]['pos'] = pos
            G.node[n2]['pin'] = 'true'
            cumheight += height / 2

        cumwidth += maxwidth / 2

    # Pin everybody
    nx.set_node_attributes(G, 'pin', 'true')
    layoutkw = dict(prog='neato', splines='line')
    #layoutkw = dict(prog='neato', splines='spline')
    layoutkw = dict(prog='neato', splines='ortho')
    G_ = G.copy()
    # delete lables for positioning
    _labels = nx.get_node_attributes(G_, 'label')
    ut.nx_delete_node_attr(G_, 'label')
    nx.set_node_attributes(G_, 'label', '')
    if 0:
        G_.remove_edges_from(list(G_.edges()))
    else:
        layout_info = pt.nx_agraph_layout(G_, inplace=True, **layoutkw)  # NOQA
    # reset labels
    if 1:
        nx.set_node_attributes(G_, 'label', _labels)
    _ = pt.show_nx(G_, fontsize=10, arrow_width=.3, layout='custom', fnum=fnum)  # NOQA
    pt.adjust_subplots2(top=1, bot=0, left=0, right=1)

    #for n1, n2s in between_edges:
    #    for n2 in n2s:
    #        style = 'invis'
    #        #style = 'visible'
    #        if not G.has_edge(n1, n2):
    #            #G.add_edge(n1, n2, {'style': 'invis', 'constraint': 'true'})
    #            G.add_edge(n1, n2, {'style': style, 'constraint': 'true',
    #                                'color': '#00FF00'})
    #        else:
    #            G.edge[n1][n2]['constraint'] = 'true'

    #layoutkw = dict(prog='dot', splines='spline', rankdir='LR', nodesep=1,
    #                rank='same',
    #                ranksep=1.5)
    #pt.show_nx(G, fontsize=6, arrow_width=.5, layoutkw=layoutkw)
    #pt.show_nx(G, layoutkw=dict(prog='neato'), fontsize=6)


def pydot_to_image(pydot_graph):
    """
    References:
        http://stackoverflow.com/questions/4596962/display-graph-without-saving-using-pydot
    """
    from PIL import Image
    from six.moves import StringIO
    #from cStringIO import StringIO
    png_str = pydot_graph.create_png(prog='dot')
    sio = StringIO()
    sio.write(png_str)
    sio.seek(0)
    pil_img = Image.open(sio)
    img = np.asarray(pil_img.convert('RGB'))
    img = img[..., ::-1]  # to bgr
    pil_img.close()
    sio.close()
    return img


#def make_architecture_image(layers, **kwargs):
#    """
#    Args:
#        layers (list): List of the layers, as obtained from lasange.layers.get_all_layers

#    Kwargs:
#        see docstring of make_architecture_pydot_graph for other options

#    References:
#        http://stackoverflow.com/questions/4596962/display-graph-without-saving-using-pydot

#    CommandLine:
#        python -m ibeis_cnn.draw_net --test-make_architecture_image --show

#    Example:
#        >>> # ENABLE_DOCTEST
#        >>> from ibeis_cnn.draw_net import *  # NOQA
#        >>> from ibeis_cnn import models
#        >>> model = models.SiameseCenterSurroundModel(autoinit=True)
#        >>> #model = models.DummyModel(autoinit=True)
#        >>> layers = model.get_all_layers()
#        >>> # execute function
#        >>> kwargs = {}
#        >>> img = make_architecture_image(layers, **kwargs)
#        >>> print(img.shape)
#        >>> ut.quit_if_noshow()
#        >>> import plottool as pt
#        >>> pt.imshow(img)
#        >>> ut.show_if_requested()
#    """
#    # from IPython.display import Image  # needed to render in notebook
#    pydot_graph = make_architecture_pydot_graph(layers, **kwargs)
#    img = pydot_to_image(pydot_graph)
#    return img


#def imwrite_arch(layers, fpath, **kwargs):
#    """
#    Draws a network diagram to a file

#    Args:
#        layers (list): List of the layers, as obtained from lasange.layers.get_all_layers
#        fpath (str): The fpath to save output to.

#        Kwargs:
#            see docstring of make_architecture_pydot_graph for other options

#    CommandLine:
#        python -m ibeis_cnn.draw_net --test-imwrite_arch --show

#    Example:
#        >>> # ENABLE_DOCTEST
#        >>> from ibeis_cnn.draw_net import *  # NOQA
#        >>> from ibeis_cnn import models
#        >>> #model = models.DummyModel(autoinit=True)
#        >>> model = models.SiameseCenterSurroundModel(autoinit=True)
#        >>> layers = model.get_all_layers()
#        >>> fpath = ut.unixjoin(ut.ensure_app_resource_dir('ibeis_cnn'), 'tmp.png')
#        >>> # execute function
#        >>> imwrite_arch(layers, fpath)
#        >>> ut.quit_if_noshow()
#        >>> ut.startfile(fpath)
#    """
#    pydot_graph = make_architecture_pydot_graph(layers, **kwargs)

#    ext = fpath[fpath.rfind('.') + 1:]
#    with open(fpath, 'w') as fid:
#        fid.write(pydot_graph.create(format=ext))


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
        >>> from ibeis_cnn import models
        >>> from lasagne import layers
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
    fig = pt.figure(fnum=fnum, pnum=pnum, docla=True)
    num, channels, height, width = all_weights.shape
    if use_color is None:
        # Try to infer if use_color should be shown
        use_color = (channels == 3)

    stacked_img = make_conv_weight_image(all_weights, limit)
    #ax = fig.add_subplot(111)
    if len(stacked_img.shape) == 3 and stacked_img.shape[-1] == 1:
        stacked_img = stacked_img.reshape(stacked_img.shape[:-1])
    pt.imshow(stacked_img)
    return fig


def make_conv_weight_image(all_weights, limit=144):
    """ just makes the image ndarray of the weights """
    import vtool as vt
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
    normalize_individually = False
    if normalize_individually:
        # Normalize each feature individually
        all_max = vt.multiaxis_reduce(np.amax, all_weights_, startaxis=1)
        all_min = vt.multiaxis_reduce(np.amin, all_weights_, startaxis=1)
        all_domain = all_max - all_min
        extra_dims = (None,) * (len(all_weights_.shape) - 1)
        broadcaster = (slice(None),) + extra_dims
        all_features = ((all_weights_ - all_min[broadcaster]) *
                        (255.0 / all_domain[broadcaster])).astype(np.uint8)
    else:
        # Normalize jointly across all filters
        _max = all_weights_.max()
        _min = all_weights_.min()
        _domain = _max - _min
        all_features = ((all_weights_ - _min) * (255.0 / _domain)).astype(np.uint8)

    #import scipy.misc
    # resize feature, give them a border, and stack them together
    new_height, new_width = max(32, height), max(32, width)
    nbp_ = 1  # num border pixels
    _resized_features = np.array([
        cv2.resize(img, (new_width, new_height),
                   interpolation=cv2.INTER_NEAREST)
        for img in all_features
    ])
    resized_features = _resized_features.reshape(
        num, new_height, new_width, channels)
    border_shape = (num, new_height + (nbp_ * 2),
                    new_width + (nbp_ * 2), channels)
    bordered_features = np.zeros(border_shape, dtype=resized_features.dtype)
    bordered_features[:, nbp_:-nbp_, nbp_:-nbp_, :] = resized_features
    #img_list = bordered_features
    stacked_img = vt.stack_square_images(bordered_features)
    return stacked_img


def output_confusion_matrix(X_test, results_path, test_results, model,
                            **kwargs):
    """ currently hacky implementation, fix it later """
    loss, accu_test, prob_list, auglbl_list, pred_list, conf_list = test_results
    # Output confusion matrix
    mapping_fn = None
    if model is not None:
        mapping_fn = getattr(model, 'label_order_mapping', None)
    # TODO: THIS NEEDS TO BE FIXED
    label_list = list(range(kwargs.get('output_dims')))
    # Encode labels if avaialble
    #encoder = kwargs.get('encoder', None)
    encoder = getattr(model, 'encoder', None)
    if encoder is not None:
        label_list = encoder.inverse_transform(label_list)
    # Make confusion matrix (pass X to write out failed cases)
    show_confusion_matrix(
        auglbl_list, pred_list, label_list, results_path, mapping_fn, X_test)


def save_confusion_matrix(results_path, correct_y, predict_y, category_list, mapping_fn=None, data_x=None):
    import plottool as pt
    fig = show_confusion_matrix(
        correct_y, predict_y, category_list, mapping_fn=mapping_fn, data_x=data_x)
    output_fpath = join(results_path, 'confusion.png')
    pt.save_figure(fig, fpath=output_fpath)
    return output_fpath


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

    TODO FIXME and simplify
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
        category_mapping = {key: index for index,
                            key in enumerate(category_list)}
        category_list_ = category_list
    else:
        category_mapping = mapping_fn(category_list)
        assert all([category in category_mapping.keys()
                    for category in category_list]), 'Not all categories are mapped'
        values = list(category_mapping.values())
        assert len(list(set(values))) == len(
            values), 'Mapped categories have a duplicate assignment'
        assert 0 in values, 'Mapped categories must have a 0 index'
        temp = list(category_mapping.iteritems())
        temp = sorted(temp, key=itemgetter(1))
        category_list_ = [t[0] for t in temp]

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
            example_name = '%s^SEEN_INCORRECTLY_AS^%s' % (
                example_correct_label, example_predict_label, )
            if example_name not in counters.keys():
                counters[example_name] = 0
            counter = counters[example_name]
            counters[example_name] += 1
            example_name = '%s^%d.png' % (example_name, counter)
            example_path = join(confused_examples, example_name)
            # TODO: make write confused examples function
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
    plt.subplots_adjust(left=margin_small, right=margin_large,
                        bottom=margin_small, top=margin_large)
    plt.xlabel('Predicted')
    plt.ylabel('Correct')
    return fig


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
