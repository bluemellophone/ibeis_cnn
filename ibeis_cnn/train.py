#!/usr/bin/env python
"""
constructs the Theano optimization and trains a learning model,
optionally by initializing the network with pre-trained weights.


CommandLine:
    python -m ibeis_cnn.train --test-train_patchmatch_pz
    python -m ibeis_cnn.train --test-train_patchmatch_pz --vtd
    python -m ibeis_cnn.train --test-train_patchmatch_pz --vtd --max-examples=5 --batch_size=128 --learning_rate .0000001
    python -m ibeis_cnn.train --test-train_patchmatch_pz --db NNP_Master3
    python -m ibeis_cnn.train --test-train_patchmatch_pz --db NNP_Master3 --vtd
"""
from __future__ import absolute_import, division, print_function
from ibeis_cnn import utils
from ibeis_cnn import batch_processing as batch
from ibeis_cnn import draw_net
from ibeis_cnn import models
from ibeis_cnn import ibsplugin
from six.moves import input
import sys
import time
import theano
import numpy as np
import cPickle as pickle
from lasagne import layers
from sklearn import preprocessing

import utool as ut
import six
from os.path import join, abspath, dirname, exists
print, rrr, profile = ut.inject2(__name__, '[ibeis_cnn.train]')


def train(data_fpath, labels_fpath, model, weights_fpath, results_dpath,
          pretrained_weights_fpath=None, pretrained_kwargs=False, **kwargs):
    """
    Driver function

    Args:
        data_fpath (?):
        labels_fpath (?):
        model (?):
        weights_fpath (?):
        pretrained_weights_fpath (None):
        pretrained_kwargs (bool):
    """

    ######################################################################################

    # Load the data
    print('\n[train] --- LOADING DATA ---')
    sys.stdout.flush()
    print('data_fpath = %r' % (data_fpath,))
    print('labels_fpath = %r' % (labels_fpath,))
    data, labels = utils.load(data_fpath, labels_fpath)

    # Ensure results dir
    weights_dpath = dirname(abspath(weights_fpath))
    ut.ensuredir(weights_dpath)
    ut.ensuredir(results_dpath)
    if pretrained_weights_fpath is not None:
        pretrained_weights_dpath = dirname(abspath(pretrained_weights_fpath))
        ut.ensuredir(pretrained_weights_dpath)
    print('[train] kwargs = \n' + (ut.dict_str(kwargs)))

    # Training parameters defaults
    utils._update(kwargs, 'center',                  True)
    utils._update(kwargs, 'encode',                  True)
    utils._update(kwargs, 'learning_rate',           0.01)
    utils._update(kwargs, 'momentum',                0.9)
    utils._update(kwargs, 'batch_size',              128)
    utils._update(kwargs, 'patience',                10)
    utils._update(kwargs, 'run_test',                10)  # Test every 10 epochs
    utils._update(kwargs, 'max_epochs',              kwargs.get('patience') * 10)
    utils._update(kwargs, 'regularization',          None)
    utils._update(kwargs, 'output_dims',             None)
    utils._update(kwargs, 'show_confusion',          True)
    utils._update(kwargs, 'show_features',           True)

    # Automatically figure out how many classes
    if kwargs.get('output_dims') is None:
        kwargs['output_dims'] = len(list(np.unique(labels)))

    # print('[load] adding channels...')
    # data = utils.add_channels(data)
    print('[train] memory(data) = %r' % (ut.get_object_size_str(data),))
    print('[train] data.shape   = %r' % (data.shape,))
    print('[train] data.dtype   = %r' % (data.dtype,))
    print('[train] labels.shape = %r' % (labels.shape,))
    print('[train] labels.dtype = %r' % (labels.dtype,))

    labelhist = {key: len(val) for key, val in six.iteritems(ut.group_items(labels, labels))}
    print('[train] label histogram = \n' + ut.dict_str(labelhist))

    print('\n[train] --- CONFIGURE SETTINGS ---')
    print('[train] kwargs = \n' + (ut.dict_str(kwargs)))
    # Encoding labels
    kwargs['encoder'] = None
    if kwargs.get('encode', False):
        kwargs['encoder'] = preprocessing.LabelEncoder()
        kwargs['encoder'].fit(labels)

    # draw_net.show_image_from_data(data)

    # Split the dataset into training and validation
    print('\n[train] --- SAMPLING DATA ---')
    print('[train] creating train, validation datasaets...')
    data_per_label = getattr(model, 'data_per_label', 1)
    _tup = utils.train_test_split(data, labels, eval_size=0.2, data_per_label=data_per_label)
    X_train, y_train, X_valid, y_valid = _tup
    _tup = utils.train_test_split(X_train, y_train, eval_size=0.1, data_per_label=data_per_label)
    X_train, y_train, X_test, y_test = _tup

    # Build and print the model
    print('\n[train] --- BUILDING MODEL ---')
    kwargs['model_shape'] = data.shape
    input_cases, input_height, input_width, input_channels = kwargs.get('model_shape', None)  # SHOULD ERROR IF NOT SET
    output_layer = model.build_model(
        kwargs['batch_size'], input_width, input_height,
        input_channels, kwargs['output_dims'])
    utils.print_layer_info(output_layer)

    # Create the Theano primitives
    print('\n[train] --- COMPILING SYMBOLIC THEANO FUNCTIONS ---')
    print('[model] creating Theano primitives...')
    learning_rate_theano = theano.shared(utils.float32(kwargs.get('learning_rate')))
    # create theano symbolic expressions that define the network
    theano_funcs = batch.create_theano_funcs(learning_rate_theano, output_layer, model, **kwargs)

    print('\n[train] --- WEIGHT INITIALIZATION ---')
    # Load the pretrained model if specified
    if pretrained_weights_fpath is not None and exists(pretrained_weights_fpath):
        print('[model] loading pretrained weights from %s' % (pretrained_weights_fpath))
        with open(pretrained_weights_fpath, 'rb') as pfile:
            kwargs_ = pickle.load(pfile)
            pretrained_weights = kwargs_.get('best_weights', None)
            layers.set_all_param_values(output_layer, pretrained_weights)
            if pretrained_kwargs:
                kwargs = kwargs_
    else:
        print('[model] no pretrained weights')

    # Center the data by subtracting the mean (AFTER KWARGS UPDATE)
    if kwargs.get('center'):
        print('[train] applying data centering...')
        utils._update(kwargs, 'center_mean', np.mean(X_train, axis=0))
        # utils._update(kwargs, 'center_std', np.std(X_train, axis=0))
        utils._update(kwargs, 'center_std', 255.0)
    else:
        utils._update(kwargs, 'center_mean', 0.0)
        utils._update(kwargs, 'center_std', 1.0)

    # Should these not be in kwargs?
    utils._update(kwargs, 'best_weights',        None)
    utils._update(kwargs, 'best_valid_accuracy', 0.0)
    utils._update(kwargs, 'best_test_accuracy',  0.0)
    utils._update(kwargs, 'best_train_loss',     np.inf)
    utils._update(kwargs, 'best_valid_loss',     np.inf)
    utils._update(kwargs, 'best_valid_accuracy', 0.0)
    utils._update(kwargs, 'best_test_accuracy',  0.0)
    training_loop(X_train, y_train, X_valid, y_valid, X_test, y_test,
                  theano_funcs, model, output_layer,
                  results_dpath, weights_fpath,
                  learning_rate_theano, **kwargs)


def training_loop(X_train, y_train, X_valid, y_valid, X_test, y_test,
                  theano_funcs, model, output_layer,
                  results_dpath, weights_fpath,
                  learning_rate_theano, **kwargs):
    print('\n[train] --- TRAINING LOOP ---')
    # Begin training the neural network
    vals = (utils.get_current_time(), kwargs.get('learning_rate'), )
    print('\n[train] starting training at %s with learning rate %.9f' % vals)
    printcol_info = utils.get_printcolinfo(kwargs.get('requested_headers', None))
    utils.print_header_columns(printcol_info)
    theano_backprop, theano_forward, theano_predict = theano_funcs
    epoch = 0
    epoch_marker = epoch
    while True:
        try:
            # Start timer
            t0 = time.time()

            # Show first weights before any training
            if kwargs.get('show_features'):
                draw_net.show_convolutional_layers(output_layer, results_dpath,
                                                   color=True, target=0, epoch=epoch)

            # Get the augmentation function, if there is one for this model
            augment_fn = getattr(model, 'augment', None)

            # compute the loss over all training and validation batches
            train_loss = batch.process_train(X_train, y_train, theano_backprop,
                                             model=model, augment=augment_fn,
                                             rand=True, **kwargs)

            data_valid = batch.process_valid(X_valid, y_valid, theano_forward,
                                             model=model, augment=None,
                                             rand=False, **kwargs)
            valid_loss, valid_accuracy = data_valid

            # If the training loss is nan, the training has diverged
            if np.isnan(train_loss):
                print('\n[train] training diverged\n')
                break

            # Calculate request_test before adding to the epoch counter
            request_test = kwargs.get('run_test') is not None and epoch % kwargs.get('run_test') == 0
            # Increment the epoch
            epoch += 1

            # Is this model the best we've ever seen?
            best_found = valid_loss < kwargs.get('best_valid_loss')
            if best_found:
                kwargs['best_epoch'] = epoch
                epoch_marker = epoch
                kwargs['best_weights'] = layers.get_all_param_values(output_layer)

            # compute the loss over all testing batches
            # if request_test or best_found:
            if request_test:
                train_determ_loss = batch.process_train(X_train, y_train, theano_forward,
                                                        model=model, augment=augment_fn,
                                                        rand=True, **kwargs)

                # If we want to output the confusion matrix, give the results path
                results_dpath_ = results_dpath if kwargs.get('show_confusion', False) else None
                test_accuracy = batch.process_test(
                    X_test, y_test, theano_forward, results_dpath_,
                    model=model, augment=None, rand=False, **kwargs)
                # Output the layer 1 features
                if kwargs.get('show_features'):
                    draw_net.show_convolutional_layers(output_layer, results_dpath,
                                                       color=True, target=0, epoch=epoch)
                    # draw_net.show_convolutional_layers(output_layer, results_dpath,
                    #                                 color=False, target=0, epoch=epoch)
            else:
                train_determ_loss = None
                test_accuracy = None

            # Running tab for what the best model
            if train_loss < kwargs.get('best_train_loss'):
                kwargs['best_train_loss'] = train_loss
            if train_determ_loss > kwargs.get('best_determ_loss'):
                kwargs['best_determ_loss'] = train_determ_loss
            if valid_loss < kwargs.get('best_valid_loss'):
                kwargs['best_valid_loss'] = valid_loss
            if valid_accuracy > kwargs.get('best_valid_accuracy'):
                kwargs['best_valid_accuracy'] = valid_accuracy
            if test_accuracy > kwargs.get('best_test_accuracy'):
                kwargs['best_test_accuracy'] = test_accuracy

            # Learning rate schedule update
            if epoch >= epoch_marker + kwargs.get('patience'):
                epoch_marker = epoch
                utils.set_learning_rate(learning_rate_theano, model.learning_rate_update)
                utils.print_header_columns(printcol_info)

            # End timer
            t1 = time.time()

            # Print the epoch
            duration = t1 - t0
            epoch_info = {
                'train_loss'          : train_loss,
                'train_determ_loss'   : train_determ_loss,
                'valid_loss'          : valid_loss,
                'valid_accuracy'      : valid_accuracy,
                'test_accuracy'       : test_accuracy,
                'epoch'               : epoch,
                'duration'            : duration,
                'best_train_loss'     : kwargs['best_train_loss'],
                'best_valid_loss'     : kwargs['best_valid_loss'],
                'best_valid_accuracy' : kwargs['best_valid_accuracy'],
                'best_test_accuracy'  : kwargs['best_test_accuracy'],
            }
            utils.print_epoch_info(printcol_info, epoch_info)
            #ut.embed()

            # Break on max epochs
            if epoch >= kwargs.get('max_epochs'):
                print('\n[train] maximum number of epochs reached\n')
                break
        except KeyboardInterrupt:
            # We have caught the Keyboard Interrupt, figure out what resolution mode
            print('\n[train] Caught CRTL+C')
            resolution = ''
            while not (resolution.isdigit()):
                print('\n[train] What do you want to do?')
                print('[train]     0 - Continue')
                print('[train]     1 - Shock weights')
                print('[train]     2 - Save best weights')
                print('[train]     3 - Stop network training')
                resolution = input('[train] Resolution: ')
            resolution = int(resolution)
            # We have a resolution
            if resolution == 0:
                print('resuming training...')
            elif resolution == 1:
                # Shock the weights of the network
                utils.shock_network(output_layer)
                epoch_marker = epoch
                utils.set_learning_rate(learning_rate_theano, model.learning_rate_shock)
                utils.print_header_columns(printcol_info)
            elif resolution == 2:
                # Save the weights of the network
                utils.save_model(kwargs, weights_fpath)
            else:
                # Terminate the network training
                raise

    # Save the best network
    utils.save_model(kwargs, weights_fpath)


def train_patchmatch_pz():
    r"""

    CommandLine:
        python -m ibeis_cnn.train --test-train_patchmatch_pz
        python -m ibeis_cnn.train --test-train_patchmatch_pz --vtd
        python -m ibeis_cnn.train --test-train_patchmatch_pz --vtd --max-examples=5 --batch_size=128 --learning_rate .0000001
        python -m ibeis_cnn.train --test-train_patchmatch_pz --db NNP_Master3
        python -m ibeis_cnn.train --test-train_patchmatch_pz --db NNP_Master3 --vtd

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.train import *  # NOQA
        >>> train_patchmatch_pz()
    """
    max_examples = ut.get_argval('--max-examples', type_=int, default=None)
    print('[train] train_patchmatch_pz')
    print('[train] max examples = {}'.format(max_examples))

    with ut.Indenter('[LOAD IBEIS DB]'):
        import ibeis
        ibs = ibeis.opendb(defaultdb='PZ_MTEST')

    with ut.Indenter('[ENSURE TRAINING DATA]'):
        pathtup = ibsplugin.get_patchmetric_training_fpaths(ibs, max_examples=max_examples)
        data_fpath, labels_fpath, training_dpath = pathtup

    model = models.SiameseModel()
    config = dict(
        patience=100,
        batch_size=ut.get_argval('--batch_size', type_=int, default=128),
        learning_rate=ut.get_argval('--learning_rate', type_=float, default=.001),
        show_confusion=False,
        requested_headers=['epoch', 'train_loss', 'valid_loss', 'trainval_rat', 'duration']
    )
    nets_dir = ibs.get_neuralnet_dir()
    ut.ensuredir(nets_dir)
    weights_fpath = join(training_dpath, 'ibeis_cnn_weights.pickle')
    train(data_fpath, labels_fpath, model, weights_fpath, training_dpath, **config)


#@ibeis.register_plugin()
def train_identification_pz():
    r"""

    CommandLine:
        python -m ibeis_cnn.train --test-train_identification_pz

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.train import *  # NOQA
        >>> train_identification_pz()
    """
    print('get_identification_decision_training_data')
    import ibeis
    ibs = ibeis.opendb('NNP_Master3')
    base_size = 128
    #max_examples = 1001
    #max_examples = None
    max_examples = 400
    data_fpath, labels_fpath, training_dpath = ibsplugin.get_identify_training_fpaths(ibs, base_size=base_size, max_examples=max_examples)

    model = models.SiameseModel()
    config = dict(
        patience=50,
        batch_size=32,
        learning_rate=.03,
        show_confusion=False,
    )
    nets_dir = ibs.get_neuralnet_dir()
    ut.ensuredir(nets_dir)
    weights_fpath = join(training_dpath, 'ibeis_cnn_weights.pickle')
    train(data_fpath, labels_fpath, model, weights_fpath, training_dpath, **config)
    #X = k


def train_viewpoint_pz():
    r"""
    CommandLine:
        python -m ibeis_cnn.train --test-train_viewpoint_pz

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.train import *  # NOQA
        >>> train_viewpoint_pz()
    """
    project_name             = 'viewpoint_pz'
    model                    = models.ViewpointModel()
    root                     = abspath(join('..', 'data'))

    train_data_fpath         = join(root, 'numpy',   project_name, 'X.npy')
    train_labels_fpath       = join(root, 'numpy',   project_name, 'y.npy')
    results_dpath            = join(root, 'results', project_name)
    weights_fpath            = join(root, 'nets',    project_name, 'ibeis_cnn_weights.pickle')
    pretrained_weights_fpath = join(root, 'nets',    project_name, 'ibeis_cnn_weights.pickle')  # NOQA
    config                   = {
        'patience': 10,
        'regularization': 0.0001,
        'pretrained_weights_fpath': pretrained_weights_fpath,
    }
    train(train_data_fpath, train_labels_fpath, model, weights_fpath, results_dpath, **config)


def train_quality_pz():
    r"""
    CommandLine:
        python -m ibeis_cnn.train --test-train_quality_pz

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.train import *  # NOQA
        >>> train_quality_pz()
    """
    project_name             = 'quality_pz'
    model                    = models.QualityModel()
    root                     = abspath(join('..', 'data'))

    train_data_fpath         = join(root, 'numpy',   project_name, 'X.npy')
    train_labels_fpath       = join(root, 'numpy',   project_name, 'y.npy')
    results_dpath            = join(root, 'results', project_name)
    weights_fpath            = join(root, 'nets',    project_name, 'ibeis_cnn_weights.pickle')
    pretrained_weights_fpath = join(root, 'nets',    project_name, 'ibeis_cnn_weights.pickle')  # NOQA
    config                   = {
        'patience': 10,
        'regularization': 0.0001,
        'pretrained_weights_fpath': pretrained_weights_fpath,
    }
    train(train_data_fpath, train_labels_fpath, model, weights_fpath, results_dpath, **config)


def train_viewpoint():
    r"""
    CommandLine:
        python -m ibeis_cnn.train --test-train_viewpoint

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.train import *  # NOQA
        >>> train_viewpoint()
    """
    project_name             = 'viewpoint'
    model                    = models.ViewpointModel()
    root                     = abspath(join('..', 'data'))

    train_data_fpath         = join(root, 'numpy',   project_name, 'X.npy')
    train_labels_fpath       = join(root, 'numpy',   project_name, 'y.npy')
    results_dpath            = join(root, 'results', project_name)
    weights_fpath            = join(root, 'nets',    project_name, 'ibeis_cnn_weights.pickle')
    pretrained_weights_fpath = join(root, 'nets',    project_name, 'ibeis_cnn_weights.pickle')  # NOQA
    config                   = {
        'patience': 10,
        'regularization': 0.0001,
        'pretrained_weights_fpath': pretrained_weights_fpath,
    }
    train(train_data_fpath, train_labels_fpath, model, weights_fpath, results_dpath, **config)


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis_cnn.train
        python -m ibeis_cnn.train --allexamples
        python -m ibeis_cnn.train --allexamples --noface --nosrc

    CommandLine:
        cd %CODE_DIR%/ibies_cnn/code
        cd $CODE_DIR/ibies_cnn/code
        code
        cd ibeis_cnn/code
        python train.py

    PythonPrereqs:
        pip install theano
        pip install git+https://github.com/Lasagne/Lasagne.git
        pip install git+git://github.com/lisa-lab/pylearn2.git
        #pip install lasagne
        #pip install pylearn2
        git clone git://github.com/lisa-lab/pylearn2.git
        git clone https://github.com/Lasagne/Lasagne.git
        cd pylearn2
        python setup.py develop
        cd ..
        cd Lesagne
        git checkout 8758ac1434175159e5c1f30123041799c2b6098a
        python setup.py develop
    """
    #train_pz()
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
