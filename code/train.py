#!/usr/bin/env python

# train.py
# constructs the Theano optimization and trains a learning model,
# optionally by initializing the network with pre-trained weights.

# our own imports
import utils
import models

# module imports
import time
import theano
import numpy as np
import cPickle as pickle

from lasagne import layers

from os.path import join, abspath


def train(data_file, labels_file, model, weights_file, pretrained_weights_file=None,
          pretrained_kwargs=False, **kwargs):

    # Training parameters
    utils._update(kwargs, 'center',         True)
    utils._update(kwargs, 'learning_rate',  0.03)
    utils._update(kwargs, 'momentum',       0.9)
    utils._update(kwargs, 'batch_size',     128)
    utils._update(kwargs, 'patience',       10)
    utils._update(kwargs, 'test',           5)  # Test every X epochs
    utils._update(kwargs, 'max_epochs',     kwargs.get('patience') * 10)
    # utils._update(kwargs, 'regularization', None)
    utils._update(kwargs, 'regularization', 0.001)
    utils._update(kwargs, 'output_dims',    None)

    ######################################################################################

    # Load the data
    print('\n[data] loading data...')
    data, labels = utils.load(data_file, labels_file)

    # Automatically figure out how many classes
    if kwargs.get('output_dims') is None:
        kwargs['output_dims'] = len(list(np.unique(labels)))

    # print('[load] adding channels...')
    # data = utils.add_channels(data)
    print('[data]     X.shape = %r' % (data.shape,))
    print('[data]     X.dtype = %r' % (data.dtype,))
    print('[data]     y.shape = %r' % (labels.shape,))
    print('[data]     y.dtype = %r' % (labels.dtype,))

    # utils.show_image_from_data(data)

    # Split the dataset into training and validation
    print('[data] creating train, validation datasaets...')
    dataset = utils.train_test_split(data, labels, eval_size=0.2)
    X_train, y_train, X_valid, y_valid = dataset
    dataset = utils.train_test_split(X_train, y_train, eval_size=0.1)
    X_train, y_train, X_test, y_test = dataset

    # Build and print the model
    print('\n[model] building model...')
    input_cases, input_channels, input_height, input_width = data.shape
    output_layer = model.build_model(kwargs.get('batch_size'), input_width, input_height,
                                     input_channels, kwargs.get('output_dims'))
    utils.print_layer_info(output_layer)

    # Create the Theano primitives
    print('[model] creating Theano primitives...')
    learning_rate_theano = theano.shared(utils.float32(kwargs.get('learning_rate')))
    all_iters = utils.create_iter_funcs(learning_rate_theano, output_layer, **kwargs)
    train_iter, valid_iter, test_iter = all_iters

    # Load the pretrained model if specified
    if pretrained_weights_file is not None:
        print('[model] loading pretrained weights from %s' % (pretrained_weights_file))
        with open(pretrained_weights_file, 'rb') as pfile:
            kwargs_ = pickle.load(pfile)
            pretrained_weights = kwargs_.get('best_weights', None)
            layers.set_all_param_values(output_layer, pretrained_weights)
            if pretrained_kwargs:
                kwargs = kwargs_

    # Center the data by subtracting the mean (AFTER KWARGS UPDATE)
    if kwargs.get('center'):
        print('[data] applying data centering...')
        utils._update(kwargs, 'center_mean', np.mean(X_train, axis=0))
        # utils._update(kwargs, 'center_std', np.std(X_train, axis=0))
        utils._update(kwargs, 'center_std', 255.0)
    else:
        utils._update(kwargs, 'center_mean', 0.0)
        utils._update(kwargs, 'center_std', 255.0)

    # Begin training the neural network
    vals = (utils.get_current_time(), kwargs.get('learning_rate'), )
    print('\n[train] starting training at %s with learning rate %.9f' % vals)
    utils.print_header_columns()

    utils._update(kwargs, 'best_weights',        None)
    utils._update(kwargs, 'best_valid_accuracy', 0.0)
    utils._update(kwargs, 'best_test_accuracy',  0.0)
    utils._update(kwargs, 'best_train_loss',     np.inf)
    utils._update(kwargs, 'best_valid_loss',     np.inf)
    utils._update(kwargs, 'best_valid_accuracy', 0.0)
    utils._update(kwargs, 'best_test_accuracy',  0.0)
    try:
        epoch = 0
        epoch_marker = epoch
        while True:
            try:
                # Start timer
                t0 = time.time()

                # compute the loss over all training and validation batches
                avg_train_loss = utils.forward_train(X_train, y_train, train_iter, rand=True,
                                                     augment=model.augment, **kwargs)
                avg_valid_data = utils.forward_valid(X_valid, y_valid, valid_iter,
                                                     augment=model.augment, **kwargs)
                avg_valid_loss, avg_valid_accuracy = avg_valid_data

                # If the training loss is nan, the training has diverged
                if np.isnan(avg_train_loss):
                    print('\n[train] training diverged\n')
                    break

                # Is this model the best we've ever seen?
                best_found = avg_valid_loss < kwargs.get('best_valid_loss')
                if best_found:
                    kwargs['best_epoch'] = epoch
                    epoch_marker = epoch
                    kwargs['best_weights'] = layers.get_all_param_values(output_layer)

                # compute the loss over all testing batches
                request_test = kwargs.get('test') is not None and epoch % kwargs.get('test') == 0
                if best_found or request_test:
                    avg_test_accuracy = utils.forward_test(X_test, y_test, test_iter, **kwargs)
                else:
                    avg_test_accuracy = None

                # Running tab for what the best model
                if avg_train_loss < kwargs.get('best_train_loss'):
                    kwargs['best_train_loss'] = avg_train_loss
                if avg_valid_loss < kwargs.get('best_valid_loss'):
                    kwargs['best_valid_loss'] = avg_valid_loss
                if avg_valid_accuracy > kwargs.get('best_valid_accuracy'):
                    kwargs['best_valid_accuracy'] = avg_valid_accuracy
                if avg_test_accuracy > kwargs.get('best_test_accuracy'):
                    kwargs['best_test_accuracy'] = avg_test_accuracy

                # Learning rate schedule update
                if epoch >= epoch_marker + kwargs.get('patience'):
                    epoch_marker = epoch
                    utils.set_learning_rate(learning_rate_theano, model.learning_rate_update)

                # End timer
                t1 = time.time()

                # Increment the epoch
                epoch += 1
                utils.print_epoch_info(avg_train_loss, avg_valid_loss,
                                       avg_valid_accuracy, avg_test_accuracy,
                                       epoch, t1 - t0, **kwargs)

                # Break on max epochs
                if epoch >= kwargs.get('max_epochs'):
                    print('\n[train] maximum number of epochs reached\n')
                    break
            except KeyboardInterrupt:
                # We have caught the Keyboard Interrupt, figure out what resolution mode
                print('\n[train] Caught CRTL+C')
                resolution = ''
                while not (resolution.isdigit() and int(resolution) in [1, 2, 3]):
                    print('\n[train] What do you want to do?')
                    print('[train]     1 - Shock weights')
                    print('[train]     2 - Save best weights')
                    print('[train]     3 - Stop network training')
                    resolution = raw_input('[train] Resolution: ')
                resolution = int(resolution)
                # We have a resolution
                if resolution == 1:
                    # Shock the weights of the network
                    utils.shock_network(output_layer)
                    epoch_marker = epoch
                    utils.set_learning_rate(learning_rate_theano, model.learning_rate_shock)
                elif resolution == 2:
                    # Save the weights of the network
                    utils.save_model(kwargs, weights_file)
                else:
                    # Terminate the network training
                    raise KeyboardInterrupt
    except KeyboardInterrupt:
        print('\n\n[train] ...stopping network training\n')

    # Save the best network
    utils.save_model(kwargs, weights_file)


if __name__ == '__main__':
    # project_name            = 'viewpoint'
    # model                   = models.PZ_GIRM_Model
    project_name            = 'plains'
    model                   = models.PZ_Model()
    config                  = {}

    root                    = abspath(join('..', 'data'))
    train_data_file         = join(root, 'numpy', project_name, 'X.npy')
    train_labels_file       = join(root, 'numpy', project_name, 'y.npy')
    weights_file            = join(root, 'nets', 'ibeis_cnn_weights.pickle')
    pretrained_weights_file = join(root, 'nets', 'pretrained_weights.pickle')

    train(train_data_file, train_labels_file, model, weights_file, **config)
    #train(train_data_file, train_labels_file, weights_file, pretrained_weights_file)
