#!/usr/bin/env python

# train.py
# constructs the Theano optimization and trains a learning model,
# optionally by initializing the network with pre-trained weights.

# our own imports
import utils
import model

# module imports
import time
import theano
import numpy as np
import cPickle as pickle

from lasagne import layers

from os.path import join, abspath
import random


def train(data_file, labels_file, weights_file, pretrained_weights_file=None):

    def augmentation(Xb, yb):
        # label_map = {
        #     0:  4,
        #     1:  5,
        #     2:  6,
        #     3:  7,
        #     8:  12,
        #     9:  13,
        #     10: 14,
        #     11: 15,
        # }
        label_map = { x: x + 4 for x in range(0, 4) + range(8, 12) }
        # Apply inverse
        for key in label_map.keys():
            label = label_map[key]
            label_map[label] = key
        # Map
        points, channels, height, width = Xb.shape
        for index in range(points):
            if random.uniform(0.0, 1.0) <= 0.5:
                Xb[index] = Xb[index, :, ::-1]
                yb[index] = label_map[yb[index]]
        return Xb, yb

    def learning_rate_update(x):
        return x / 10.0

    def learning_rate_shock(x):
        return min(learning_rate, x * 10.0)

    # Training parameters
    center        = True
    learning_rate = 0.03
    momentum      = 0.9
    batch_size    = 128
    patience      = 15
    max_epochs    = patience * 10
    output_dims   = 16    # the number of outputs from the softmax layer (# classes)

    ######################################################################################

    # Load the data
    print('\n[data] loading data...')
    data, labels = utils.load(data_file, labels_file)
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

    # Center the data by subtracting the mean
    if center:
        print('[data] applying data centering...')
        center_mean = np.mean(X_train, axis=0)
        # center_std  = np.std(X_train, axis=0)
        center_std  = 255.0
    else:
        center_mean = 0.0
        center_std  = 255.0

    # Build and print the model
    print('\n[model] building model...')
    input_cases, input_channels, input_height, input_width = data.shape
    output_layer = model.build_model(batch_size, input_width, input_height,
                                     input_channels, output_dims)
    utils.print_layer_info(output_layer)

    # Load the pretrained model if specified
    if pretrained_weights_file is not None:
        print('[model] loading pretrained weights from %s' % (pretrained_weights_file))
        with open(pretrained_weights_file, 'rb') as pfile:
            pretrained_weights = pickle.load(pfile)
            layers.set_all_param_values(output_layer, pretrained_weights)

    # Create the Theano primitives
    print('[model] creating Theano primitives...')
    learning_rate_ = theano.shared(utils.float32(learning_rate))
    all_iters = utils.create_iter_funcs(learning_rate_, momentum, output_layer)
    train_iter, valid_iter, predict_iter = all_iters

    # Begin training the neural network
    print('\n[train] starting training at %s with learning rate %.9f' % (utils.get_current_time(), learning_rate, ))
    utils.print_header_columns()
    epoch, best_weights, best_accuracy, best_epoch = 0, None, 0.0, 0
    best_train_loss, best_valid_loss, best_valid_accuracy = np.inf, np.inf, 0.0
    try:
        while True:
            try:
                # Reset the loses for the batch
                train_losses, valid_losses, valid_accuracies = [], [], []

                t0 = time.time()
                # compute the loss over all training batches
                for Xb, yb in utils.batch_iterator(X_train, y_train, batch_size,
                                                   center_mean, center_std,
                                                   rand=True, augment=augmentation):
                    batch_train_loss = train_iter(Xb, yb)
                    train_losses.append(batch_train_loss)

                # compute the loss over all validation batches
                for Xb, yb in utils.batch_iterator(X_valid, y_valid, batch_size,
                                                   center_mean, center_std):
                    batch_valid_loss, batch_accuracy = valid_iter(Xb, yb)
                    valid_losses.append(batch_valid_loss)
                    valid_accuracies.append(batch_accuracy)

                # estimate the loss over all batches
                avg_train_loss = np.mean(train_losses)
                avg_valid_loss = np.mean(valid_losses)
                avg_valid_accuracy = np.mean(valid_accuracies)

                # If the training loss is nan, the training has diverged
                if np.isnan(avg_train_loss):
                    print('\n[train] training diverged\n')
                    break

                # Running tab for what the best model
                if avg_train_loss < best_train_loss:
                    best_train_loss = avg_train_loss
                if avg_valid_loss < best_valid_loss:
                    best_valid_loss = avg_valid_loss
                    best_epoch = epoch
                    best_weights = layers.get_all_param_values(output_layer)
                    best_accuracy = avg_valid_accuracy
                if avg_valid_accuracy > best_valid_accuracy:
                    best_valid_accuracy = avg_valid_accuracy

                # Learning rate schedule update
                if epoch >= best_epoch + patience:
                    best_epoch = epoch
                    utils.set_learning_rate(learning_rate_, learning_rate_update)

                # Increment the epoch
                epoch += 1
                utils.print_epoch_info(avg_valid_loss, best_valid_loss, avg_valid_accuracy,
                                       best_valid_accuracy, avg_train_loss, best_train_loss,
                                       epoch, time.time() - t0)

                # Break on max epochs
                if epoch >= max_epochs:
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
                    best_epoch = epoch
                    utils.set_learning_rate(learning_rate_, learning_rate_shock)
                elif resolution == 2:
                    # Save the weights of the network
                    utils.save_best_model(best_weights, best_accuracy, weights_file)
                else:
                    # Terminate the network training
                    raise KeyboardInterrupt
    except KeyboardInterrupt:
        print('\n\n[train] ...stopping network training\n')

    # Save the best network
    utils.save_best_model(best_weights, best_accuracy, weights_file)


if __name__ == '__main__':
    project_name            = 'viewpoint'
    root                    = abspath(join('..', 'data'))
    train_data_file         = join(root, 'numpy', project_name, 'X.npy')
    train_labels_file       = join(root, 'numpy', project_name, 'y.npy')
    weights_file            = join(root, 'nets', 'ibeis_cnn_weights.pickle')
    pretrained_weights_file = join(root, 'nets', 'pretrained_weights.pickle')

    train(train_data_file, train_labels_file, weights_file)
    #train(train_data_file, train_labels_file, weights_file, pretrained_weights_file)
