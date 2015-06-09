#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
constructs the Theano optimization and trains a learning model,
optionally by initializing the network with pre-trained weights.
"""
from __future__ import absolute_import, division, print_function
from ibeis_cnn import utils
from ibeis_cnn import batch_processing as batch
from six.moves import input
import numpy as np
import utool as ut
print, rrr, profile = ut.inject2(__name__, '[ibeis_cnn.harness]')


def sample_train_valid_test(model, data, labels):
    # TODO: make this less memory intensive
    print('[train] creating train, validation datasaets...')
    memtrack = ut.MemoryTracker(disable=True)
    memtrack.report('sample_data0')
    data_per_label = getattr(model, 'data_per_label', 1)
    train_split = .2
    #train_split = .4
    #train_split = .5
    _tup = utils.train_test_split(data, labels, eval_size=train_split, data_per_label=data_per_label)
    X_train, y_train, X_valid, y_valid = _tup
    memtrack.report('sample_data1')
    _tup = utils.train_test_split(X_train, y_train, eval_size=0.1, data_per_label=data_per_label)
    X_train, y_train, X_test, y_test = _tup
    memtrack.report('sample_data2')
    #memtrack.report_obj(X_train, 'X_train')
    #memtrack.report_obj(X_valid, 'X_valid')
    #memtrack.report_obj(X_test, 'X_test')
    #memtrack.report_obj(data, 'data')
    print('len(X_train) = %r' % (len(X_train),))
    print('len(X_valid) = %r' % (len(X_valid),))
    print('len(X_test)  = %r' % (len(X_test),))
    return X_train, y_train, X_valid, y_valid, X_test, y_test


# ---------------

@profile
def train(model, X_train, y_train, X_valid, y_valid, trainset, config):
    r"""
    CommandLine:
        python -m ibeis_cnn.harness --test-train

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.harness import *  # NOQA
        >>> result = train(model, X_train, y_train, X_valid, y_valid, config)
        >>> print(result)
    """
    print('\n[train] --- TRAINING LOOP ---')
    # Center the data by subtracting the mean
    model.ensure_training_state(X_train, y_train)

    print('\n[train] --- MODEL INFO ---')
    model.print_architecture_str()
    model.print_layer_info()

    # Create the Theano primitives
    # create theano symbolic expressions that define the network
    print('\n[train] --- COMPILING SYMBOLIC THEANO FUNCTIONS ---')
    print('[model] creating Theano primitives...')
    theano_funcs = model.build_theano_funcs()
    theano_backprop, theano_forward, theano_predict = theano_funcs

    patience = config.get('patience')
    max_epochs = config.get('max_epochs', None)
    test_freq = config.get('test_freq', None)
    #show_times    = kwargs.get('print_timing', False)
    #show_features = kwargs.get('show_features', False)
    #test_freq      = kwargs.get('test_freq', None)

    epoch = model.best_results['epoch']
    #ut.embed()
    if epoch is None:
        epoch = 0
        print('Initializng training at epoch=%r' % (epoch,))
    else:
        print('Resuming training at epoch=%r' % (epoch,))
    epoch_marker  = epoch

    # number of non-best iterations after, that triggers a best save
    save_freq = 10
    save_after_best_wait_epochs = 5
    save_after_best_countdown = None

    # Begin training the neural network
    print('\n[train] starting training at %s with learning rate %.9f' %
          (utils.get_current_time(), model.learning_rate))
    printcol_info = utils.get_printcolinfo(model.requested_headers)
    utils.print_header_columns(printcol_info)

    model.start_new_era(X_train, y_train, trainset.alias_key)

    batchiter_kw = dict(
        #showprog=False,
        showprog=True,
        time_thresh=4,
        time_thresh_growth=ut.PHI * 2,
    )

    tt = ut.Timer(verbose=False)
    while True:
        try:
            # Begin epoch

            epoch_info = {
                'epoch': epoch,
            }

            tt.tic()

            # ---------------------------------------
            # Run training set
            """
            X_train = X_train[0:128 * 10]
            y_train = y_train[0:128 * 5]
            X = X_train
            y = y_train
            theano_fn = theano_backprop
            """
            train_outputs = batch.process_batch(
                model, X_train, y_train, theano_backprop, augment_on=True,
                rand=True, **batchiter_kw)
            # compute the loss over all testing batches
            epoch_info['train_loss'] = train_outputs['loss_regularized'].mean()
            #if 'valid_acc' in model.requested_headers:
            #    epoch_info['test_acc']  = train_outputs['accuracy']

            # If the training loss is nan, the training has diverged
            if np.isnan(epoch_info['train_loss']):
                print('\n[train] training diverged\n')
                break

            # Run validation set
            valid_outputs = batch.process_batch(
                model, X_valid, y_valid, theano_forward, augment_on=False,
                rand=False, **batchiter_kw)
            epoch_info['valid_loss'] = valid_outputs['loss_determ'].mean()
            if 'valid_acc' in model.requested_headers:
                # bit of a hack to bring accuracy back in
                #np.mean(valid_outputs['predictions'] == valid_outputs['auglbl_list'])
                epoch_info['valid_acc'] = valid_outputs['accuracy'].mean()

            # Calculate request_test before adding to the epoch counter
            request_test = utils.checkfreq(test_freq, epoch)
            if request_test:
                raise NotImplementedError('not done yet')
                test_outputs = batch.process_batch(
                    model, X_train, y_train, theano_forward, augment_on=False,
                    rand=False, **batchiter_kw)
                test_loss = test_outputs['loss_determ'].mean()  # NOQA
                #if kwargs.get('show_confusion', False):
                #    #output_confusion_matrix(results_path, **kwargs)
                #    batch.output_confusion_matrix(X_test, results_dpath, test_results, model=model, **kwargs)

            # ---------------------------------------
            # Summarize the epoch

            duration = tt.toc()
            epoch_info['duration'] = duration
            epoch_info['trainval_rat'] = epoch_info['train_loss'] / epoch_info['valid_loss']

            # ---------------------------------------
            # Record this epoch in history
            model.record_epoch(epoch_info)

            # ---------------------------------------
            # Check how we are learning
            best_found = epoch_info['valid_loss'] < model.best_results['valid_loss']
            if best_found:
                model.best_weights = model.get_all_param_values()
                model.best_results['epoch'] = epoch_info['epoch']
                for key in model.requested_headers:
                    model.best_results[key] = epoch_info[key]
                save_after_best_countdown = save_after_best_wait_epochs
                epoch_marker = epoch

            # Print the epoch
            utils.print_epoch_info(model, printcol_info, epoch_info)

            # Check frequencies and countdowns
            output_diagnostics = utils.checkfreq(save_freq, epoch)
            if save_after_best_countdown is not None:
                if save_after_best_countdown == 0:
                    ## Callbacks on best found
                    save_after_best_countdown = None
                    output_diagnostics = True
                else:
                    save_after_best_countdown -= 1

            # Output any diagnostics
            if output_diagnostics:
                model.checkpoint_save_model_state()
                model.save_model_state()
                #model.draw_convolutional_layers(epoch=epoch)
                model.draw_convolutional_layers()

            # Learning rate schedule update
            if epoch >= epoch_marker + patience:
                epoch_marker = epoch
                model.learning_rate = model.learning_rate_update(model.learning_rate)
                utils.print_header_columns(printcol_info)
                model.start_new_era(X_train, y_train, trainset.alias_key)

            # Break on max epochs
            if max_epochs is not None and epoch >= max_epochs:
                print('\n[train] maximum number of epochs reached\n')
                break

            # Increment the epoch
            epoch += 1

        except KeyboardInterrupt:
            # We have caught the Keyboard Interrupt, figure out what resolution mode
            print('\n[train] Caught CRTL+C')
            resolution = ''
            while not (resolution.isdigit()):
                print('\n[train] What do you want to do?')
                print('[train]     0 - Continue')
                print('[train]     1 - Shock weights')
                print('[train]     2 - Save best weights')
                print('[train]     3 - View training directory')
                print('[train]     4 - Embed into IPython')
                print('[train]     5 - Draw current weights')
                print('[train]     6 - Show training state')
                print('[train]  ELSE - Stop network training')
                resolution = input('[train] Resolution: ')
            resolution = int(resolution)
            # We have a resolution
            if resolution == 0:
                print('resuming training...')
            elif resolution == 1:
                # Shock the weights of the network
                utils.shock_network(model.output_layer)
                epoch_marker = epoch
                model.learning_rate = model.learning_rate_shock(model.learning_rate)
                utils.print_header_columns(printcol_info)
            elif resolution == 2:
                # Save the weights of the network
                model.checkpoint_save_model_state()
                model.save_model_state()
            elif resolution == 3:
                ut.view_directory(model.training_dpath)
            elif resolution == 4:
                ut.embed()
            elif resolution == 5:
                output_fpath_list = model.draw_convolutional_layers()
                for fpath in output_fpath_list:
                    ut.startfile(fpath)
            elif resolution == 6:
                print(model.get_state_str())
            else:
                # Terminate the network training
                raise
    # Save the best network
    model.checkpoint_save_model_state()
    model.save_model_state()


def test_data2(model, X_test, y_test):
    """
    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.test import *  # NOQA
    """

    print('\n[train] --- MODEL INFO ---')
    model.print_architecture_str()
    model.print_layer_info()

    # Create the Theano primitives
    # create theano symbolic expressions that define the network
    print('\n[train] --- COMPILING SYMBOLIC THEANO FUNCTIONS ---')
    print('[model] creating Theano primitives...')
    theano_funcs = model.build_theano_funcs(request_predict=True, request_forward=False, request_backprop=False)
    theano_backprop, theano_forward, theano_predict = theano_funcs

    # Begin testing with the neural network
    print('\n[test] starting testing with batch size %0.1f' % (model.batch_size))

    batchiter_kw = dict(
        showprog=True,
        time_thresh=10,
    )

    #X_test = X_test[0:259]
    # Start timer
    test_outputs = batch.process_batch(model, X_test, None, theano_predict, fix_output=True, **batchiter_kw)
    return test_outputs
