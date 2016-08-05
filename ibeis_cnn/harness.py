#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
constructs the Theano optimization and trains a learning model,
optionally by initializing the network with pre-trained weights.

http://cs231n.github.io/neural-networks-3/#distr
"""
from __future__ import absolute_import, division, print_function
from ibeis_cnn import utils
from ibeis_cnn import batch_processing as batch
#from ibeis_cnn import draw_net
from six.moves import input, zip, range
import numpy as np
import utool as ut
import time
import cv2
print, rrr, profile = ut.inject2(__name__, '[ibeis_cnn.harness]')


@profile
def train(model, X_train, y_train, X_valid, y_valid, dataset, config):
    r"""
    CommandLine:
        python -m ibeis_cnn.harness --test-train

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.harness import *  # NOQA
        >>> result = train(model, X_train, y_train, X_valid, y_valid, config)
        >>> print(result)
    """

    learning_rate_schedule = config.get('learning_rate_schedule', 15)
    max_epochs = config.get('max_epochs', None)
    test_freq  = config.get('test_freq', None)
    learning_rate_adjust  = config.get('learning_rate_adjust', .8)

    batchiter_kw = dict(
        #showprog=ut.get_argflag('--monitor'),
        showprog=True,
        time_thresh=4,
        time_thresh_growth=ut.PHI * 2,
    )

    batchtrain_kw = ut.merge_dicts(
        batchiter_kw,
        {
            'augment_on': True,
            #'augment_on': False,
            'randomize_batch_order': True,
            'buffered': True,
        }
    )

    batchtest_kw = ut.merge_dicts(
        batchiter_kw,
        {
            'augment_on': False,
            'randomize_batch_order': False,
        }
    )

    print('\n[train] --- TRAINING LOOP ---')
    # Center the data by subtracting the mean
    model.assert_valid_data(X_train)
    model.ensure_training_state(X_train, y_train)

    print('\n[train] --- MODEL INFO ---')
    model.print_architecture_str()
    model.print_layer_info()

    MONITOR_PROGRESS = ut.get_argflag('--monitor')
    if MONITOR_PROGRESS:
        # FIXME; put into better place
        progress_dir = ut.unixjoin(model.training_dpath, 'progress')
        ut.ensuredir(progress_dir)
        def progress_metric_path(x):
            return ut.get_nonconflicting_path(ut.unixjoin(progress_dir, x))
        def progress_metric_dir(x):
            return ut.ensuredir(progress_metric_path(x))
        history_progress_dir = progress_metric_dir('%s_%02d_history' % (model.arch_tag,))
        weights_progress_dir = progress_metric_dir('%s_%02d_weights' % (model.arch_tag,))
        history_text_fpath = progress_metric_path('%s_%02d_era_history' % (model.arch_tag,))
        ut.vd(progress_dir)

        def overwrite_latest_image(fpath, new_name):
            """
            copies the new image to a path to be overwritten so new updates are
            shown
            """
            from os.path import split, join, splitext, dirname
            import shutil
            dpath, fname = split(fpath)
            ext = splitext(fpath)[1]
            shutil.copy(fpath, join(dpath, 'latest ' + new_name + ext))
            shutil.copy(fpath, join(dirname(dpath), 'latest ' + new_name + ext))

        # Write initial states of the weights
        fpath = model.imwrite_weights(dpath=weights_progress_dir, fnum=2,
                                      verbose=0)
        overwrite_latest_image(fpath, 'weights')

    # Create the Theano primitives
    # create theano symbolic expressions that define the network
    theano_funcs = model.build()
    theano_backprop, theano_forward, theano_predict, updates = theano_funcs

    #show_times    = kwargs.get('print_timing', False)
    #show_features = kwargs.get('show_features', False)
    #test_freq     = kwargs.get('test_freq', None)

    epoch = model.best_results['epoch']
    #ut.embed()
    if epoch is None:
        epoch = 0
        print('Initializng training at epoch=%r' % (epoch,))
    else:
        print('Resuming training at epoch=%r' % (epoch,))
    #epoch_marker  = epoch

    # number of non-best iterations after, that triggers a best save
    save_freq = 10
    save_after_best_wait_epochs = 5
    save_after_best_countdown = None

    # Begin training the neural network
    print('\n[train] starting training at %s with learning rate %.9f' %
          (utils.get_current_time(), model.learning_rate))
    print('learning_state = %s' % ut.dict_str(model.learning_state))
    printcol_info = utils.get_printcolinfo(model.requested_headers)
    utils.print_header_columns(printcol_info)

    model.start_new_era(X_train, y_train, X_valid, y_valid, dataset.alias_key)

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
            train_outputs = batch.process_batch(
                model, X_train, y_train, theano_backprop, **batchtrain_kw)
            # compute the loss over all testing batches
            epoch_info['train_loss'] = train_outputs['loss'].mean()
            epoch_info['train_loss_regularized'] = (
                train_outputs['loss_regularized'].mean())
            #if 'valid_acc' in model.requested_headers:
            #    epoch_info['test_acc']  = train_outputs['accuracy']

            # If the training loss is nan, the training has diverged
            regularization_amount  = (
                train_outputs['loss_regularized'] - train_outputs['loss'])
            regularization_ratio   = (
                regularization_amount / train_outputs['loss'])
            regularization_percent = (
                regularization_amount / train_outputs['loss_regularized'])

            epoch_info['regularization_percent'] = regularization_percent
            epoch_info['regularization_ratio'] = regularization_ratio

            param_update_mags = {}
            for key, val in train_outputs.items():
                if key.startswith('param_update_magnitude_'):
                    key_ = key.replace('param_update_magnitude_', '')
                    param_update_mags[key_] = (val.mean(), val.std())
            epoch_info['param_update_mags'] = param_update_mags
            #if key.startswith('param_update_mag_')

            if np.isnan(epoch_info['train_loss']):
                print('\n[train] training diverged\n')
                break

            # Run validation set
            valid_outputs = batch.process_batch(
                model, X_valid, y_valid, theano_forward, **batchtest_kw)
            epoch_info['valid_loss'] = valid_outputs['loss_determ'].mean()
            epoch_info['valid_loss_std'] = valid_outputs['loss_determ'].std()
            if 'valid_acc' in model.requested_headers:
                # bit of a hack to bring accuracy back in
                #np.mean(valid_outputs['predictions'] ==
                #valid_outputs['auglbl_list'])
                epoch_info['valid_acc'] = valid_outputs['accuracy'].mean()

            # TODO
            request_determ_loss = False
            if request_determ_loss:
                train_determ_outputs = batch.process_batch(
                    model, X_train, y_train, theano_forward, **batchiter_kw)
                epoch_info['train_loss_determ'] = (
                    train_determ_outputs['loss_determ'].mean())

            # Calculate request_test before adding to the epoch counter
            request_test = utils.checkfreq(test_freq, epoch)
            if request_test:
                raise NotImplementedError('not done yet')
                test_outputs = batch.process_batch(
                    model, X_train, y_train, theano_forward, **batchiter_kw)
                test_loss = test_outputs['loss_determ'].mean()  # NOQA
                #if kwargs.get('show_confusion', False):
                #    draw_net.output_confusion_matrix(X_test, results_dpath,
                #    test_results, model=model, **kwargs)

            # ---------------------------------------
            # Summarize the epoch

            duration = tt.toc()
            epoch_info['duration'] = duration
            epoch_info['trainval_rat'] = (
                epoch_info['train_loss'] / epoch_info['valid_loss'])

            # ---------------------------------------
            # Record this epoch in history
            model.record_epoch(epoch_info)

            # ---------------------------------------
            # Check how we are learning
            best_found = (
                epoch_info['valid_loss'] < model.best_results['valid_loss'])
            if best_found:
                model.best_weights = model.get_all_param_values()
                model.best_results['epoch'] = epoch_info['epoch']
                for key in model.requested_headers:
                    model.best_results[key] = epoch_info[key]
                save_after_best_countdown = save_after_best_wait_epochs
                #epoch_marker = epoch

            # Check frequencies and countdowns
            output_diagnostics = utils.checkfreq(save_freq, epoch)
            if save_after_best_countdown is not None:
                if save_after_best_countdown == 0:
                    ## Callbacks on best found
                    save_after_best_countdown = None
                    output_diagnostics = True
                else:
                    save_after_best_countdown -= 1

            if duration < 60:
                # dont show prog on short iterations
                #batchiter_kw['showprog'] = False
                batchiter_kw['showprog'] = True

            # ---------------------------------------
            # Output Diagnostics

            # Print the epoch
            utils.print_epoch_info(model, printcol_info, epoch_info)

            # Output any diagnostics
            if output_diagnostics:
                model.checkpoint_save_model_info()
                model.save_model_info()
                model.checkpoint_save_model_state()
                model.save_model_state()

            if MONITOR_PROGRESS:
                fpath = model.imwrite_era_history(dpath=history_progress_dir,
                                                  fnum=1, verbose=0)
                overwrite_latest_image(fpath, 'history')
                fpath = model.imwrite_weights(dpath=weights_progress_dir,
                                              fnum=2, verbose=0)
                overwrite_latest_image(fpath, 'weights')
                history_text = ut.list_str(model.era_history, newlines=True)
                ut.write_to(history_text_fpath, history_text, verbose=False)

            # Learning rate schedule update
            if epoch % learning_rate_schedule == (learning_rate_schedule - 1):
                #epoch_marker = epoch
                model.learning_rate = (
                    model.learning_rate * learning_rate_adjust)
                model.start_new_era(
                    X_train, y_train, X_valid, y_valid, dataset.alias_key)
                utils.print_header_columns(printcol_info)

            # Break on max epochs
            if max_epochs is not None and epoch >= max_epochs:
                print('\n[train] maximum number of epochs reached\n')
                break

            # Increment the epoch
            epoch += 1

        except KeyboardInterrupt:
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
                print('[train]     7 - Clean dataset [WARNING: EXPERIMENTAL]')
                print('[train]  ELSE - Stop network training')
                resolution = input('[train] Resolution: ')
            resolution = int(resolution)
            # We have a resolution
            if resolution == 0:
                print('resuming training...')
            elif resolution == 1:
                # Shock the weights of the network
                utils.shock_network(model.output_layer)
                model.learning_rate = model.learning_rate * 2
                #epoch_marker = epoch
                utils.print_header_columns(printcol_info)
            elif resolution == 2:
                # Save the weights of the network
                model.checkpoint_save_model_info()
                model.save_model_info()
                model.checkpoint_save_model_state()
                model.save_model_state()
            elif resolution == 3:
                ut.view_directory(model.training_dpath)
            elif resolution == 4:
                ut.embed()
            elif resolution == 5:
                output_fpath_list = model.imwrite_weights(index=0)
                for fpath in output_fpath_list:
                    ut.startfile(fpath)
            elif resolution == 6:
                print(model.get_state_str())
            elif resolution == 7:
                y_train = _clean(model, theano_forward, X_train, y_train,
                                 **batchiter_kw)
                y_valid = _clean(model, theano_forward, X_valid, y_valid,
                                 **batchiter_kw)
            else:
                # Terminate the network training
                raise
    # Save the best network
    model.checkpoint_save_model_state()
    model.save_model_state()


def _clean(model, theano_forward, X_list, y_list, min_conf=0.95,
           **batchiter_kw):
    import random
    # Perform testing
    clean_outputs = batch.process_batch(
        model, X_list, y_list, theano_forward, augment_on=False,
        randomize_batch_order=False, **batchiter_kw)
    prediction_list = clean_outputs['labeled_predictions']
    confidence_list = clean_outputs['confidences']
    enumerated = enumerate(zip(y_list, prediction_list, confidence_list))

    switched_counter = 0
    switched = {}
    for index, (y, prediction, confidence) in enumerated:
        if confidence < min_conf:
            continue
        if y == prediction:
            continue
        if random.uniform(0.0, 1.0) > confidence:
            continue
        # Perform the switching
        y_list[index] = prediction
        switched_counter += 1
        # Keep track of changes
        y = str(y)
        prediction = str(prediction)
        if y not in switched:
            switched[y] = {}
        if prediction not in switched[y]:
            switched[y][prediction] = 0
        switched[y][prediction] += 1

    total = len(y_list)
    ratio = switched_counter / total
    args = (switched_counter, total, ratio, )
    print('[_clean] Cleaned Data... [ %d / %d ] ( %0.04f )' % args)
    for src in sorted(switched.keys()):
        for dst in sorted(switched[src].keys()):
            print('[_clean] \t%r -> %r : %d' % (src, dst, switched[src][dst], ))

    return y_list


def test_data2(model, X_test, y_test):
    """
    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.harness import *  # NOQA

    Ignore:
        # vars for process_batch
        X = X_test = data
        y = y_test = labels
        y = None
        theano_fn = theano_predict
        fix_output=True
        kwargs = batchiter_kw

        f = list(batch_iter)
        Xb, yb = f[0]
    """

    if ut.VERBOSE:
        print('\n[train] --- MODEL INFO ---')
        model.print_architecture_str()
        model.print_layer_info()

    # Create the Theano primitives
    # create theano symbolic expressions that define the network
    print('\n[train] --- COMPILING SYMBOLIC THEANO FUNCTIONS ---')
    theano_funcs = model._build_theano_funcs(request_predict=True,
                                             request_forward=False,
                                             request_backprop=False)
    theano_backprop, theano_forward, theano_predict, updates = theano_funcs

    # Begin testing with the neural network
    print('\n[test] starting testing with batch size %0.1f' % (
        model.batch_size))

    batchiter_kw = dict(
        showprog=True,
        time_thresh=10,
    )

    #X_test = X_test[0:259]
    # Start timer
    test_outputs = batch.process_batch(model, X_test, None, theano_predict,
                                       fix_output=True, **batchiter_kw)
    return test_outputs


def test_convolutional(model, theano_predict, image, patch_size='auto',
                       stride='auto', padding=32, batch_size=None,
                       verbose=False, **kwargs):
    """ Using a network, test an entire image full convolutionally

    This function will test an entire image full convolutionally (or a close
    approximation of full convolutionally).  The CUDA framework and driver is a
    limiting factor for how large an image can be given to a network for full
    convolutional inference.  As a result, we implement a non-overlapping (or
    little overlapping) patch extraction approximation that processes the entire
    image within a single batch or very few batches.  This is an extremely
    efficient process for processing an image with a CNN.

    The patches are given a slight overlap in order to smooth the effects of
    boundary conditions, which are seen on every patch.  We also mirror the
    border of each patch and add an additional amount of padding to cater to the
    architecture's receptive field reduction.

    See :func:`utils.extract_patches_stride` for patch extraction behavior.

    Args:
        model (Model): the network to use to perform feedforward inference
        image (numpy.ndarray): the image passed in to make a coreresponding
            sized dictionarf of response maps
        patch_size (int, tuple of int, optional): the size of the patches
            extracted across the image, passed in as a 2-tuple of (width,
            height).  Defaults to (200, 200).
        stride (int, tuple of int, optional): the stride of the patches
            extracted across the image.  Defaults to [patch_size - padding].
        padding (int, optional): the mirrored padding added to every patch
            during testing, which can be used to offset the effects of the
            receptive field reduction in the network.  Defaults to 32.
        **kwargs: arbitrary keyword arguments, passed to
            :func:`model.test()`

    Returns:
        samples, canvas_dict (tuple of int and dict): the number of total
            samples used to generate the response map and the actual response
            maps themselves as a dictionary.  The dictionary uses the class
            labels as the strings and the numpy array image as the values.
    """

    def _add_pad(data_):
        if len(data_.shape) == 2:
            data_padded = np.pad(data_, padding, 'reflect', reflect_type='even')
        else:
            h, w, c = data_.shape
            data_padded = np.dstack([
                np.pad(data_[:, :, _], padding, 'reflect', reflect_type='even')
                for _ in range(c)
            ])
        return data_padded

    def _resize_target(image, target_height=None, target_width=None):
        assert target_height is not None or target_width is not None
        height, width = image.shape[:2]
        if target_height is not None and target_width is not None:
            h = target_height
            w = target_width
        elif target_height is not None:
            h = target_height
            w = (width / height) * h
        elif target_width is not None:
            w = target_width
            h = (height / width) * w
        w, h = int(w), int(h)
        return cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)

    if verbose:
        # Start timer
        t0 = time.time()
        print('[harness] Loading the testing data (convolutional)...')
    # Try to get the image's shape
    h, w = image.shape[:2]

    original_shape = None
    if h < w and h < 256:
        original_shape = image.shape
        image = _resize_target(image, target_height=256)
    if w < h and w < 256:
        original_shape = image.shape
        image = _resize_target(image, target_width=256)

    h, w = image.shape[:2]

    #GLOBAL_LIMIT = min(256, w, h)
    # HACK, this only works for square data shapes
    GLOBAL_LIMIT = model.data_shape[0]
    # Inference
    if patch_size == 'auto':
        patch_size = (GLOBAL_LIMIT - 2 * padding, GLOBAL_LIMIT - 2 * padding)
    if stride == 'auto':
        psx, psy = patch_size
        stride = (psx - padding, psy - padding)
    _tup = utils.extract_patches_stride(image, patch_size, stride)
    data_list, coord_list = _tup
    samples = len(data_list)
    if batch_size is None:
        batch_size = samples
    start = 0
    label_list = []
    confidence_list = []
    while start < samples:
        end = min(samples, start + batch_size)
        data_list_segment = data_list[start: end]
        # coord_list_segment = coord_list[start: end]

        # Augment the data_list by adding a reflected pad
        data_list_ = np.array([
            _add_pad(data_)
            for data_ in data_list_segment
        ])

        batchiter_kw = dict(
            fix_output=False,
            #showprog=False,
            showprog=True,
            time_thresh=10,
            spatial=True,
        )

        test_results = batch.process_batch(model, data_list_, None,
                                           theano_predict, **batchiter_kw)

        label_list.extend(test_results['labeled_predictions'])
        confidence_list.extend(test_results['confidences'])
        start += batch_size

    # Get all of the labels for the data, inheritted from the model
    label_list_ = list(model.encoder.classes_)
    # Create a dictionary of canvases
    canvas_dict = {}
    for label in label_list_:
        canvas_dict[label] = np.zeros((h, w))  # We want float precision
    # Construct the canvases using the forward inference results
    label_list_ = label_list_[::-1]
    # print('[harness] Labels: %r' %(label_list_, ))
    zipped = list(zip(data_list, coord_list, label_list, confidence_list))
    for label in label_list_:
        for data, coord, label_, confidence in zipped:
            x1, y1, x2, y2 = coord
            # Get label and apply to confidence
            confidence_ = np.copy(confidence)
            confidence_[label_ != label] = 0
            confidence_ *= 255.0
            # Blow up canvas
            mask = cv2.resize(confidence_, data.shape[0:2])
            # Get the current values
            current = canvas_dict[label][y1:y2, x1:x2]
            # Where the current canvas is zero (most of it), make it mask
            flags = current == 0
            current[flags] = mask[flags]
            # Average the current with the mask, which address overlapping areas
            mask = 0.5 * mask + 0.5 * current
            # Aggregate
            canvas_dict[label][y1:y2, x1:x2] = mask
        # Blur
        # FIXME: Should this postprocessing step applied here?
        # There is postprocessing in ibeis/algos/preproc/preproc_probchip.py
        ksize = 3
        kernel = (ksize, ksize)
        canvas_dict[label] = cv2.blur(canvas_dict[label], kernel)
    # Cast all images to uint8
    for label in label_list_:
        canvas = np.around(canvas_dict[label])
        canvas = canvas.astype(np.uint8)
        if original_shape is not None:
            canvas = _resize_target(
                canvas,
                target_height=original_shape[0],
                target_width=original_shape[1]
            )
        canvas_dict[label] = canvas
    if verbose:
        # End timer
        t1 = time.time()
        duration = t1 - t0
        print('[harness] Interface took %s seconds...' % (duration, ))
    # Return the canvas dict
    return samples, canvas_dict
