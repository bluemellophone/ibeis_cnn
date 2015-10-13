# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from ibeis_cnn import utils
from ibeis_cnn import draw_net
#from lasagne import layers
#from lasagne import objectives
import lasagne
import numpy as np
import six
import theano
import theano.tensor as T
import utool as ut
#import warnings
print, rrr, profile = ut.inject2(__name__, '[ibeis_cnn.batch_processing]')


VERBOSE_BATCH = ut.get_argflag(('--verbose-batch', '--verbbatch')) or utils.VERBOSE_CNN
VERYVERBOSE_BATCH = ut.get_argflag(('--veryverbose-batch', '--veryverbbatch')) or ut.VERYVERBOSE


def process_batch(model, X, y, theano_fn, fix_output=False, buffered=False,
                  show=False, spatial=False, **kwargs):
    """
    compute the loss over all training batches

    CommandLine:
        python -m ibeis_cnn.batch_processing --test-process_batch --verbose
        python -m ibeis_cnn.batch_processing --test-process_batch:0 --verbose
        python -m ibeis_cnn.batch_processing --test-process_batch:1 --verbose

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_cnn.batch_processing import *  # NOQA
        >>> from ibeis_cnn import models
        >>> model = models.DummyModel(batch_size=128)
        >>> X, y = model.make_random_testdata(num=2000, seed=None)
        >>> model.initialize_architecture()
        >>> theano_funcs = model.build_theano_funcs(request_predict=True)
        >>> theano_fn = theano_funcs[1]
        >>> kwargs = {'X_is_cv2_native': False, 'showprog': True, 'randomize_batch_order': True}
        >>> outputs_ = process_batch(model, X_train, y_train, theano_fn, **kwargs)
        >>> result = ut.dict_str(outputs_)
        >>> print(result)

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_cnn.batch_processing import *  # NOQA
        >>> from ibeis_cnn import models
        >>> model = models.SiameseL2(batch_size=128, data_shape=(32, 32, 1), strict_batch_size=True)
        >>> X, y = model.make_random_testdata(num=2000, seed=None)
        >>> model.initialize_architecture()
        >>> theano_funcs = model.build_theano_funcs(request_predict=True)
        >>> theano_fn = theano_funcs[1]
        >>> kwargs = {'X_is_cv2_native': False, 'showprog': True, 'randomize_batch_order': True}
        >>> outputs_ = process_batch(model, X_train, y_train, theano_fn, **kwargs)
        >>> result = ut.dict_str(outputs_)
        >>> print(result)

    Ignore:
        Xb, yb = batch_iter.next()
        assert Xb.shape == (8, 1, 4, 4)
        yb.shape == (8,)
    """
    batch_output_list = []
    output_names = [
        str(outexpr.variable) if outexpr.variable.name is None else outexpr.variable.name
        for outexpr in theano_fn.outputs
    ]
    batch_target_list = []  # augmented label list
    show = VERBOSE_BATCH or show

    # Break data into generated batches
    # generated data with explicit iteration
    batch_iter = batch_iterator(model, X, y, lbl=theano_fn.name, **kwargs)
    if buffered:
        batch_iter = ut.buffered_generator(batch_iter)
    if y is None:
        # Labels are not known, only one argument
        for Xb, yb in batch_iter:
            batch_output = theano_fn(Xb)
            batch_output_list.append(batch_output)
    else:
        # TODO: sliced batches
        for Xb, yb in batch_iter:
            # Runs a batch through the network and updates the weights. Just
            # returns what it did
            batch_output = theano_fn(Xb, yb)
            batch_output_list.append(batch_output)
            batch_target_list.append(yb)

            if show:
                # Print the network output for the first batch
                print('--------------')
                print(ut.list_str(zip(output_names, batch_output)))
                print('Correct: ', yb)
                print('--------------')
                show = False

    # get outputs of each type
    if ut.inIPython():
        unstacked_output_gen = [[bop[count] for bop in batch_output_list]
                                for count, name in enumerate(output_names)]
    else:
        unstacked_output_gen = ([bop[count] for bop in batch_output_list]
                                for count, name in enumerate(output_names))

    if spatial:
        unstacked_output_gen = list(unstacked_output_gen)
        stacked_output_list = [ [] for _ in range(len(unstacked_output_gen)) ]
        for index, output in enumerate(unstacked_output_gen):
            output = np.vstack(output)
            stacked_output_list[index] = output
    else:
        stacked_output_list  = [utils.concatenate_hack(_output_unstacked, axis=0)
                                for _output_unstacked in unstacked_output_gen]

    outputs_ = dict(zip(output_names, stacked_output_list))

    if y  is not None:
        auglbl_list = np.hstack(batch_target_list)
        outputs_['auglbl_list'] = auglbl_list

    if fix_output:
        # batch iteration may wrap-around returned data. slice of the padding
        num_inputs = X.shape[0] / model.data_per_label_input
        num_outputs = num_inputs * model.data_per_label_output
        for key in six.iterkeys(outputs_):
            outputs_[key] = outputs_[key][0:num_outputs]

    if getattr(model, 'encoder', None) is not None and 'predictions' in outputs_:
        outputs_['labeled_predictions'] = model.encoder.inverse_transform(outputs_['predictions'])
    return outputs_


@profile
def batch_iterator(model, X, y, randomize_batch_order=False, augment_on=False,
                   X_is_cv2_native=True, verbose=VERBOSE_BATCH,
                   veryverbose=VERYVERBOSE_BATCH, showprog=ut.VERBOSE,
                   lbl='verbose batch iteration',
                   time_thresh=10, time_thresh_growth=1.0):
    r"""
    CommandLine:
        python -m ibeis_cnn.batch_processing --test-batch_iterator:0
        python -m ibeis_cnn.batch_processing --test-batch_iterator:1
        python -m ibeis_cnn.batch_processing --test-batch_iterator:1 --DEBUG_AUGMENTATION

        python -m ibeis_cnn.batch_processing --test-batch_iterator:1 --noaugment
        python -m ibeis_cnn.batch_processing --test-batch_iterator:1 --augment

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_cnn.batch_processing import *  # NOQA
        >>> from ibeis_cnn import models
        >>> # build test data
        >>> model = models.DummyModel(batch_size=16, strict_batch_size=False)
        >>> X, y = model.make_random_testdata(num=99, seed=None, cv2_format=True)
        >>> model.ensure_training_state(X, y)
        >>> encoder = None
        >>> randomize_batch_order = True
        >>> # execute function
        >>> result_list = [(Xb, Yb) for Xb, Yb in batch_iterator(model, X, y, randomize_batch_order)]
        >>> # verify results
        >>> result = ut.depth_profile(result_list, compress_consecutive=True)
        >>> print(result)
        [[(16, 1, 4, 4), 16]] * 6 + [[(3, 1, 4, 4), 3]]

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_cnn.batch_processing import *  # NOQA
        >>> from ibeis_cnn import models
        >>> # build test data
        >>> model = models.SiameseL2(batch_size=128, data_shape=(32, 32, 1), strict_batch_size=True)
        >>> X, y = model.make_random_testdata(num=4000, seed=None, cv2_format=True)
        >>> model.ensure_training_state(X, y)
        >>> encoder = None
        >>> # execute function
        >>> result_list1 = []
        >>> result_list2 = []
        >>> iterkw = dict(randomize_batch_order=True, augment_on=not ut.get_argflag('--noaugment'), showprog=True, verbose=ut.VERBOSE)
        >>> import time
        >>> sleep_time = .2
        >>> with ut.Timer('unbuffered') as t:
        >>>     generator =  batch_iterator(model, X, y, **iterkw)
        >>>     for Xb, Yb in generator:
        >>>         time.sleep(sleep_time)
        >>>         result_list1.append(Xb.shape)
        >>> with ut.Timer('buffered') as t:
        >>>     generator =  batch_iterator(model, X, y, **iterkw)
        >>>     for Xb, Yb in ut.buffered_generator(generator):
        >>>         time.sleep(sleep_time)
        >>>         result_list2.append(Xb.shape)
        >>> assert result_list1 == result_list2
        >>> print(len(result_list2))
        >>> # verify results
        >>> #result = ut.depth_profile(result_list, compress_consecutive=True)
    """
    data_per_label_input = model.data_per_label_input
    # need to be careful with batchsizes if directly specified to theano
    equal_batch_sizes = model.input_shape[0] is not None
    augment_on = augment_on and hasattr(model, 'augment')
    encoder = getattr(model, 'encoder', None)
    # divides X and y into batches of size bs for sending to the GPU
    if randomize_batch_order:
        # Randomly shuffle data
        X, y = utils.data_label_shuffle(X, y, data_per_label_input)  # 0.079 mnist time fraction
    if verbose:
        print('[batchiter] BEGIN')
        print('[batchiter] X.shape %r' % (X.shape, ))
        if y is not None:
            print('[batchiter] y.shape %r' % (y.shape, ))
        print('[batchiter] augment_on %r' % (augment_on, ))
        print('[batchiter] encoder %r' % (encoder, ))
        print('[batchiter] equal_batch_sizes %r' % (equal_batch_sizes, ))
        print('[batchiter] data_per_label_input %r' % (data_per_label_input, ))
    if y is not None:
        assert X.shape[0] == (y.shape[0] * data_per_label_input), 'bad data / label alignment'
    batch_size = model.batch_size
    num_batches = (X.shape[0] + batch_size - 1) // batch_size
    if verbose:
        print('[batchiter] num_batches = %r' % (num_batches,))

    batch_index_iter = range(num_batches)
    # FIXME: put in a layer?
    center_mean = None
    center_std  = None
    if model.preproc_kw is not None:
        center_std  = np.array(model.preproc_kw['center_std'], dtype=np.float32)
        center_mean = np.array(model.preproc_kw['center_mean'], dtype=np.float32)
    do_whitening = center_mean is not None and center_std is not None and center_std != 0.0
    assert do_whitening, 'should be whitening'

    if showprog:
        # progress iterator should be outside of this function
        batch_index_iter = ut.ProgressIter(batch_index_iter,
                                           nTotal=num_batches, lbl=lbl,
                                           time_thresh=time_thresh,
                                           time_thresh_growth=time_thresh_growth)

    DEBUG_AUGMENTATION = ut.get_argflag('--DEBUG_AUGMENTATION')

    # messy messy messy
    needs_convert = ut.is_int(X)
    if needs_convert:
        ceneter_mean01 = center_mean / np.array(255.0, dtype=np.float32)
        center_std01 = center_std / np.array(255.0, dtype=np.float32)
    else:
        ceneter_mean01 = center_mean
        center_std01 = center_std

    for batch_index in batch_index_iter:
        # Get batch slice
        Xb_orig, yb_orig = utils.slice_data_labels(
            X, y, batch_size, batch_index,
            data_per_label_input, wraparound=equal_batch_sizes)  # .113 time fraction
        # FIRST CONVERT TO 0/1
        Xb = Xb_orig.copy().astype(np.float32)
        if needs_convert:
            Xb /= 255.0
        if yb_orig is not None:
            yb = yb_orig.copy()
        else:
            yb = None
        # Augment
        # MAKE SURE DATA AUGMENTATION HAS MEAN FILL VALUES NOT 0
        # AUGMENT DATA IN 0-1 SPACE
        if augment_on:
            if verbose or veryverbose:
                if veryverbose or (batch_index + 1) % num_batches <= 1:
                    print('Augmenting Data')
                    # only copy if we have't yet
            Xb, yb = model.augment(Xb, yb)
            if DEBUG_AUGMENTATION:
                #Xb, yb = augment.augment_siamese_patches2(Xb, yb)
                from ibeis_cnn import augment
                import plottool as pt
                '''
                from ibeis_cnn import augment
                import plottool as pt
                import IPython; IPython.get_ipython().magic('pylab qt4')
                augment.show_augmented_patches(Xb_orig, Xb, yb_orig, yb)
                '''
                augment.show_augmented_patches(Xb_orig, Xb, yb_orig, yb)
                pt.show_if_requested()
                ut.embed()
        # DO WHITENING AFTER DATA AUGMENTATION
        # MOVE DATA INTO -1 to 1 space
        # Whiten (applies centering), not really whitening
        if do_whitening:
            # .563 time fraction
            Xb = (Xb - (ceneter_mean01)) / (center_std01,)
        # Encode
        if yb is not None:
            if encoder is not None:
                yb = encoder.transform(yb)  # .201 time fraction
            # Get corret dtype for y (after encoding)
            if data_per_label_input > 1:
                # TODO: FIX data_per_label_input ISSUES
                if getattr(model, 'needs_padding', False):
                    # most models will do the padding implicitly in the layer architecture
                    yb_buffer = -np.ones(len(yb) * (data_per_label_input - 1), np.int32)
                    yb = np.hstack((yb, yb_buffer))
            yb = yb.astype(np.int32)
        # Convert cv2 format to Lasagne format for batching
        if X_is_cv2_native:
            Xb = Xb.transpose((0, 3, 1, 2))
        if verbose or veryverbose:
            if veryverbose or (batch_index + 1) % num_batches <= 1:
                print('[batchiter] Yielding batch: batch_index = %r ' % (batch_index,))
                print('[batchiter]   * Xb.shape = %r, Xb.dtype=%r' % (Xb.shape, Xb.dtype))
                print('[batchiter]   * yb.shape = %r, yb.dtype=%r' % (yb.shape, yb.dtype))
                print('[batchiter]   * yb.sum = %r' % (yb.sum(),))
        # Ugg, we can't have data and labels of different lengths
        #del Xb_orig
        #del yb_orig
        yield Xb, yb
    if verbose:
        print('[batchiter] END')


def build_theano_funcs(model,
                        input_type=T.tensor4, output_type=T.ivector,
                        request_backprop=True,
                        request_forward=True,
                        request_predict=False):
    """
    build the Theano functions (symbolic expressions) that will be used in the
    optimization refer to this link for info on tensor types:

    References:
        http://deeplearning.net/software/theano/library/tensor/basic.html
    """
    print('[batch.build_theano_funcs] enter')
    X = input_type('x')
    y = output_type('y')
    X_batch = input_type('x_batch')
    y_batch = output_type('y_batch')

    loss, loss_determ, loss_regularized, network_output, network_output_determ = model.build_loss_expressions(X_batch, y_batch)
    #network_output = lasagne.layers.get_output(model.output_layer, X_batch, deterministic=True)
    #network_output.name = 'network_output'

    # Run inference and get other_outputs
    unlabeled_outputs = model.build_unlabeled_output_expressions(network_output_determ)
    labeled_outputs   = model.build_labeled_output_expressions(network_output_determ, y_batch)
    updates = None

    if request_backprop:
        print('[batch.build_theano_funcs] request_backprop')
        learning_rate_theano = model.shared_learning_rate
        momentum = model.learning_state['momentum']
        # Define how to update network parameters based on the training loss
        parameters = model.get_all_params(trainable=True)
        gradients_regularized = theano.grad(loss_regularized, parameters, add_names=True)
        updates = lasagne.updates.nesterov_momentum(gradients_regularized, parameters, learning_rate_theano, momentum)

        # Build outputs to babysit training
        import theano.tensor as T
        monitor_outputs = []
        for param in parameters:
            # The vector each param was udpated with
            param_update_vector = updates[param] - param
            param_update_vector.name = 'param_update_vector_' + param.name
            #param_update_magnitude = (param_update_vector.reshape(param_update_vector.shape[0], -1) ** 2).sum(-1)
            flat_param_update_vector = param_update_vector.reshape((param_update_vector.shape[0], T.prod(param_update_vector.shape[1:])))
            param_update_magnitude = (flat_param_update_vector ** 2).sum(-1)
            param_update_magnitude.name = 'param_update_magnitude_' + param.name
            monitor_outputs.append(param_update_magnitude)

        theano_backprop = theano.function(
            inputs=[theano.Param(X_batch), theano.Param(y_batch)],
            outputs=[loss_regularized, loss] + labeled_outputs + monitor_outputs,
            updates=updates,
            givens={
                X: X_batch,
                y: y_batch,
            },
        )
        #theano_backprop = theano.function(
        #    inputs=[theano.Param(X_batch), theano.Param(y_batch)],
        #    outputs=[loss_regularized] + labeled_outputs,
        #    updates=updates,
        #    givens={
        #        X: X_batch,
        #        y: y_batch,
        #    },
        #)
        theano_backprop.name = ':theano_backprob:explicit'
    else:
        theano_backprop = None

    if request_forward:
        print('[batch.build_theano_funcs] request_forward')
        theano_forward = theano.function(
            inputs=[theano.Param(X_batch), theano.Param(y_batch)],
            outputs=[loss_determ] + labeled_outputs + unlabeled_outputs,
            updates=None,
            givens={
                X: X_batch,
                y: y_batch,
            },
        )
        theano_forward.name = ':theano_forward:explicit'
    else:
        theano_forward = None

    if request_predict:
        print('[batch.build_theano_funcs] request_predict')
        theano_predict = theano.function(
            inputs=[theano.Param(X_batch)],
            outputs=[network_output_determ] + unlabeled_outputs,
            updates=None,
            givens={
                X: X_batch,
            },
        )
        theano_predict.name = ':theano_predict:explicit'
    else:
        theano_predict = None

    print('[batch.build_theano_funcs] exit')
    return theano_backprop, theano_forward, theano_predict, updates


def output_confusion_matrix(X_test, results_path,  test_results, model, **kwargs):
    """ currently hacky implementation, fix it later """
    loss, accu_test, prob_list, auglbl_list, pred_list, conf_list = test_results
    # Output confusion matrix
    # Grab model
    #model = kwargs.get('model', None)
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
    draw_net.show_confusion_matrix(auglbl_list, pred_list, label_list, results_path,
                                   mapping_fn, X_test)


#def create_sliced_iter_funcs_train2(model, X_unshared, y_unshared):
#    """
#    WIP: NEW IMPLEMENTATION WITH PRELOADING GPU DATA

#    build the Theano functions (symbolic expressions) that will be used in the
#    optimization refer to this link for info on tensor types:

#    References:
#        http://deeplearning.net/software/theano/library/tensor/basic.html
#        http://deeplearning.net/software/theano/tutorial/aliasing.html#borrowing-when-creating-shared-variables
#        http://deeplearning.net/tutorial/lenet.html
#        # TODO: Deal with batching to the GPU by setting the value of the shared variables.

#    CommandLine:
#        python -m ibeis_cnn.batch_processing --test-create_sliced_iter_funcs_train2

#    Example:
#        >>> # DISABLE_DOCTEST
#        >>> from ibeis_cnn.batch_processing import *  # NOQA
#        >>> from ibeis_cnn import draw_net
#        >>> from ibeis_cnn import models
#        >>> model = models.DummyModel(autoinit=True)
#        >>> X_unshared, y_unshared = model.make_random_testdata()
#        >>> train_iter = model.build_theano_funcs(model)
#        >>> print(train_iter)
#        >>> loss_train, newtork_output, prediction, accuracy = train_iter(0)
#        >>> print('loss = %r' % (loss,))
#        >>> print('net_out = %r' % (outvec,))
#        >>> print('newtork_output = %r' % (newtork_output,))
#        >>> print('accuracy = %r' % (accuracy,))
#        >>> #draw_net.draw_theano_symbolic_expression(train_iter)
#        >>> assert outvec.shape == (model.batch_size, model.output_dims)
#    """
#    # Attempt to load data on to the GPU
#    # Labels to go into the GPU as float32 and then cast to int32 once inside
#    X_unshared = np.asarray(X_unshared, dtype=theano.config.floatX)
#    y_unshared = np.asarray(y_unshared, dtype=theano.config.floatX)

#    X_shared = theano.shared(X_unshared, borrow=True)
#    y_shared = T.cast(theano.shared(y_unshared, borrow=True), 'int32')

#    # Build expressions which sample a batch
#    batch_size = model.batch_size

#    # Initialize symbolic input variables
#    index = T.lscalar(name='index')
#    X_batch = T.tensor4(name='X_batch')
#    y_batch = T.ivector(name='y_batch')

#    WHITEN = False
#    if WHITEN:
#        # We might be able to perform some data augmentation here symbolicly
#        data_mean = X_unshared.mean()
#        data_std = X_unshared.std()
#        givens = {
#            X_batch: (X_shared[index * batch_size: (index + 1) * batch_size] - data_mean) / data_std,
#            y_batch: y_shared[index * batch_size: (index + 1) * batch_size],
#        }
#    else:
#        givens = {
#            X_batch: X_shared[index * batch_size: (index + 1) * batch_size],
#            y_batch: y_shared[index * batch_size: (index + 1) * batch_size],
#        }

#    output_layer = model.get_output_layer()

#    # Build expression to evalute network output without dropout
#    #newtork_output = output_layer.get_output(X_batch, deterministic=True)
#    newtork_output = layers.get_output(output_layer, X_batch, deterministic=True)
#    newtork_output.name = 'network_output'

#    # Build expression to evaluate loss
#    objective = objectives.Objective(output_layer, loss_function=model.loss_function)
#    loss_train = objective.get_loss(X_batch, target=y_batch)  # + 0.0001 * lasagne.regularization.l2(output_layer)
#    loss_train.name = 'loss_train'

#    # Build expression to evaluate updates
#    with warnings.catch_warnings():
#        warnings.filterwarnings('ignore', '.*topo.*')
#        all_params = lasagne.layers.get_all_params(output_layer, trainable=True)
#    updates = lasagne.updates.nesterov_momentum(loss_train, all_params, model.learning_rate, model.momentum)

#    # Get performance indicator outputs:

#    # Build expression to convert network output into a prediction
#    prediction = model.make_prediction_expr(newtork_output)

#    # Build expression to compute accuracy
#    accuracy = model.make_accuracy_expr(prediction, y_batch)

#    theano_backprop = theano.function(
#        inputs=[index],
#        outputs=[loss_train, newtork_output, prediction, accuracy],
#        updates=updates,
#        givens=givens
#    )
#    theano_backprop.name += ':theano_backprob:indexed'

#    #other_outputs = [probabilities, predictions, confidences]

#    #theano_backprop = theano.function(
#    #    inputs=[theano.Param(X_batch), theano.Param(y_batch)],
#    #    outputs=[loss] + other_outputs,
#    #    updates=updates,
#    #    givens={
#    #        X: X_batch,
#    #        y: y_batch,
#    #    },
#    #)

#    #theano_forward = theano.function(
#    #    inputs=[theano.Param(X_batch), theano.Param(y_batch)],
#    #    outputs=[loss_determ] + other_outputs,
#    #    updates=None,
#    #    givens={
#    #        X: X_batch,
#    #        y: y_batch,
#    #    },
#    #)

#    #theano_predict = theano.function(
#    #    inputs=[theano.Param(X_batch)],
#    #    outputs=other_outputs,
#    #    updates=None,
#    #    givens={
#    #        X: X_batch,
#    #    },
#    #)

#    return theano_backprop


#def create_sliced_network_output_func(model):
#    # Initialize symbolic input variables
#    X_batch = T.tensor4(name='X_batch')
#    # weird, idk why X and y exist
#    X = T.tensor4(name='X_batch')

#    output_layer = model.get_output_layer()

#    # Build expression to evalute network output without dropout
#    #newtork_output = output_layer.get_output(X_batch, deterministic=True)
#    newtork_output = layers.get_output(output_layer, X_batch, deterministic=True)
#    newtork_output.name = 'network_output'

#    theano_forward = theano.function(
#        inputs=[theano.Param(X_batch)],
#        outputs=[newtork_output],
#        givens={
#            X: X_batch,
#        }
#    )
#    theano_forward.name += ':theano_forward:sliced'
#    return theano_forward


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis_cnn.batch_processing
        python -m ibeis_cnn.batch_processing --allexamples
        python -m ibeis_cnn.batch_processing --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
