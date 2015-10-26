# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from ibeis_cnn import utils
import numpy as np
import six
import utool as ut
#import warnings
print, rrr, profile = ut.inject2(__name__, '[ibeis_cnn.batch_processing]')


VERBOSE_BATCH = (ut.get_argflag(('--verbose-batch', '--verbbatch')) or
                 utils.VERBOSE_CNN)
VERYVERBOSE_BATCH = (
    ut.get_argflag(('--veryverbose-batch', '--veryverbbatch')) or
    ut.VERYVERBOSE)


def process_batch(model, X, y, theano_fn, fix_output=False, buffered=False,
                  show=False, spatial=False, **kwargs):
    """
    compute the loss over all training batches

    CommandLine:
        python -m ibeis_cnn --tf process_batch --verbose
        python -m ibeis_cnn --tf process_batch:0 --verbose
        python -m ibeis_cnn --tf process_batch:1 --verbose

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_cnn.batch_processing import *  # NOQA
        >>> from ibeis_cnn import models
        >>> model = models.DummyModel(batch_size=128)
        >>> X, y = model.make_random_testdata(num=2000, seed=None)
        >>> model.initialize_architecture()
        >>> theano_funcs = model.build_theano_funcs(request_predict=True)
        >>> theano_fn = theano_funcs.theano_forward
        >>> kwargs = {'X_is_cv2_native': False, 'showprog': True,
        ...           'randomize_batch_order': True}
        >>> outputs_ = process_batch(model, X, y, theano_fn, **kwargs)
        >>> result = ut.dict_str(outputs_)
        >>> print(result)

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_cnn.batch_processing import *  # NOQA
        >>> from ibeis_cnn import models
        >>> model = models.SiameseL2(batch_size=128, data_shape=(32, 32, 1),
        ...                          strict_batch_size=True)
        >>> X, y = model.make_random_testdata(num=2000, seed=None)
        >>> model.initialize_architecture()
        >>> theano_funcs = model.build_theano_funcs(request_predict=True)
        >>> theano_fn = theano_funcs[1]
        >>> kwargs = {'X_is_cv2_native': False, 'showprog': True,
        ...           'randomize_batch_order': True}
        >>> outputs_ = process_batch(model, X, y, theano_fn, **kwargs)
        >>> result = ut.dict_str(outputs_)
        >>> print(result)

    Ignore:
        Xb, yb = batch_iter.next()
        assert Xb.shape == (8, 1, 4, 4)
        yb.shape == (8,)

    Ignore:
        X, y = model.make_random_testdata(num=2000, seed=None)
        kwargs = {'X_is_cv2_native': False, 'showprog': True,
                  'randomize_batch_order': True, 'time_thresh': .5,
                  'adjust': False,
                  }

        print('Testing Unbuffered')
        batch_iter = batch_iterator(model, X, y, lbl=theano_fn.name, **kwargs)
        for Xb, yb in ut.ProgressIter(batch_iter, lbl=':EXEC FG'):
            [ut.is_prime(346373) for _ in range(2)]

        # Notice how the progress iters are not interlaced like
        # they are in the unbuffered version
        import sys
        sys.stdout.flush()
        print('Testing Buffered')
        sys.stdout.flush()
        batch_iter2 = batch_iterator(model, X, y, lbl=theano_fn.name, **kwargs)
        batch_iter2 = ut.buffered_generator(batch_iter2, buffer_size=4)
        print('Iterating')
        for Xb, yb in ut.ProgressIter(batch_iter2, lbl=':EXEC FG'):
            [ut.is_prime(346373) for _ in range(2)]
    """
    batch_output_list = []
    output_names = [
        str(outexpr.variable)
        if outexpr.variable.name is None else
        outexpr.variable.name
        for outexpr in theano_fn.outputs
    ]
    # augmented label list
    batch_target_list = []
    show = VERBOSE_BATCH or show

    # Break data into generated batches
    # generated data with explicit iteration
    batch_iter = batch_iterator(model, X, y, lbl=theano_fn.name, **kwargs)
    if buffered:
        batch_iter = ut.buffered_generator(batch_iter)
    if y is None:
        # Labels are not known, only one argument
        for Xb, yb in batch_iter:
            pass
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
    unstacked_output_gen = ([bop[count] for bop in batch_output_list]
                            for count, name in enumerate(output_names))

    if spatial:
        unstacked_output_gen = list(unstacked_output_gen)
        stacked_output_list = [[] for _ in range(len(unstacked_output_gen))]
        for index, output in enumerate(unstacked_output_gen):
            output = np.vstack(output)
            stacked_output_list[index] = output
    else:
        stacked_output_list  = [
            concatenate_hack(_output_unstacked, axis=0)
            for _output_unstacked in unstacked_output_gen
        ]

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

    encoder = getattr(model, 'encoder', None)
    if encoder is not None and 'predictions' in outputs_:
        pred = outputs_['predictions']
        outputs_['labeled_predictions'] = encoder.inverse_transform(pred)
    return outputs_


@profile
def batch_iterator(model, X, y, randomize_batch_order=False, augment_on=False,
                   X_is_cv2_native=True, verbose=VERBOSE_BATCH,
                   veryverbose=VERYVERBOSE_BATCH, showprog=ut.VERBOSE,
                   lbl='verbose batch iteration',
                   time_thresh=10, time_thresh_growth=1.0, adjust=True):
    r"""
    Breaks up data into to batches

    CommandLine:
        python -m ibeis_cnn --tf batch_iterator:0
        python -m ibeis_cnn --tf batch_iterator:1
        python -m ibeis_cnn --tf batch_iterator:1 --DEBUG_AUGMENTATION

        python -m ibeis_cnn --tf batch_iterator:1 --noaugment
        # Threaded buffering seems to help a lot
        python -m ibeis_cnn --tf batch_iterator:1 --augment

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
        >>> result_list = [(Xb, Yb) for Xb, Yb in batch_iterator(model, X, y,
        ...                randomize_batch_order)]
        >>> # verify results
        >>> result = ut.depth_profile(result_list, compress_consecutive=True)
        >>> print(result)
        [[(16, 1, 4, 4), 16]] * 6 + [[(3, 1, 4, 4), 3]]

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_cnn.batch_processing import *  # NOQA
        >>> from ibeis_cnn import models
        >>> import time
        >>> # build test data
        >>> model = models.SiameseL2(batch_size=128, data_shape=(8, 8, 1),
        ...                          strict_batch_size=True)
        >>> X, y = model.make_random_testdata(num=1000, seed=None, cv2_format=True)
        >>> model.ensure_training_state(X, y)
        >>> encoder = None
        >>> # execute function
        >>> result_list1 = []
        >>> result_list2 = []
        >>> augment_on=not ut.get_argflag('--noaugment')
        >>> iterkw = dict(randomize_batch_order=True,
        >>>              augment_on=augment_on,
        >>>              showprog=True, verbose=ut.VERBOSE)
        >>> sleep_time = .05
        >>> inside_time1 = 0
        >>> inside_time2 = 0
        >>> with ut.Timer('buffered') as t2:
        >>>     generator =  batch_iterator(model, X, y, **iterkw)
        >>>     for Xb, Yb in ut.buffered_generator(generator, buffer_size=3):
        >>>         with ut.Timer('Inside', verbose=False) as t:
        >>>             time.sleep(sleep_time)
        >>>             result_list2.append(Xb.shape)
        >>>         inside_time2 += t.ellapsed
        >>> with ut.Timer('unbuffered') as t1:
        >>>     generator =  batch_iterator(model, X, y, **iterkw)
        >>>     for Xb, Yb in generator:
        >>>         with ut.Timer('Inside', verbose=False) as t:
        >>>             time.sleep(sleep_time)
        >>>             result_list1.append(Xb.shape)
        >>>         inside_time1 += t.ellapsed
        >>> print('\nInside times should be the same')
        >>> print('inside_time1 = %r' % (inside_time1,))
        >>> print('inside_time2 = %r' % (inside_time2,))
        >>> print('Outside times show the overhead of data augmentation')
        >>> print('Overhead Unbuffered = %r' % (t1.ellapsed - inside_time1,))
        >>> print('Overhead Buffered   = %r' % (t2.ellapsed - inside_time2,))
        >>> print('Efficiency Unbuffered  = %.2f' % (100 * inside_time1 / t1.ellapsed,))
        >>> print('Efficiency Buffered    = %.2f' % (100 * inside_time2 / t2.ellapsed,))
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
        # 0.079 mnist time fraction
        X, y = utils.data_label_shuffle(X, y, data_per_label_input)
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
        assert X.shape[0] == (y.shape[0] * data_per_label_input), (
            'bad data / label alignment')
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
    do_whitening = (center_mean is not None and
                    center_std is not None and
                    center_std != 0.0)
    #assert do_whitening, 'should be whitening'

    if showprog:
        # progress iterator should be outside of this function
        batch_index_iter = ut.ProgressIter(batch_index_iter,
                                           nTotal=num_batches, lbl=lbl,
                                           time_thresh=time_thresh,
                                           time_thresh_growth=time_thresh_growth,
                                           adjust=adjust)

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
        # .113 time fraction
        Xb_orig, yb_orig = utils.slice_data_labels(
            X, y, batch_size, batch_index,
            data_per_label_input, wraparound=equal_batch_sizes)
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
                    # most models will do the padding implicitly
                    # in the layer architecture
                    pad_size = len(yb) * (data_per_label_input - 1)
                    yb_buffer = -np.ones(pad_size, np.int32)
                    yb = np.hstack((yb, yb_buffer))
            yb = yb.astype(np.int32)
        # Convert cv2 format to Lasagne format for batching
        if X_is_cv2_native:
            Xb = Xb.transpose((0, 3, 1, 2))
        if verbose or veryverbose:
            if veryverbose or (batch_index + 1) % num_batches <= 1:
                print('[batchiter] Yielding batch: batch_index = %r ' % (
                    batch_index,))
                print('[batchiter]   * Xb.shape = %r, Xb.dtype=%r' % (
                    Xb.shape, Xb.dtype))
                print('[batchiter]   * yb.shape = %r, yb.dtype=%r' % (
                    yb.shape, yb.dtype))
                print('[batchiter]   * yb.sum = %r' % (yb.sum(),))
        # Ugg, we can't have data and labels of different lengths
        #del Xb_orig
        #del yb_orig
        yield Xb, yb
    if verbose:
        print('[batchiter] END')


def concatenate_hack(sequence, axis=0):
    # Hack to fix numpy bug. concatenate should do hstacks on 0-dim arrays
    if len(sequence) > 0 and len(sequence[1].shape) == 0:
        res = np.hstack(sequence)
    else:
        res = np.concatenate(sequence, axis=axis)
    return res


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
