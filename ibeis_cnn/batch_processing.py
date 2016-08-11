# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from ibeis_cnn import utils
import numpy as np
import six
import utool as ut
#import warnings
print, rrr, profile = ut.inject2(__name__, '[ibeis_cnn.batch_processing]')


VERBOSE_BATCH = ut.get_module_verbosity_flags('batch')[0] or utils.VERBOSE_CNN
if ut.VERYVERBOSE:
    VERBOSE_BATCH = 2

DEBUG_AUGMENTATION = ut.get_argflag('--DEBUG_AUGMENTATION')


@profile
def process_batch(model, X, y, theano_fn, fix_output=False, buffered=False,
                  show=False, spatial=False, **kwargs):
    """
    Compute the loss over all training batches.
    Passes data to function that splits it into batches and appropriately
    preproecsses the data. Then this function sends that data to theano. Then
    the results are packaged up nicely before returning.

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
        >>> theano_fn = model.build_predict_func()
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
        >>> theano_fn = model.build_predict_func()
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
    batch_iter = batch_iterator(model, X, y, **kwargs)
    if buffered:
        batch_iter = ut.buffered_generator(batch_iter)

    showprog = True

    if showprog:
        bs = VERBOSE_BATCH < 1
        num_batches = (X.shape[0] + model.batch_size - 1) // model.batch_size
        # progress iterator should be outside of this function
        batch_iter = ut.ProgressIter(batch_iter, nTotal=num_batches, lbl=theano_fn.name,
                                     freq=10, bs=bs, adjust=True)
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
        # batch iteration may wrap-around returned data. slice off the padding
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
                   X_is_cv2_native=True, verbose=None, showprog=None,
                   lbl='verbose batch iteration'):
    r"""
    Breaks up data into to batches defined by model batch size

    CommandLine:
        python -m ibeis_cnn --tf batch_iterator:0
        python -m ibeis_cnn --tf batch_iterator:1
        python -m ibeis_cnn --tf batch_iterator:2
        python -m ibeis_cnn --tf batch_iterator:1 --DEBUG_AUGMENTATION

        python -m ibeis_cnn --tf batch_iterator:1 --noaugment
        # Threaded buffering seems to help a lot
        python -m ibeis_cnn --tf batch_iterator:1 --augment

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_cnn.batch_processing import *  # NOQA
        >>> from ibeis_cnn import models
        >>> model = models.DummyModel(batch_size=16)
        >>> X, y = model.make_random_testdata(num=99, seed=None, cv2_format=True)
        >>> model.ensure_training_state(X, y)
        >>> y = None
        >>> encoder = None
        >>> randomize_batch_order = True
        >>> result_list = [(Xb, Yb) for Xb, Yb in batch_iterator(model, X, y,
        ...                randomize_batch_order)]
        >>> result = ut.depth_profile(result_list, compress_consecutive=True)
        >>> print(result)
        [[(16, 1, 4, 4), 16]] * 6 + [[(3, 1, 4, 4), 3]]

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_cnn.batch_processing import *  # NOQA
        >>> from ibeis_cnn import models
        >>> import time
        >>> model = models.SiameseL2(batch_size=128, data_shape=(8, 8, 1))
        >>> X, y = model.make_random_testdata(num=1000, seed=None, cv2_format=True)
        >>> model.ensure_training_state(X, y)
        >>> encoder = None
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

    Example2:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.batch_processing import *  # NOQA
        >>> from ibeis_cnn.models.mnist import MNISTModel
        >>> from ibeis_cnn import ingest_data
        >>> # should yield float32 regardlesss of original format
        >>> ut.exec_funckw(batch_iterator, globals())
        >>> randomize_batch_order = False
        >>> # ---
        >>> dataset1 = ingest_data.grab_mnist_category_dataset_float()
        >>> model1 = MNISTModel(batch_size=8, data_shape=dataset1.data_shape,
        >>>                    output_dims=dataset1.output_dims,
        >>>                    arch_tag=dataset1.alias_key,
        >>>                    training_dpath=dataset1.training_dpath)
        >>> X1, y1 = dataset1.load_subset('train')
        >>> model1.ensure_training_state(X1, y1)
        >>> _iter1 = batch_iterator(model1, X1, y1, randomize_batch_order)
        >>> Xb1, yb1 = six.next(_iter1)
        >>> # ---
        >>> dataset2 = ingest_data.grab_mnist_category_dataset()
        >>> model2 = MNISTModel(batch_size=8, data_shape=dataset2.data_shape,
        >>>                    output_dims=dataset2.output_dims,
        >>>                    arch_tag=dataset2.alias_key,
        >>>                    training_dpath=dataset2.training_dpath)
        >>> X2, y2 = dataset2.load_subset('train')
        >>> model2.ensure_training_state(X2, y2)
        >>> _iter2 = batch_iterator(model2, X2, y2, randomize_batch_order)
        >>> Xb2, yb2 = six.next(_iter2)
        >>> # ---
        >>> X, y, model = X1, y1, model1
        >>> assert np.all(yb2 == yb2)
        >>> # The uint8 and float32 data should produce similar values
        >>> # For this mnist set it will be a bit more off because the
        >>> # original uint8 scaling value was 256 not 255.
        >>> assert (Xb1[0] - Xb2[0]).max() < .1
        >>> assert np.isclose(Xb1.mean(), 0, atol=.01)
        >>> assert np.isclose(Xb2.mean(), 0, atol=.01)
        >>> assert Xb1.max() <  1.0 and Xb2.max() <  1.0, 'out of (-1, 1)'
        >>> assert Xb1.min() > -1.0 and Xb1.min() > -1.0, 'out of (-1, 1)'
        >>> assert Xb1.min() < 0, 'should have some negative values'
        >>> assert Xb2.min() < 0, 'should have some negative values'
        >>> assert Xb1.max() > 0, 'should have some positive values'
        >>> assert Xb2.max() > 0, 'should have some positive values'
    """
    if verbose:
        verbose = VERBOSE_BATCH

    if showprog is None:
        showprog = ut.VERBOSE
        showprog = True

    # need to be careful with batchsizes if directly specified to theano
    wraparound = model.input_shape[0] is not None
    augment_on = augment_on and hasattr(model, 'augment')
    encoder = getattr(model, 'encoder', None)
    needs_convert = ut.is_int(X)
    if y is not None:
        assert X.shape[0] == (y.shape[0] * model.data_per_label_input), (
            'bad data / label alignment')
    num_batches = (X.shape[0] + model.batch_size - 1) // model.batch_size

    if randomize_batch_order:
        # Randomly shuffle data
        # 0.079 mnist time fraction
        X, y = utils.data_label_shuffle(X, y, model.data_per_label_input)
    if verbose:
        print('[batchiter] BEGIN')
        print('[batchiter] X.shape %r' % (X.shape, ))
        if y is not None:
            print('[batchiter] y.shape %r' % (y.shape, ))
        print('[batchiter] augment_on %r' % (augment_on, ))
        print('[batchiter] encoder %r' % (encoder, ))
        print('[batchiter] wraparound %r' % (wraparound, ))
        print('[batchiter] model.data_per_label_input %r' % (model.data_per_label_input, ))
        print('[batchiter] needs_convert = %r' % (needs_convert,))

    # FIXME: put in a layer?
    center_mean = None
    center_std  = None
    # Load precomputed whitening parameters
    if model.preproc_kw is not None:
        center_mean = np.array(model.preproc_kw['center_mean'], dtype=np.float32)
        center_std  = np.array(model.preproc_kw['center_std'], dtype=np.float32)
    do_whitening = (center_mean is not None and
                    center_std is not None and
                    center_std != 0.0)

    if needs_convert:
        ceneter_mean01 = center_mean / np.array(255.0, dtype=np.float32)
        center_std01 = center_std / np.array(255.0, dtype=np.float32)
    else:
        ceneter_mean01 = center_mean
        center_std01 = center_std

    # Slice and preprocess data in batch
    for batch_index in range(num_batches):
        # Take a slice from the data
        Xb_orig, yb_orig = utils.slice_data_labels(
            X, y, model.batch_size, batch_index, model.data_per_label_input,
            wraparound=wraparound)
        # Ensure correct format for the GPU
        Xb = Xb_orig.astype(np.float32)
        yb = None if Xb_orig is None else yb_orig.astype(np.int32)
        if needs_convert:
            # Rescale the batch data to the range 0 to 1
            Xb = Xb / 255.0
        if augment_on:
            # Apply defined transformations
            Xb, yb, = augment_batch(model, Xb, yb, batch_index, verbose)
        if do_whitening:
            # Center the batch data in the range (-1.0, 1.0)
            Xb = (Xb - ceneter_mean01) / (center_std01)
        if X_is_cv2_native:
            # Convert from cv2 to lasange format
            Xb = Xb.transpose((0, 3, 1, 2))
        if encoder is not None:
            # Apply an encoding if applicable
            yb = encoder.transform(yb).astype(np.int32)
        if model.data_per_label_input > 1 and getattr(model, 'needs_padding', False):
            # Pad data for siamese networks
            yb = pad_labels(model, yb)
        if verbose:
            # Print info if requested
            print_batch_info(Xb, yb, verbose)
        yield Xb, yb
    if verbose:
        print('[batch] END')


def augment_batch(model, Xb, yb, batch_index, verbose):
    """
    Make sure to augment data in 0-1 space.
    This means use a mean fill values not 0.

    >>> from ibeis_cnn import augment
    >>> import plottool as pt
    >>> pt.qt4ensure()
    >>> augment.show_augmented_patches(Xb_orig, Xb, yb_orig, yb)
    """
    if verbose:
        if verbose > 1 or (batch_index + 1) % num_batches <= 1:
            print('Augmenting Data')
            # only copy if we have't yet
    Xb, yb = model.augment(Xb, yb)
    if DEBUG_AUGMENTATION:
        #Xb, yb = augment.augment_siamese_patches2(Xb, yb)
        from ibeis_cnn import augment
        import plottool as pt
        augment.show_augmented_patches(Xb_orig, Xb, yb_orig, yb)
        pt.show_if_requested()
        ut.embed()
    return Xb, yb

def pad_labels(model, yb):
    # TODO: FIX data_per_label_input ISSUES
    # most models will do the padding implicitly
    # in the layer architecture
    pad_size = len(yb) * (model.data_per_label_input - 1)
    yb_buffer = -np.ones(pad_size, dtype=np.int32)
    yb = np.hstack((yb, yb_buffer))
    return yb


def print_batch_info(Xb, yb, verbose):
    if verbose > 1 or (batch_index + 1) % num_batches <= 1:
        print('[batch] Yielding batch: batch_index = %r ' % (batch_index,))
        print('[batch]   * Xb.shape = %r, Xb.dtype=%r' % (Xb.shape, Xb.dtype))
        if yb is not None:
            print('[batch]   * yb.shape = %r, yb.dtype=%r' % (yb.shape, yb.dtype))
            print('[batch]   * yb.sum = %r' % (yb.sum(),))


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
