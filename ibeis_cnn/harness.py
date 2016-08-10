#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
constructs the Theano optimization and trains a learning model,
optionally by initializing the network with pre-trained weights.

http://cs231n.github.io/neural-networks-3/#distr

Pretrained Models:
    https://github.com/fchollet/deep-learning-models
"""
from __future__ import absolute_import, division, print_function
from six.moves import input, zip, range  # NOQA
import numpy as np
import utool as ut
import time
import cv2
print, rrr, profile = ut.inject2(__name__, '[ibeis_cnn.harness]')


def _clean(model, theano_forward, X_list, y_list, min_conf=0.95):
    from ibeis_cnn import batch_processing as batch
    import random
    # Perform testing
    clean_outputs = batch.process_batch(
        model, X_list, y_list, theano_forward, augment_on=False,
        randomize_batch_order=False)
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
    from ibeis_cnn import batch_processing as batch
    from ibeis_cnn import utils

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
            showprog=True,
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
