#!/usr/bin/env python
"""
tests a test set of data using a specified, pre-trained model and weights

python -c "import ibeis_cnn"
"""
from __future__ import absolute_import, division, print_function
# from ibeis_cnn import utils
from ibeis_cnn import models
#from ibeis_cnn import test
from ibeis_cnn import _plugin_grabmodels as grabmodels
import utool as ut
import cv2
import numpy as np
import random
print, print_, printDBG, rrr, profile = ut.inject(__name__, '[ibeis_cnn._plugin]')


try:
    from ibeis.control.controller_inject import make_ibs_register_decorator
    from ibeis.constants import Species, VIEWTEXT_TO_YAW_RADIANS
    CLASS_INJECT_KEY, register_ibs_method = make_ibs_register_decorator(__name__)
except ImportError as ex:
    register_ibs_method = ut.identity
    raise


def convert_species_viewpoint(species, viewpoint):
    species_mapping = {
        'ZEBRA_PLAINS':        Species.ZEB_PLAIN,
        'ZEBRA_GREVYS':        Species.ZEB_GREVY,
        'ELEPHANT_SAVANNA':    Species.ELEPHANT_SAV,
        'GIRAFFE_RETICULATED': Species.GIRAFFE,
        'GIRAFFE_MASAI':       Species.GIRAFFE_MASAI,
    }
    viewpoint_list = VIEWTEXT_TO_YAW_RADIANS.keys()
    viewpoint_mapping = {
        'LEFT':        viewpoint_list[4],
        'FRONT_LEFT':  viewpoint_list[3],
        'FRONT':       viewpoint_list[2],
        'FRONT_RIGHT': viewpoint_list[1],
        'RIGHT':       viewpoint_list[0],
        'BACK_RIGHT':  viewpoint_list[7],
        'BACK':        viewpoint_list[6],
        'BACK_LEFT':   viewpoint_list[5],
    }
    species_ = species_mapping[species]
    viewpoint_ = viewpoint_mapping[viewpoint]
    return species_, viewpoint_


def convert_label(label):
    species, viewpoint = label.strip().split(':')
    species = species.strip()
    viewpoint = viewpoint.strip()
    species_, viewpoint_ = convert_species_viewpoint(species, viewpoint)
    return species_, viewpoint_


@register_ibs_method
def get_neuralnet_dir(ibs):
    nets_dir = ut.unixjoin(ibs.get_cachedir(), ibs.const.PATH_NAMES.nets)
    return nets_dir


@register_ibs_method
def get_verified_aid_pairs(ibs):
    """
    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.train import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('NNP_Master3')
        >>> verified_aid1_list, verified_aid2_list = get_verified_aid_pairs(ibs)
    """
    # Grab marked hard cases
    am_rowids = ibs._get_all_annotmatch_rowids()
    remove_photobombs = True
    if remove_photobombs:
        flags = ibs.get_annotmatch_is_photobomb(am_rowids)
        am_rowids = ut.filterfalse_items(am_rowids, flags)
    verified_aid1_list = ibs.get_annotmatch_aid1(am_rowids)
    verified_aid2_list = ibs.get_annotmatch_aid2(am_rowids)
    return verified_aid1_list, verified_aid2_list


@register_ibs_method
def detect_annot_zebra_background_mask(ibs, aid_list, config2_=None):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        aid_list (int):  list of annotation ids

    Returns:
        list: species_viewpoint_list

    CommandLine:
        python -m ibeis_cnn._plugin --exec-detect_annot_zebra_background_mask --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> import ibeis_cnn
        >>> import ibeis
        >>> from ibeis_cnn._plugin import *  # NOQA
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> mask_list = detect_annot_zebra_background_mask(ibs, aid_list)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> iteract_obj = pt.interact_multi_image.MultiImageInteraction(mask_list, nPerPage=4)
        >>> #pt.imshow(mask_list[0])
        >>> ut.show_if_requested()

        #>>> from ibeis_cnn.draw_results import *  # NOQA
        #>>> from ibeis_cnn import ingest_data
        #>>> data, labels = ingest_data.testdata_patchmatch2()
        #>>> flat_metadata = {'fs': np.arange(len(labels))}
        #>>> result = interact_siamsese_data_patches(labels, data, flat_metadata)
        #>>> ut.show_if_requested()

    """
    from ibeis_cnn import harness

    # Load chips and resize to the target
    data_shape = (256, 256, 3)
    # Define model and load weights
    print('\n[harness] Loading model...')
    batch_size = int(min(128, 2 ** np.floor(np.log2(len(aid_list)))))
    model = models.BackgroundModel(batch_size=batch_size, data_shape=data_shape)

    weights_path = grabmodels.ensure_model('background', redownload=False)
    old_weights_fpath = weights_path
    model.load_old_weights_kw2(old_weights_fpath)

    # Create the Theano primitives
    # create theano symbolic expressions that define the network
    print('\n[harness] --- COMPILING SYMBOLIC THEANO FUNCTIONS ---')
    print('[model] creating Theano primitives...')
    theano_funcs = model.build_theano_funcs(request_predict=True, request_forward=False, request_backprop=False)
    theano_backprop, theano_forward, theano_predict, updates = theano_funcs

    # Read the data
    print('\n[harness] Loading chips...')
    chip_list = ibs.get_annot_chips(aid_list, verbose=True, config2_=config2_)

    print('[harness] Performing inference...')
    mask_list = []

    for chip in ut.ProgressIter(chip_list, lbl='zebra background inference', adjust=True, freq=5):
        samples, canvas_dict = harness.test_convolutional(model, theano_predict, chip, padding=24)
        mask = canvas_dict['positive']
        mask_list.append(mask)

    return mask_list


@register_ibs_method
def generate_species_background_mask(ibs, chip_fpath_list, species=None):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        aid_list (int):  list of annotation ids

    Returns:
        list: species_viewpoint_list

    CommandLine:
        python -m ibeis_cnn._plugin --exec-generate_species_background_mask --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> import ibeis_cnn
        >>> import ibeis
        >>> from ibeis_cnn._plugin import *  # NOQA
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> chip_fpath_list = ibs.get_annot_chip_fpath(aid_list)
        >>> mask_list = generate_species_background_mask(ibs, chip_fpath_list)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> iteract_obj = pt.interact_multi_image.MultiImageInteraction(mask_list, nPerPage=4)
        >>> #pt.imshow(mask_list[0])
        >>> ut.show_if_requested()

        #>>> from ibeis_cnn.draw_results import *  # NOQA
        #>>> from ibeis_cnn import ingest_data
        #>>> data, labels = ingest_data.testdata_patchmatch2()
        #>>> flat_metadata = {'fs': np.arange(len(labels))}
        #>>> result = interact_siamsese_data_patches(labels, data, flat_metadata)
        #>>> ut.show_if_requested()

    """
    from ibeis_cnn import harness

    # Load chips and resize to the target
    data_shape = (256, 256, 3)
    # Define model and load weights
    print('\n[harness] Loading model...')
    batch_size = int(min(128, 2 ** np.floor(np.log2(len(chip_fpath_list)))))
    model = models.BackgroundModel(batch_size=batch_size, data_shape=data_shape)

    weights_path = grabmodels.ensure_model('background', redownload=False)
    old_weights_fpath = weights_path
    model.load_old_weights_kw2(old_weights_fpath)

    # Create the Theano primitives
    # create theano symbolic expressions that define the network
    print('\n[harness] --- COMPILING SYMBOLIC THEANO FUNCTIONS ---')
    print('[model] creating Theano primitives...')
    theano_funcs = model.build_theano_funcs(request_predict=True, request_forward=False, request_backprop=False)
    theano_backprop, theano_forward, theano_predict, updates = theano_funcs

    # Read the data
    print('\n[harness] Loading chips...')
    import vtool as vt
    chip_list = [vt.imread(fpath) for fpath in ut.ProgressIter(chip_fpath_list, lbl='loading chips', adjust=True)]

    print('[harness] Performing inference...')
    mask_list = []

    for chip in ut.ProgressIter(chip_list, lbl='zebra background inference'):
        samples, canvas_dict = harness.test_convolutional(model, theano_predict, chip, padding=24)
        mask = canvas_dict['positive']
        mask_list.append(mask)

    return mask_list


@register_ibs_method
def detect_annot_species_viewpoint_cnn(ibs, aid_list):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        aid_list (int):  list of annotation ids

    Returns:
        list: species_viewpoint_list

    CommandLine:
        python -m ibeis_cnn._plugin --exec-detect_annot_species_viewpoint_cnn

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn._plugin import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> species_viewpoint_list = detect_annot_species_viewpoint_cnn(ibs, aid_list)
        >>> result = ('species_viewpoint_list = %s' % (str(species_viewpoint_list),))
        >>> print(result)
    """
    # Load chips and resize to the target
    data_shape = (96, 96, 3)
    # Define model and load weights
    print('Loading model...')
    batch_size = int(min(128, 2 ** np.floor(np.log2(len(aid_list)))))
    model = models.ViewpointModel(batch_size=batch_size, data_shape=data_shape)
    weights_path = grabmodels.ensure_model('viewpoint', redownload=False)
    old_weights_fpath = weights_path
    model.load_old_weights_kw(old_weights_fpath)
    # Read the data
    target = data_shape[0:2]
    print('Loading chips...')
    chip_list = ibs.get_annot_chips(aid_list, verbose=True)
    print('Resizing chips...')
    chip_list_resized = [ cv2.resize(chip, target, interpolation=cv2.INTER_LANCZOS4) for chip in ut.ProgressIter(chip_list, lbl='resizing chips') ]
    # Build data for network
    X_test = np.array(chip_list_resized, dtype=np.uint8)
    y_test = None

    from ibeis_cnn import harness
    # Predict on the data and convert labels to IBEIS namespace
    test_outputs = harness.test_data2(model, X_test, y_test)
    label_list = test_outputs['labeled_predictions']
    species_viewpoint_list = [ convert_label(label) for label in label_list ]
    #pred_list, label_list, conf_list = test.test_data(X_test, y_test, model, weights_path)
    #species_viewpoint_list = [ convert_label(label) for label in label_list ]
    return species_viewpoint_list


@register_ibs_method
def validate_annot_species_viewpoint_cnn(ibs, aid_list, verbose=False):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        aid_list (int):  list of annotation ids
        verbose (bool):  verbosity flag(default = False)

    Returns:
        tuple: (bad_species_list, bad_viewpoint_list)

    CommandLine:
        python -m ibeis_cnn._plugin --exec-validate_annot_species_viewpoint_cnn --db PZ_FlankHack
        python -m ibeis_cnn._plugin --exec-validate_annot_species_viewpoint_cnn --db GZ_Master1

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn._plugin import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> verbose = False
        >>> (bad_species_list, bad_viewpoint_list) = validate_annot_species_viewpoint_cnn(ibs, aid_list, verbose)
        >>> print('bad_species_list = %s' % (bad_species_list,))
        >>> print('bad_species_list = %s' % (bad_viewpoint_list,))
        >>> print(result)

     Ignore:
        bad_viewpoint_list_ = [item for item in bad_viewpoint_list if item[2] is not None and item[0] > 1200]
        grouped_dict = ut.group_items(bad_viewpoint_list, ut.get_list_column(bad_viewpoint_list_, 3))
        grouped_list = grouped_dict.values()
        regrouped_items = ut.flatten(ut.sortedby(grouped_list, map(len, grouped_list)))
        candidate_aid_list = ut.get_list_column(regrouped_items, 0)
        print('candidate_aid_list = %r' % (candidate_aid_list,))
    """
    # Load chips and metadata
    species_list = ibs.get_annot_species(aid_list)
    viewpoint_list = ibs.get_annot_yaw_texts(aid_list)
    species_viewpoint_list = ibs.detect_annot_species_viewpoint_cnn(aid_list)
    # Find all bad
    bad_species_list = []
    bad_viewpoint_list = []
    data = zip(aid_list, species_list, viewpoint_list, species_viewpoint_list)
    for aid, species, viewpoint, (species_, viewpoint_) in data:
        if species != species_:
            bad_species_list.append( (aid, species, species_) )
            continue
        if viewpoint != viewpoint_:
            bad_viewpoint_list.append( (aid, species, viewpoint, viewpoint_) )
            continue
    # Print bad if verbose
    if verbose:
        print('Found conflicting species:')
        for bad_species in bad_species_list:
            print('    AID %4d (%r) should be %r' % bad_species)
        print('Found conflicting viewpoints:')
        for bad_viewpoint in bad_viewpoint_list:
            print('    AID %4d (%r, %r) should be %r' % bad_viewpoint)
    # Return bad
    return bad_species_list, bad_viewpoint_list


def _suggest_random_candidate_regions(ibs, image, min_size, num_candidates=2000):
    h, w, c = image.shape
    h -= 1
    w -= 1
    min_x, min_y = min_size
    def _candidate():
        x0, y0, x1, y1 = 0, 0, 0, 0
        while x1 - x0 < min_x or y1 - y0 < min_y:
            x0 = int(random.uniform(0, w))
            y0 = int(random.uniform(0, h))
            x1 = int(random.uniform(0, w))
            y1 = int(random.uniform(0, h))
            if x0 > x1:
                x0, x1 = x1, x0
            if y0 > y1:
                y0, y1 = y1, y0
        return x0, y0, x1, y1
    candidate_list = [ _candidate() for _ in range(num_candidates) ]
    return candidate_list


def _suggest_bing_candidate_regions(ibs, image_path_list):
    def _dedictify(dict_list):
        return [ [d_['minx'], d_['miny'], d_['maxx'], d_['maxy']] for d_ in dict_list ]

    from pybing import BING_Detector
    detector = BING_Detector()
    results_list = detector.detect(image_path_list)
    result_list = [ _dedictify(results[1]) for results in results_list ]
    return result_list


def non_max_suppression_fast(box_list, conf_list, overlapThresh=0.5):
    """
        Python version of Malisiewicz's Matlab code:
        https://github.com/quantombone/exemplarsvm

        NOTE: This is adapted from Pedro Felzenszwalb's version (nms.m),
        but an inner loop has been eliminated to significantly speed it
        up in the case of a large number of boxes

        Reference: https://github.com/rbgirshick/rcnn/blob/master/nms/nms.m
        Reference: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    """
    # if there are no boxes, return an empty list
    if len(box_list) == 0:
        return []

    # Convert to Numpy
    box_list  = np.array(box_list)
    conf_list = np.array(conf_list)

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if box_list.dtype.kind == "i":
        box_list = box_list.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    # Our boxes are stored as y1, y2, x1, x2 to be in-line with OpenCV indexing
    x1 = box_list[:, 0]
    y1 = box_list[:, 1]
    x2 = box_list[:, 2]
    y2 = box_list[:, 3]
    s  = conf_list

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(s)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return pick


@register_ibs_method
def detect_image_cnn(ibs, gid, confidence=0.90, extraction='bing'):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        gid (?):
        confidence (float): (default = 0.9)
        extraction (str): (default = 'bing')

    CommandLine:
        python -m ibeis_cnn._plugin --exec-detect_image_cnn

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn._plugin import *  # NOQA
        >>> from ibeis_cnn._plugin import _suggest_random_candidate_regions, _suggest_bing_candidate_regions  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> gid = 1
        >>> confidence = 0.9
        >>> extraction = 'bing'
        >>> result = detect_image_cnn(ibs, gid, confidence, extraction)
        >>> print(result)
    """
    # Load chips and resize to the target
    target = (96, 96)
    targetx, targety = target
    # gid = gid_list[random.randint(0, len(gid_list))]
    # gid = gid_list[0]
    print('Detecting with gid=%r...' % (gid, ))
    image = ibs.get_images(gid)
    rects = np.copy(image)
    h, w, c = image.shape

    print('Querrying for candidate regions...')
    image_path = ibs.get_image_paths(gid)
    if extraction == 'random':
        candidate_list = _suggest_random_candidate_regions(ibs, image, (32, 32))
    else:
        candidate_list = _suggest_bing_candidate_regions(ibs, [image_path])[0]

    print('Num candidates: %r' % (len(candidate_list), ))
    chip_list_resized = []
    print('Extracting candidate regions...')
    for candidate in candidate_list:
        x0, y0, x1, y1 = candidate
        chip = image[y0 : y1, x0 : x1]
        chip = cv2.resize(chip, target, interpolation=cv2.INTER_LANCZOS4)
        chip_list_resized.append(chip)
        color = (255, 0, 0)
        # cv2.rectangle(rects, (x0, y0), (x1, y1), color)
        mx = int((x1 - x0) * 0.5)
        my = int((y1 - y0) * 0.5)
        cv2.circle(rects, (x0 + mx, y0 + my), 5, color, -1)
    # cv2.imshow('', rects)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Build data for network
    X_test = np.array(chip_list_resized, dtype=np.uint8)
    y_test = None
    # Define model and load weights
    print('Loading model...')
    from ibeis_cnn import harness
    data_shape = (96, 96, 3)
    # Define model and load weights
    print('Loading model...')
    batch_size = int(min(128, 2 ** np.floor(np.log2(len(chip_list_resized)))))
    model = models.ViewpointModel(batch_size=batch_size, data_shape=data_shape)
    weights_path = grabmodels.ensure_model('viewpoint', redownload=False)
    old_weights_fpath = weights_path
    model.load_old_weights_kw(old_weights_fpath)

    # Predict on the data and convert labels to IBEIS namespace
    test_outputs = harness.test_data2(model, X_test, y_test)
    conf_list = test_outputs['confidences']
    label_list = test_outputs['labeled_predictions']
    pred_list = test_outputs['predictions']
    #pred_list, label_list, conf_list = test.test_data(X_test, y_test, model, weights_path)
    species_viewpoint_list = [ convert_label(label) for label in label_list ]

    num_all_candidates = len(conf_list)
    index_list = non_max_suppression_fast(candidate_list, conf_list)
    print('Surviving candidates: %r' % (index_list, ))
    num_supressed_candidates = num_all_candidates - len(index_list)
    print('Supressed: %d candidates' % (num_supressed_candidates, ))

    candidate_list         = np.take(candidate_list, index_list, axis=0)
    pred_list              = np.take(pred_list, index_list, axis=0)
    species_viewpoint_list = np.take(species_viewpoint_list, index_list, axis=0)
    conf_list              = np.take(conf_list, index_list, axis=0)

    values = zip(candidate_list, pred_list, species_viewpoint_list, conf_list)
    rects = np.copy(image)
    color_dict = {
        'giraffe': (255, 0, 0),
        'giraffe_masai': (255, 255, 0),
        'zebra_plains': (0, 0, 255),
        'zebra_grevys': (0, 255, 0),
        'elephant_savanna': (0, 0, 0),
    }
    skipped = 0
    for candidate, pred, species_viewpoint, conf in values:
        x0, y0, x1, y1 = tuple(candidate)
        species, viewpoint = species_viewpoint
        if conf < confidence:
            skipped += 1
            continue
        print('%r Found %s (%s, %s) at %s' % (candidate, pred, species, viewpoint, conf, ))
        color = color_dict[species]
        cv2.rectangle(rects, (x0, y0), (x1, y1), color)
        # mx = int((x1 - x0) * 0.5)
        # my = int((y1 - y0) * 0.5)
        # cv2.circle(rects, (x0 + mx, y0 + my), 5, color, -1)
    print('Skipped [ %d / %d ]' % (skipped, len(values), ))

    cv2.imshow('', rects)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_siam_l2_model():
    """
    model.show_weights_image()
    """
    model_url = 'https://www.dropbox.com/s/k92s6i5i1hwwy07/siaml2_128_model_state.pkl'
    model_dpath = ut.ensure_app_resource_dir('ibeis_cnn', 'models')
    model_fpath = ut.grab_file_url(model_url, download_dir=model_dpath)
    model_state = ut.load_cPkl(model_fpath)
    import ibeis_cnn
    ibeis_cnn.models
    model = models.SiameseL2(
        input_shape=model_state['input_shape'],
        arch_tag=model_state['arch_tag'], autoinit=True)
    model.load_model_state(fpath=model_fpath)
    return model


def generate_siam_l2_128_feats(ibs, cid_list, config2_=None):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        cid_list (list):
        config2_ (dict): (default = None)

    CommandLine:
        python -m ibeis_cnn._plugin --test-generate_siam_l2_128_feats
        python -m ibeis_cnn._plugin --test-generate_siam_l2_128_feats --db PZ_Master0

    SeeAlso:
        ~/code/ibeis/ibeis/model/preproc/preproc_feat.py

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn._plugin import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> cid_list = ibs.get_annot_chip_rowids(ibs.get_valid_aids())
        >>> config2_ = None
        >>> # megahack
        >>> config2_ = dict(feat_type='hesaff+siam128',
        >>>                 feat_cfgstr=ibs.cfg.feat_cfg.get_cfgstr().replace('sift', 'siam128'),
        >>>                 hesaff_params=ibs.cfg.feat_cfg.get_hesaff_params())
        >>> featgen = generate_siam_l2_128_feats(ibs, cid_list, config2_)
        >>> result = ut.depth_profile(list(featgen))
        >>> print(result)
    """
    #if config2_ is not None:
    #    # Get config from config2_ object
    #    #print('id(config2_) = ' + str(id(config2_)))
    #    feat_cfgstr     = config2_.get('feat_cfgstr')
    #    hesaff_params   = config2_.get('hesaff_params')
    #    assert feat_cfgstr is not None
    #    assert hesaff_params is not None
    #else:
    #    # Get config from IBEIS controller
    #    feat_cfgstr     = ibs.cfg.feat_cfg.get_cfgstr()
    #    hesaff_params   = ibs.cfg.feat_cfg.get_hesaff_params()

    # hack because we need the old features
    import vtool as vt
    import ibeis_cnn
    model = get_siam_l2_model()
    colorspace = 'gray' if model.input_shape[1] else None  # 'bgr'
    patch_size = model.input_shape[-1]
    if config2_ is not None:
        # Get config from config2_ object
        #print('id(config2_) = ' + str(id(config2_)))
        feat_cfgstr     = config2_.get('feat_cfgstr')
        hesaff_params   = config2_.get('hesaff_params')
        assert feat_cfgstr is not None
        assert hesaff_params is not None
    else:
        # Get config from IBEIS controller
        feat_cfgstr     = ibs.cfg.feat_cfg.get_cfgstr()
        hesaff_params   = ibs.cfg.feat_cfg.get_hesaff_params()
    hack_config2_ = dict(feat_type='hesaff+sift',
                         feat_cfgstr=feat_cfgstr.replace('siam128', 'sift'),
                         hesaff_params=hesaff_params)
    print('Generating siam128 features for %d chips' % (len(cid_list),))
    BATCHED = True
    if BATCHED:
        ibs.get_chip_feat_rowid(cid_list, config2_=hack_config2_, ensure=True)
        for cid_batch in ut.ProgressIter(list(ut.ichunks(cid_list, 128)), lbl='siam128 chip chunk'):
            sift_fid_list = ibs.get_chip_feat_rowid(cid_batch, config2_=hack_config2_)
            print('Reading keypoints')
            kpts_list = ibs.get_feat_kpts(sift_fid_list)
            print('Reading chips')
            chip_list = vt.convert_image_list_colorspace(ibs.get_chips(cid_batch, ensure=True), colorspace)
            print('Warping patches')
            warped_patches_list = [vt.get_warped_patches(chip, kpts, patch_size=patch_size)[0]
                                   for chip, kpts in zip(chip_list, kpts_list)]
            flat_list, cumlen_list = ut.invertible_flatten2(warped_patches_list)
            stacked_patches = np.transpose(np.array(flat_list)[None, :], (1, 2, 3, 0))

            test_outputs = ibeis_cnn.harness.test_data2(model, stacked_patches, None)
            network_output_determ = test_outputs['network_output_determ']
            #network_output_determ.min()
            #network_output_determ.max()
            siam128_vecs_list = ut.unflatten2(network_output_determ, cumlen_list)

            for cid, kpts, vecs in zip(cid_batch, kpts_list, siam128_vecs_list):
                yield cid, len(kpts), kpts, vecs
    else:
        sift_fid_list = ibs.get_chip_feat_rowid(cid_list, config2_=hack_config2_, ensure=True)  # NOQA
        print('Reading keypoints')
        kpts_list = ibs.get_feat_kpts(sift_fid_list)
        print('Reading chips')
        chip_list = vt.convert_image_list_colorspace(ibs.get_chips(cid_list, ensure=True), colorspace)
        print('Warping patches')
        warped_patches_list = [vt.get_warped_patches(chip, kpts, patch_size=patch_size)[0]
                               for chip, kpts in zip(chip_list, kpts_list)]
        flat_list, cumlen_list = ut.invertible_flatten2(warped_patches_list)
        stacked_patches = np.transpose(np.array(flat_list)[None, :], (1, 2, 3, 0))

        test_outputs = ibeis_cnn.harness.test_data2(model, stacked_patches, None)
        network_output_determ = test_outputs['network_output_determ']
        #network_output_determ.min()
        #network_output_determ.max()
        siam128_vecs_list = ut.unflatten2(network_output_determ, cumlen_list)

        for cid, kpts, vecs in zip(cid_list, kpts_list, siam128_vecs_list):
            yield cid, len(kpts), kpts, vecs


def extract_siam128_vecs(chip_list, kpts_list):
    """ duplicate testing func for vtool """
    import vtool as vt
    import ibeis_cnn
    model = get_siam_l2_model()
    colorspace = 'gray' if model.input_shape[1] else None  # 'bgr'
    patch_size = model.input_shape[-1]
    chip_list_ = vt.convert_image_list_colorspace(chip_list, colorspace)

    warped_patches_list = [vt.get_warped_patches(chip, kpts, patch_size=patch_size)[0]
                           for chip, kpts in zip(chip_list_, kpts_list)]
    flat_list, cumlen_list = ut.invertible_flatten2(warped_patches_list)
    stacked_patches = np.transpose(np.array(flat_list)[None, :], (1, 2, 3, 0))

    test_outputs = ibeis_cnn.harness.test_data2(model, stacked_patches, None)
    network_output_determ = test_outputs['network_output_determ']
    #network_output_determ.min()
    #network_output_determ.max()
    siam128_vecs_list = ut.unflatten2(network_output_determ, cumlen_list)
    return siam128_vecs_list


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis_cnn._plugin
        python -m ibeis_cnn._plugin --allexamples
        python -m ibeis_cnn._plugin --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
