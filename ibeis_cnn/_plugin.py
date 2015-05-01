#!/usr/bin/env python
"""
tests a test set of data using a specified, pre0trained model and weights
"""
from __future__ import absolute_import, division, print_function
from ibeis.control.controller_inject import make_ibs_register_decorator
from ibeis.constants import Species, VIEWTEXT_TO_YAW_RADIANS
# from ibeis_cnn import utils
from ibeis_cnn import models
from ibeis_cnn import test
from ibeis_cnn import _plugin_grabmodels as grabmodels
import utool as ut
import cv2
import numpy as np
import random
print, print_, printDBG, rrr, profile = ut.inject(__name__, '[ibeis_cnn._plugin]')


CLASS_INJECT_KEY, register_ibs_method = make_ibs_register_decorator(__name__)


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
    nets_dir = ut.unixjoin(ibs.get_cachedir(), 'nets')
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
def detect_annot_species_viewpoint_cnn(ibs, aid_list):
    # Load chips and resize to the target
    print('detect_annot_species_viewpoint_cnn')
    target = (96, 96)
    chip_list = ibs.get_annot_chips(aid_list)
    print('resizing chips')
    chip_list_resized = [ cv2.resize(chip, target, interpolation=cv2.INTER_LANCZOS4) for chip in ut.ProgressIter(chip_list, lbl='resizing chips') ]
    # Build data for network
    X_test = np.array(chip_list_resized, dtype=np.uint8)
    y_test = None
    # Define model and load weights
    model = models.ViewpointModel()
    weights_path = grabmodels.ensure_model('viewpoint', redownload=False)
    # Predict on the data and convert labels to IBEIS namespace
    pred_list, label_list, conf_list = test.test_data(X_test, y_test, model, weights_path)
    species_viewpoint_list = [ convert_label(label) for label in label_list ]
    return species_viewpoint_list


@register_ibs_method
def validate_annot_species_viewpoint_cnn(ibs, aid_list, verbose=False):
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


def _suggest_candidate_regions(ibs, image, min_size, num_candidates=1000):
    h, w, c = image.shape
    h -= 1
    w -= 1
    min_x, min_y = min_size
    def _candidate():
        x0 = int(random.uniform(0,  w - min_x))
        y0 = int(random.uniform(0,  h - min_y))
        x1 = int(random.uniform(x0 + min_x, w))
        y1 = int(random.uniform(y0 + min_y, h))
        if random.uniform(0.0, 1.0) < 0.5:
            x0, x1 = w - x1, w - x0
            y0, y1 = h - y1, h - y0
        return y0, y1, x0, x1
    candidate_list = [ _candidate() for _ in range(num_candidates) ]
    return candidate_list


@register_ibs_method
def detect_image_cnn(ibs, gid_list):
    # Load chips and resize to the target
    print('detect_image_cnn')
    target = (96, 96)
    targetx, targety = target
    gid = gid_list[0]
    print('Detecting with gid=%r' % (gid, ))
    image = ibs.get_images(gid)
    rects = np.copy(image)
    h, w, c = image.shape
    candidate_list = _suggest_candidate_regions(ibs, image, target)
    chip_list_resized = []
    for candidate in candidate_list:
        y0, y1, x0, x1 = candidate
        chip = image[y0 : y1, x0 : x1]
        chip = cv2.resize(chip, target, interpolation=cv2.INTER_LANCZOS4)
        chip_list_resized.append(chip)
        cv2.rectangle(rects, (x0, y0), (x1, y1), (255, 0, 0))
    cv2.imshow('', rects)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Build data for network
    X_test = np.array(chip_list_resized, dtype=np.uint8)
    y_test = None
    # Define model and load weights
    model = models.ViewpointModel()
    weights_path = grabmodels.ensure_model('viewpoint', redownload=False)
    # Predict on the data and convert labels to IBEIS namespace
    pred_list, label_list, conf_list = test.test_data(X_test, y_test, model, weights_path)
    species_viewpoint_list = [ convert_label(label) for label in label_list ]
    print(species_viewpoint_list)
