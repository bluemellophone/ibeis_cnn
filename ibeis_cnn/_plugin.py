#!/usr/bin/env python
"""
tests a test set of data using a specified, pre0trained model and weights
"""
from __future__ import absolute_import, division, print_function
from ibeis.control.controller_inject import make_ibs_register_decorator
# from ibeis_cnn import utils
from ibeis_cnn import models
from ibeis_cnn import test
from ibeis_cnn import _plugin_grabmodels as grabmodels
import utool as ut
from os.path import join
import cv2
import numpy as np
print, print_, printDBG, rrr, profile = ut.inject(__name__, '[ibeis_cnn._plugin]')


CLASS_INJECT_KEY, register_ibs_method = make_ibs_register_decorator(__name__)


@register_ibs_method
def detect_annot_viewpoint_cnn(ibs, aid_list):
    target = (96, 96)
    chip_list = ibs.get_annot_chips(aid_list)
    chip_list_resized = [ cv2.resize(chip, target, interpolation=cv2.INTER_LANCZOS4) for chip in chip_list ]
    X_test = np.array(chip_list_resized, dtype=np.uint8)
    y_test = None
    model = models.PZ_GIRM_LARGE_Model()

    model_path = grabmodels.ensure_model('viewpoint')
    weights_path = join(model_path, 'viewpoint', 'ibeis_cnn_weights.pickle')

    all_predict, labels = test.test_data(X_test, y_test, model, weights_path)
    print(all_predict)
    print(labels)
