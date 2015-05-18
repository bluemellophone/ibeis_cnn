from __future__ import absolute_import, division, print_function
import utool as ut
import numpy as np
from ibeis_cnn import utils
print, rrr, profile = ut.inject2(__name__, '[ibeis_cnn.ibsplugin]')


def interact_view_data_fpath_patches(data_fpath, labels_fpath):
    data, labels = utils.load(data_fpath, labels_fpath)
    interact_view_data_patches(labels, data)


def interact_view_data_patches(labels, data):
    warped_patch1_list, warped_patch2_list = list(zip(*ut.ichunks(data, 2)))
    interact_view_patches(labels, warped_patch1_list, warped_patch2_list)
    pass


def interact_view_patches(label_list, warped_patch1_list, warped_patch2_list, aid1_list_=None, aid2_list_=None):
    from ibeis.viz import viz_helpers as vh
    import plottool as pt
    iter_ = list(enumerate(zip(label_list, warped_patch1_list, warped_patch2_list)))
    for count, (label, patch1, patch2) in ut.InteractiveIter(iter_, display_item=False):
        if aid1_list_ is not None:
            aid1 = aid1_list_[count]
            aid2 = aid2_list_[count]
            print('aid1=%r, aid2=%r, label=%r' % (aid1, aid2, label))
        pt.figure(fnum=1, pnum=(1, 2, 1), doclf=True)
        pt.imshow(patch1, pnum=(1, 2, 1))
        pt.draw_border(pt.gca(), color=vh.get_truth_color(label))
        pt.imshow(patch2, pnum=(1, 2, 2))
        pt.draw_border(pt.gca(), color=vh.get_truth_color(label))
        pt.update()


class NewConfigBase(object):
    def update(self, **kwargs):
        self.__dict__.update(**kwargs)

    def kw(self):
        return ut.KwargsWrapper(self)


class PatchMetricDataConfig(NewConfigBase):
    def __init__(pmcfg):
        pmcfg.patch_size = 64

    def get_cfgstr(pmcfg):
        cfgstr_list = [
            'patch_size=%d' % (pmcfg.patch_size,),
        ]
        return ','.join(cfgstr_list)


class IdentifyDataConfig(NewConfigBase):
    def __init__(idcfg):
        idcfg.stacked = False
        idcfg.data_format = 'cv2'
        idcfg.base_size = 64

    def get_cfgstr(idcfg):
        cfgstr_list = [
            'stacked' if idcfg.stacked else 'strided',
            idcfg.data_format,
            'sz=%d' % (idcfg.base_size,),
        ]
        return ','.join(cfgstr_list)


def view_training_data(ibs, **kwargs):
    """
    CommandLine:
        python -m ibeis_cnn.ibsplugin --test-view_training_data --db NNP_Master3

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.ibsplugin import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> view_training_data(ibs)
    """
    data_fpath, labels_fpath, training_dpath = get_identify_training_fpaths(ibs, **kwargs)
    data = np.load(data_fpath, mmap_mode='r')
    #width = height = base_size * 2  # HACK FIXME
    #channels = 3
    #img_list = utils.convert_data_to_imglist(data, width, height, channels)
    import plottool as pt
    for img in ut.InteractiveIter(data, display_item=False):
        pt.imshow(img)
        pt.update()


def get_hotspotter_aid_pairs(ibs):
    aid_list = ibs.get_valid_aids()
    import utool as ut
    aid_list = ut.list_compress(aid_list, ibs.get_annot_has_groundtruth(aid_list))
    qres_list = ibs.query_chips(aid_list, aid_list)
    num_top = 3
    aid1_list = np.array(ut.flatten([[qres.qaid] * num_top for qres in qres_list]))
    aid2_list = np.array(ut.flatten([qres.get_top_aids()[0:num_top] for qres in qres_list]))
    return aid1_list, aid2_list


def filter_aid_pairs(aid1_list, aid2_list):
    """
    TODO: move to results_organizer
    """
    np.vstack((aid1_list, aid2_list)).T
    import vtool as vt
    index_list = vt.find_best_undirected_edge_indexes(np.vstack((aid1_list, aid2_list)).T)
    aid1_list = ut.list_take(aid1_list, index_list)
    aid2_list = ut.list_take(aid2_list, index_list)
    return aid1_list, aid2_list


def compute_target_size_from_aspect(chip_list, target_height=256):
    r"""
    Args:
        chip_list (list):
        target_height (int):

    Returns:
        ?: data

    CommandLine:
        python -m ibeis_cnn.ibsplugin --test-compute_target_size_from_aspect

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.ibsplugin import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> (aid1_list, aid2_list) = get_identify_training_aid_pairs(ibs)
        >>> # execute function
        >>> chip_list = ibs.get_annot_chips(ut.unique_keep_order2(aid1_list + aid2_list))
        >>> target_height = 256
        >>> target_size = compute_target_size_from_aspect(chip_list, target_height)
        >>> # verify results
        >>> result = str(target_size)
        >>> print(result)
        (381, 256)
    """
    import vtool as vt
    sizes1 = np.array([vt.get_size(chip1) for chip1 in chip_list])
    ar1_list = sizes1.T[0] / sizes1.T[1]
    ave_ar = ar1_list.mean()
    target_size = (int(np.round(ave_ar * target_height)), target_height)
    return target_size


def extract_chip_from_gpath_into_square_worker(args):
    import vtool as vt
    gfpath, bbox, theta, target_size = args
    imgBGR = vt.imread(gfpath)  # Read parent image
    return vt.extract_chip_into_square(imgBGR, bbox, theta, target_size)


def extract_square_chips_from_images(ibs, aid_list, target_size):
    """
    Simple function for computing a set of chips without going through the whole ibeis stuff

    Args:
        ibs (IBEISController):  ibeis controller object
        aid_list (int):  list of annotation ids

    CommandLine:
        python -m ibeis_cnn.ibsplugin --test-extract_square_chips_from_images --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.ibsplugin import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> aid_list = ibs.get_valid_aids()[0:40]
        >>> target_size = (220, 220)
        >>> # execute function
        >>> chipBGR_square_list = extract_square_chips_from_images(ibs, aid_list, target_size)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> for chipBGR in ut.InteractiveIter(chipBGR_square_list, display_item=False):
        ...     pt.imshow(chipBGR)
        ...     pt.update()
        >>> # verify results
        >>> print(result)
    """
    gfpath_list = ibs.get_annot_image_paths(aid_list)
    bbox_list   = ibs.get_annot_bboxes(aid_list)
    theta_list  = ibs.get_annot_thetas(aid_list)
    target_size_list = [target_size] * len(aid_list)
    args_iter = zip(gfpath_list, bbox_list, theta_list, target_size_list)
    chipBGR_square_gen = ut.generate(extract_chip_from_gpath_into_square_worker, args_iter)
    chipBGR_square_list = list(chipBGR_square_gen)
    return chipBGR_square_list


def get_aidpair_identify_images(ibs, aid1_list, aid2_list, base_size=64, stacked=False):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        aid1_list (list):
        aid2_list (list):

    Returns:
        tuple: (data, labels)

    CommandLine:
        python -m ibeis_cnn.ibsplugin --test-get_aidpair_identify_images --show
        python -m ibeis_cnn.ibsplugin --test-get_aidpair_identify_images --show --db NNP_Master3
        python -m ibeis_cnn.ibsplugin --test-get_aidpair_identify_images

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.ibsplugin import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> (aid1_list, aid2_list) = get_identify_training_aid_pairs(ibs)
        >>> aid1_list = aid1_list[0:min(100, len(aid1_list))]
        >>> aid2_list = aid2_list[0:min(100, len(aid2_list))]
        >>> # execute function
        >>> thumb1_list, thumb2_list = get_aidpair_identify_images(ibs, aid1_list, aid2_list)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> for aid1, aid2, thumb1, thumb2 in ut.InteractiveIter(zip(aid1_list, aid2_list, thumb1_list, thumb2_list), display_item=False):
        ...     print('aid1=%r, aid2=%r' % (aid1, aid2))
        ...     pt.imshow(thumb1, pnum=(1, 2, 1))
        ...     pt.imshow(thumb2, pnum=(1, 2, 2))
        ...     pt.update()
        >>> ut.show_if_requested()
    """
    # TODO: Cache this with visual uuids

    import vtool as vt
    print('[ibeget_aidpair_identify_images len(aid1_list)=%r' % (len(aid1_list)))
    print('base_size = %r' % (base_size,))
    assert len(aid1_list) == len(aid2_list)

    # Flatten to only apply chip operations once
    flat_unique, reconstruct_tup = ut.inverable_unique_two_lists(aid1_list, aid2_list)

    print('get_chips')
    chip_list = ibs.get_annot_chips(flat_unique)
    print('resize chips: base_size = %r' % (base_size,))
    target_height = base_size
    AUTO_SIZE = False
    if AUTO_SIZE:
        target_size = compute_target_size_from_aspect(chip_list, target_height)
    else:
        if stacked:
            target_size = (2 * base_size, base_size)
        else:
            target_size = (base_size, base_size)

    thumb_list = [vt.padded_resize(chip, target_size) for chip in ut.ProgressIter(chip_list)]

    thumb1_list, thumb2_list = ut.uninvert_unique_two_lists(thumb_list, reconstruct_tup)

    return thumb1_list, thumb2_list


#def get_unique_featureids(aid1_list, aid2_list, fm_list):
#    import vtool as vt
#    vt.group_indices(np.array(aid1_list))
#    fx1_list = [fm.T[0] for fm in fm_list]
#    fx2_list = [fm.T[1] for fm in fm_list]
#    #aidfx1_list = [zip([aid1] * len(fx1), fx1) for aid1, fx1 in zip(aid1_list, fx1_list)]
#    #aidfx2_list = [zip([aid2] * len(fx2), fx2) for aid2, fx2 in zip(aid2_list, fx2_list)]
#    #iddict_ = {}
#    #unique_feat_ids1 = [vt.compute_unique_data_ids_(rows, iddict_=iddict_) for rows in aidfx1_list]
#    #unique_feat_ids2 = [vt.compute_unique_data_ids_(rows, iddict_=iddict_) for rows in aidfx2_list]
#    #unique_feat_ids = ut.flatten(unique_feat_ids1) + ut.flatten(unique_feat_ids2)
#    #return unique_feat_ids

#    #


def get_aidpair_patchmatch_training_data(ibs, aid1_list, aid2_list, kpts1_m_list, kpts2_m_list, fm_list, patch_size):
    """
    CommandLine:
        python -m ibeis_cnn.ibsplugin --test-get_aidpair_patchmatch_training_data --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.ibsplugin import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb(defaultdb='PZ_MTEST')
        >>> (aid1_list, aid2_list, kpts1_m_list, kpts2_m_list, fm_list) = get_aidpairs_and_matches(ibs, 10)
        >>> # execute function
        >>> tup = get_aidpair_patchmatch_training_data(ibs, aid1_list, aid2_list, kpts1_m_list, kpts2_m_list, fm_list)
        >>> aid1_list_, aid2_list_, warped_patch1_list, warped_patch2_list = tup
        >>> ut.quit_if_noshow()
        >>> label_list = get_aidpair_training_labels(ibs, aid1_list_, aid2_list_)
        >>> interact_view_patches(label_list, warped_patch1_list, warped_patch2_list, aid1_list_, aid2_list_)
        >>> ut.show_if_requested()
    """
    # Flatten to only apply chip operations once
    print('get_aidpair_patchmatch_training_data num_pairs = %r' % (len(aid1_list)))
    assert len(aid1_list) == len(aid2_list)
    assert len(aid1_list) == len(kpts1_m_list)
    assert len(aid1_list) == len(kpts2_m_list)
    assert len(aid1_list) == len(fm_list)
    print('geting_unflat_chips')
    flat_unique, reconstruct_tup = ut.inverable_unique_two_lists(aid1_list, aid2_list)
    print('grabbing %d unique chips' % (len(flat_unique)))
    chip_list = ibs.get_annot_chips(flat_unique)
    #if False:
    #    aid_multilist = (np.array(aid1_list), np.array(aid2_list))
    #    flat_unique_, flat_groupx_multilist = ut.inverable_group_multi_list()
    #    fx1_list = [fm.T[0] for fm in fm_list]
    #    fx2_list = [fm.T[1] for fm in fm_list]
    #    fx_multilist = (fx1_list, fx2_list)
    #    fx_groups = [[ut.list_take(fx_multilist[listx], groupx) for listx, groupx in enumerate(groupx_list)] for groupx_list in flat_groupx_multilist]

    #    for fxgrp in fx_groups:
    #        flat_list1, cumsum1 = ut.invertible_flatten2(fxgrp)
    #        flat_list2, cumsum2 = ut.invertible_flatten2(flat_list1)
    #        unique_fx, invert_fx3 = np.unique(flat_list2, return_inverse=True)
    #        pass
    #    #[[ut.list_take(aid_multilist[listx], groupx) for listx, groupx in enumerate(groupx_list)] for groupx_list in flat_groupx_multilist]

    ut.print_object_size(chip_list, 'chip_list')
    chip1_list, chip2_list = ut.uninvert_unique_two_lists(chip_list, reconstruct_tup)
    import vtool as vt
    import itertools
    print('warping')
    # TODO: Finishme
    #NODUP_KEYPOINT_WARP = True
    #if NODUP_KEYPOINT_WARP:
    #    # Prevent duplicate keypoint warping computations
    #    fx1_list = [fm.T[0] for fm in fm_list]
    #    fx2_list = [fm.T[1] for fm in fm_list]
    #    aidfx1_list = [zip([aid1] * len(fx1), fx1) for aid1, fx1 in zip(aid1_list, fx1_list)]
    #    aidfx2_list = [zip([aid2] * len(fx2), fx2) for aid2, fx2 in zip(aid2_list, fx2_list)]
    #    iddict_ = {}
    #    unique_feat_ids1 = [vt.compute_unique_data_ids_(rows, iddict_=iddict_) for rows in aidfx1_list]
    #    unique_feat_ids2 = [vt.compute_unique_data_ids_(rows, iddict_=iddict_) for rows in aidfx2_list]
    #    unique_feat_ids = ut.flatten(unique_feat_ids1) + ut.flatten(unique_feat_ids2)
    #    print(len(set(unique_feat_ids)) / len(unique_feat_ids))

    USE_CACHEFUNC = True

    if USE_CACHEFUNC:
        def idcache_find_misses(id_list, cache_):
            # Generalize?
            val_list = [cache_.get(id_, None) for id_ in id_list]
            ismiss_list = [val is None for val in val_list]
            return val_list, ismiss_list

        def idcache_save(ismiss_list, miss_vals, id_list, val_list, cache_):
            # Generalize?
            miss_indices = ut.list_where(ismiss_list)
            miss_ids  = ut.filter_items(id_list, ismiss_list)
            # overwrite missed output
            for index, val in zip(miss_indices, miss_vals):
                val_list[index] = val
            # cache save
            for id_, val in zip(miss_ids, miss_vals):
                cache_[id_] = val

        def cacheget_wraped_patches(aid, fxs, chip, kpts, cache_={}):
            # +-- Custom ids
            id_list = [(aid, fx) for fx in fxs]
            # L__
            val_list, ismiss_list = idcache_find_misses(id_list, cache_)
            if any(ismiss_list):
                # +-- Custom evaluate misses
                kpts_miss = kpts.compress(ismiss_list, axis=0)
                miss_vals = vt.get_warped_patches(chip, kpts_miss, patch_size=patch_size)[0]
                # L__
                idcache_save(ismiss_list, miss_vals, id_list, val_list, cache_)
            return val_list
            #patch_list = []
            #for fx, kp in zip(fxs, kpts):
            #    try:
            #        patch = cache_[(aid, fxs)]
            #    except KeyError:
            #        patch = vt.get_warped_patches(chip, kpts)[0]
            #        cache_[(aid, fx)] = patch
            #    patch_list.append(patch)
            #return patch_list
        fx1_list = [fm.T[0] for fm in fm_list]
        fx2_list = [fm.T[1] for fm in fm_list]
        warp_iter1 = ut.ProgressIter(zip(aid1_list, fx1_list, chip1_list, kpts1_m_list), nTotal=len(kpts1_m_list), lbl='warp1')
        warp_iter2 = ut.ProgressIter(zip(aid2_list, fx2_list, chip2_list, kpts2_m_list), nTotal=len(kpts2_m_list), lbl='warp2')
        warped_patches1_list = list(itertools.starmap(cacheget_wraped_patches, warp_iter1))
        warped_patches2_list = list(itertools.starmap(cacheget_wraped_patches, warp_iter2))
    else:
        warp_iter1 = ut.ProgressIter(zip(chip1_list, kpts1_m_list), nTotal=len(kpts1_m_list), lbl='warp1')
        warp_iter2 = ut.ProgressIter(zip(chip2_list, kpts2_m_list), nTotal=len(kpts2_m_list), lbl='warp2')
        warped_patches1_list = [vt.get_warped_patches(chip1, kpts1, patch_size=patch_size)[0] for chip1, kpts1 in warp_iter1]
        warped_patches2_list = [vt.get_warped_patches(chip2, kpts2, patch_size=patch_size)[0] for chip2, kpts2 in warp_iter2]

    del chip_list
    del chip1_list
    del chip2_list
    ut.print_object_size(warped_patches1_list, 'warped_patches1_list')
    ut.print_object_size(warped_patches2_list, 'warped_patches2_list')
    len1_list = list(map(len, warped_patches1_list))
    assert len1_list == list(map(len, warped_patches2_list))
    print('flattening')
    aid1_list_ = ut.flatten([[aid1] * len1 for len1, aid1 in zip(len1_list, aid1_list)])
    aid2_list_ = ut.flatten([[aid2] * len1 for len1, aid2 in zip(len1_list, aid2_list)])
    warped_patch1_list = ut.flatten(warped_patches1_list)
    del warped_patches1_list
    warped_patch2_list = ut.flatten(warped_patches2_list)
    del warped_patches2_list
    return aid1_list_, aid2_list_, warped_patch1_list, warped_patch2_list


def get_aidpair_training_data_and_labels(ibs, aid1_list, aid2_list, base_size=64, stacked=False, data_format='cv2'):
    thumb1_list, thumb2_list = get_aidpair_identify_images(ibs, aid1_list, aid2_list, base_size=base_size, stacked=stacked)
    assert data_format == 'cv2'
    # Stacking these might not be the exact correct thing to do.
    if stacked:
        img_list = [
            np.vstack((thumb1, thumb2)) for thumb1, thumb2, in
            zip(thumb1_list, thumb2_list)
        ]
    else:
        # Strided data comes in in pairs of two
        img_list = ut.flatten(list(zip(thumb1_list, thumb2_list)))
    data = np.array(img_list)
    del img_list
    #data = [img[None, :] for img in im
    #data = utils.convert_imagelist_to_data(img_list)
    labels = get_aidpair_training_labels(ibs, aid1_list, aid2_list)
    assert labels.shape[0] == data.shape[0] // 2
    from ibeis import const
    assert np.all(labels != const.TRUTH_UNKNOWN)
    return data, labels


def get_patchmetric_training_data_and_labels(ibs, aid1_list, aid2_list, kpts1_m_list, kpts2_m_list, fm_list, patch_size):
    """
    Notes:
        # FIXME: THERE ARE INCORRECT CORRESPONDENCES LABELED AS CORRECT THAT
        # NEED MANUAL CORRECTION EITHER THROUGH EXPLICIT LABLEING OR
        # SEGMENTATION MASKS

    CommandLine:
        python -m ibeis_cnn.ibsplugin --test-get_patchmetric_training_data_and_labels --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.ibsplugin import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb(defaultdb='PZ_MTEST')
        >>> (aid1_list, aid2_list, kpts1_m_list, kpts2_m_list, fm_list) = get_aidpairs_and_matches(ibs, 10)
        >>> patch_size = 64
        >>> data, labels = get_patchmetric_training_data_and_labels(ibs, aid1_list, aid2_list, kpts1_m_list, kpts2_m_list, fm_list, patch_size)
        >>> ut.quit_if_noshow()
        >>> interact_view_data_patches(labels, data)
        >>> ut.show_if_requested()
    """
    # To the removal of unknown pairs before computing the data

    aid1_list_, aid2_list_, warped_patch1_list, warped_patch2_list = get_aidpair_patchmatch_training_data(
        ibs, aid1_list, aid2_list, kpts1_m_list, kpts2_m_list, fm_list, patch_size)
    labels = get_aidpair_training_labels(ibs, aid1_list_, aid2_list_)
    img_list = ut.flatten(list(zip(warped_patch1_list, warped_patch2_list)))
    data = np.array(img_list)
    del img_list
    #data_per_label = 2
    assert labels.shape[0] == data.shape[0] // 2
    from ibeis import const
    assert np.all(labels != const.TRUTH_UNKNOWN)
    return data, labels


def mark_inconsistent_viewpoints(ibs, aid1_list, aid2_list):
    import vtool as vt
    yaw1_list = np.array(ut.replace_nones(ibs.get_annot_yaws(aid2_list), np.nan))
    yaw2_list = np.array(ut.replace_nones(ibs.get_annot_yaws(aid1_list), np.nan))
    yawdist_list = vt.ori_distance(yaw1_list, yaw2_list)
    TAU = np.pi * 2
    isinconsistent_list = yawdist_list > TAU / 8
    return isinconsistent_list


def get_aidpair_training_labels(ibs, aid1_list, aid2_list):
    """
    Returns:
        ndarray: true in positions of matching, and false in positions of not matching

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.ibsplugin import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb(defaultdb='PZ_MTEST')
        >>> (aid1_list, aid2_list) = get_identify_training_aid_pairs(ibs)
        >>> aid1_list = aid1_list[0:min(100, len(aid1_list))]
        >>> aid2_list = aid2_list[0:min(100, len(aid2_list))]
        >>> # execute function
        >>> labels = get_aidpair_training_labels(ibs, aid1_list, aid2_list)
    """
    from ibeis import const
    truth_list = ibs.get_aidpair_truths(aid1_list, aid2_list)
    labels = truth_list
    # Mark different viewpoints as unknown for training
    isinconsistent_list = mark_inconsistent_viewpoints(ibs, aid1_list, aid2_list)
    labels[isinconsistent_list] = const.TRUTH_UNKNOWN
    return labels


def get_semantic_trainingpair_dir(ibs, aid1_list, aid2_list, lbl):
    nets_dir = ibs.get_neuralnet_dir()
    training_dname = get_semantic_trainingpair_dname(ibs, aid1_list, aid2_list, lbl)
    training_dpath = ut.unixjoin(nets_dir, training_dname)
    ut.ensuredir(nets_dir)
    ut.ensuredir(training_dpath)
    view_train_dir = ut.get_argflag('--vtd')
    if view_train_dir:
        ut.view_directory(training_dpath)
    return training_dpath


def get_semantic_trainingpair_dname(ibs, aid1_list, aid2_list, lbl):
    """
    Args:
        ibs (IBEISController):  ibeis controller object
        aid1_list (list):
        aid2_list (list):

    Returns:
        tuple: (data_fpath, labels_fpath)

    CommandLine:
        python -m ibeis_cnn.ibsplugin --test-get_identify_training_dname

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.ibsplugin import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid1_list = ibs.get_valid_aids()
        >>> aid2_list = ibs.get_valid_aids()
        >>> training_dpath = get_identify_training_dname(ibs, aid1_list, aid2_list, 'training')
        >>> # verify results
        >>> result = str(training_dpath)
        >>> print(result)
        training((13)e%dn&kgqfibw7dbu)
    """
    semantic_uuids1 = ibs.get_annot_semantic_uuids(aid1_list)
    semantic_uuids2 = ibs.get_annot_semantic_uuids(aid2_list)
    aidpair_hashstr_list = map(ut.hashstr, zip(semantic_uuids1, semantic_uuids2))
    training_dname = ut.hashstr_arr(aidpair_hashstr_list, hashlen=16, lbl=lbl)
    return training_dname


def cached_identify_training_data_fpaths(ibs, aid1_list, aid2_list, base_size=64):
    """
    todo use size in cfgstrings
    """
    print('[cached_identify_training_data_fpaths] begin')
    training_dpath = get_semantic_trainingpair_dir(ibs, aid1_list, aid2_list, 'train_identity')

    idcfg = IdentifyDataConfig()
    idcfg.base_size = base_size

    data_fpath = ut.unixjoin(training_dpath, 'data_' + idcfg.get_cfgstr())
    labels_fpath = ut.unixjoin(training_dpath, 'labels.pkl')
    NOCACHE_TRAIN = ut.get_argflag('--nocache-train')

    if NOCACHE_TRAIN or not (
            ut.checkpath(data_fpath, verbose=True)
            and ut.checkpath(labels_fpath, verbose=True)
    ):
        data, labels = get_aidpair_training_data_and_labels(ibs, aid1_list, aid2_list, **idcfg.kw())
        utils.write_data_and_labels(data, labels, data_fpath, labels_fpath)
    else:
        print('data and labels cache hit')
    return data_fpath, labels_fpath, training_dpath


def cached_patchmetric_training_data_fpaths(ibs, aid1_list, aid2_list, kpts1_m_list, kpts2_m_list, fm_list):
    """
    todo use size in cfgstrings
    """
    training_dpath = get_semantic_trainingpair_dir(ibs, aid1_list, aid2_list, 'train_patchmetric')

    cfg = PatchMetricDataConfig()

    fm_hashstr = ut.hashstr_arr(fm_list, lbl='fm')
    data_fpath = ut.unixjoin(training_dpath, 'data_' + fm_hashstr + '_' + cfg.get_cfgstr() + '.pkl')
    labels_fpath = ut.unixjoin(training_dpath, 'labels.pkl')
    NOCACHE_TRAIN = ut.get_argflag('--nocache-train')

    if NOCACHE_TRAIN or not (
            ut.checkpath(data_fpath, verbose=True)
            and ut.checkpath(labels_fpath, verbose=True)
    ):
        data, labels = get_patchmetric_training_data_and_labels(ibs, aid1_list, aid2_list, kpts1_m_list, kpts2_m_list, fm_list, **cfg.kw())
        ut.assert_eq(data.shape[1], cfg.patch_size)
        ut.assert_eq(data.shape[2], cfg.patch_size)
        utils.write_data_and_labels(data, labels, data_fpath, labels_fpath)
    else:
        print('data and labels cache hit')
    return data_fpath, labels_fpath, training_dpath


def get_identify_training_aid_pairs(ibs):
    r"""
    Returns:
        tuple: (aid1_list, aid2_list)

    CommandLine:
        python -m ibeis_cnn.ibsplugin --test-get_identify_training_aid_pairs

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.ibsplugin import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('NNP_Master3')
        >>> (aid1_list, aid2_list) = get_identify_training_aid_pairs(ibs)
        >>> result = str((len(aid1_list), len(aid2_list)))
        >>> assert aid1_list != aid2_list
        >>> print(result)
    """
    verified_aid1_list, verified_aid2_list = ibs.get_verified_aid_pairs()
    if len(verified_aid1_list) > 100:
        aid1_list = verified_aid1_list
        aid2_list = verified_aid2_list
    else:
        aid1_list, aid2_list = get_hotspotter_aid_pairs(ibs)
    aid1_list, aid2_list = filter_aid_pairs(aid1_list, aid2_list)
    return aid1_list, aid2_list


def remove_unknown_training_pairs(ibs, aid1_list, aid2_list):
    return aid1_list, aid2_list


def get_aidpairs_and_matches(ibs, max_examples=None):
    """
    Returns:
        aid pairs and matching keypoint pairs as well as the original index of the feature matches

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.ibsplugin import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb(defaultdb='PZ_MTEST')
        >>> max_examples = None

    """
    aid_list = ibs.get_valid_aids()
    if max_examples is not None:
        aid_list = aid_list[0:min(max_examples, len(aid_list))]
    import utool as ut
    #from ibeis.model.hots import chip_match
    aid_list = ut.list_compress(aid_list, ibs.get_annot_has_groundtruth(aid_list))
    qres_list, qreq_ = ibs.query_chips(aid_list, aid_list, return_request=True)
    # TODO: Use ChipMatch2 instead of QueryResult
    #cm_list = [chip_match.ChipMatch2.from_qres(qres) for qres in qres_list]
    #for cm in cm_list:
    #    cm.evaluate_nsum_score(qreq_=qreq_)
    #aids1_list = [[cm.qaid] * num_top for cm in cm_list]
    #aids2_list = [[cm.qaid] * num_top for cm in cm_list]

    # Get aid pairs and feature matches
    num_top = 3
    aids2_list = [qres.get_top_aids()[0:num_top] for qres in qres_list]
    aids1_list = [[qres.qaid] * len(aids2) for qres, aids2 in zip(qres_list, aids2_list)]
    fms_list = [ut.dict_take(qres.aid2_fm, aids2) for qres, aids2 in zip(qres_list, aids2_list)]
    aid1_list = np.array(ut.flatten(aids1_list))
    aid2_list = np.array(ut.flatten(aids2_list))
    fm_list = ut.flatten(fms_list)

    # Filter out bad training examples
    labels = get_aidpair_training_labels(ibs, aid1_list, aid2_list)
    from ibeis import const
    isvalid = (labels != const.TRUTH_UNKNOWN)
    aid1_list = ut.list_compress(aid1_list, isvalid)
    aid2_list = ut.list_compress(aid2_list, isvalid)
    fm_list   = ut.list_compress(fm_list, isvalid)
    labels    = ut.list_compress(labels, isvalid)

    EQUALIZE_LABELS = True
    #def expand_patchmatch_aids(aid1_list, aid2_list, labels, fm_list):
    if EQUALIZE_LABELS:
        #def equalize_labels(fm_list,
        import vtool as vt
        import six
        print('flattening')
        #aid1_list_ = ut.flatten([[aid1] * len1 for len1, aid1 in zip(len1_list, aid1_list)])
        #aid2_list_ = ut.flatten([[aid2] * len1 for len1, aid2 in zip(len1_list, aid2_list)])
        len1_list = list(map(len, fm_list))
        labels_ = ut.flatten([[label] * len1 for len1, label in zip(len1_list, labels)])
        labelhist = {key: len(val) for key, val in six.iteritems(ut.group_items(labels_, labels_))}
        print('[ibsplugin] original label histogram = \n' + ut.dict_str(labelhist))
        labels_ = np.array(labels_)

        allowed_ratio = ut.PHI
        true_max_ = max(labelhist.values())
        true_min_ = min(labelhist.values())
        min_ = min(true_min_ * allowed_ratio, true_max_)
        #min_ = 500
        keep_indicies_list = []
        for key, val in six.iteritems(labelhist):
            # Be stupid and grab the first few labels of each type
            # should grab some optimal set based on feature scores or something
            # either that or random sample
            type_indicies = np.where(labels_ == key)[0]
            size = min(min_, len(type_indicies))
            if size == len(type_indicies):
                keep_indicies = type_indicies
            else:
                randstate = np.random.RandomState(0)
                keep_indicies = randstate.choice(type_indicies, size=min_, replace=False)
            keep_indicies_list.append(keep_indicies)
        flag_list = np.zeros(len(labels_), dtype=np.bool)
        flag_list[np.hstack(keep_indicies_list)] = True

        #fm_flat, cumsum = ut.invertible_flatten2_numpy(fm_list)
        flags_list = ut.unflatten2(flag_list, np.cumsum(len1_list))
        fm_list = vt.zipcompress(fm_list, flags_list, axis=0)

        # remove empty aids
        isnonempty_list = [len(fm) != 0 for fm in fm_list]
        fm_list = ut.list_compress(fm_list, isnonempty_list)
        aid1_list = ut.list_compress(aid1_list, isnonempty_list)
        aid2_list = ut.list_compress(aid2_list, isnonempty_list)
        labels    = ut.list_compress(labels, isnonempty_list)

        # PRINT NEW LABEL STATS
        len1_list = list(map(len, fm_list))
        labels_ = ut.flatten([[label] * len1 for len1, label in zip(len1_list, labels)])
        labelhist = {key: len(val) for key, val in six.iteritems(ut.group_items(labels_, labels_))}
        print('[ibsplugin] equalized label histogram = \n' + ut.dict_str(labelhist))

    fx1_list = [fm.T[0] for fm in fm_list]
    fx2_list = [fm.T[1] for fm in fm_list]
    # check for any duplicates
    #if False:
    #    aidfx1_list = [zip([aid1] * len(fx1), fx1) for aid1, fx1 in zip(aid1_list, fx1_list)]
    #    aidfx2_list = [zip([aid2] * len(fx2), fx2) for aid2, fx2 in zip(aid2_list, fx2_list)]
    #    aidfx_list = ut.flatten(aidfx1_list) + ut.flatten(aidfx2_list)
    #    import vtool as vt
    #    unique_ids = vt.compute_unique_data_ids_(aidfx_list)
    #    print(len(set(unique_ids)) / len(unique_ids))

    kpts1_list = ibs.get_annot_kpts(aid1_list)
    kpts2_list = ibs.get_annot_kpts(aid2_list)
    kpts1_m_list = [kpts1.take(fx1, axis=0) for kpts1, fx1 in zip(kpts1_list, fx1_list)]
    kpts2_m_list = [kpts2.take(fx2, axis=0) for kpts2, fx2 in zip(kpts2_list, fx2_list)]
    return aid1_list, aid2_list, kpts1_m_list, kpts2_m_list, fm_list


def get_identify_training_fpaths(ibs, **kwargs):
    """
    Notes:
        Blog post: http://benanne.github.io/2015/03/17/plankton.html
        Code: https://github.com/benanne/kaggle-ndsb
        You need to do something like this to pass two images through the network:
        https://github.com/benanne/kaggle-ndsb/blob/master/dihedral.py#L89
        And then something like this to combine them again:
        https://github.com/benanne/kaggle-ndsb/blob/master/dihedral.py#L224

    CommandLine:
        python -m ibeis_cnn.ibsplugin --test-get_identify_training_fpaths

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.ibsplugin import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> (data_fpath, labels_fpath) = get_identify_training_fpaths(ibs)
    """
    print('get_identify_training_fpaths')
    aid1_list, aid2_list = get_identify_training_aid_pairs(ibs)
    max_examples = kwargs.pop('max_examples', None)
    if max_examples is not None:
        print('max_examples = %r' % (max_examples,))
        aid1_list = aid1_list[0:min(max_examples, len(aid1_list))]
        aid2_list = aid2_list[0:min(max_examples, len(aid2_list))]
    data_fpath, labels_fpath, training_dpath = cached_identify_training_data_fpaths(
        ibs, aid1_list, aid2_list, **kwargs)
    return data_fpath, labels_fpath, training_dpath


def get_patchmetric_training_fpaths(ibs, **kwargs):
    """
    CommandLine:
        python -m ibeis_cnn.ibsplugin --test-get_patchmetric_training_fpaths --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.ibsplugin import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> kwargs = {}
        >>> (data_fpath, labels_fpath, training_dpath) = get_patchmetric_training_fpaths(ibs, **kwargs)
        >>> ut.quit_if_noshow()
        >>> interact_view_data_fpath_patches(data_fpath, labels_fpath)
    """
    print('\n\n[get_patchmetric_training_fpaths] START')
    max_examples = kwargs.get('max_examples', None)
    aid1_list, aid2_list, kpts1_m_list, kpts2_m_list, fm_list = get_aidpairs_and_matches(ibs, max_examples)
    data_fpath, labels_fpath, training_dpath = cached_patchmetric_training_data_fpaths(
        ibs, aid1_list, aid2_list, kpts1_m_list, kpts2_m_list, fm_list)
    print('\n[get_patchmetric_training_fpaths] FINISH\n\n')
    return data_fpath, labels_fpath, training_dpath


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis_cnn.ibsplugin
        python -m ibeis_cnn.ibsplugin --allexamples
        python -m ibeis_cnn.ibsplugin --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
