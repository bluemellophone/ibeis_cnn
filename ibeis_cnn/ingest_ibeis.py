# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import utool as ut
import numpy as np
import six
import cv2
#from ibeis_cnn import utils
from ibeis_cnn import draw_results  # NOQA
print, rrr, profile = ut.inject2(__name__, '[ibeis_cnn.ingest_ibeis]')


FIX_HASH = True


def get_aidpair_patchmatch_training_data(ibs, aid1_list, aid2_list,
                                         kpts1_m_list, kpts2_m_list, fm_list,
                                         metadata_lists, patch_size,
                                         colorspace):
    """
    FIXME: errors on get_aidpairs_and_matches(ibs, 1)

    CommandLine:
        python -m ibeis_cnn.ingest_ibeis --test-get_aidpair_patchmatch_training_data --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.ingest_ibeis import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb(defaultdb='PZ_MTEST')
        >>> tup = get_aidpairs_and_matches(ibs, 6)
        >>> (aid1_list, aid2_list, kpts1_m_list, kpts2_m_list, fm_list, metadata_lists) = tup
        >>> pmcfg = PatchMetricDataConfig()
        >>> patch_size = pmcfg.patch_size
        >>> colorspace = pmcfg.colorspace
        >>> # execute function
        >>> tup = get_aidpair_patchmatch_training_data(ibs, aid1_list,
        ...     aid2_list, kpts1_m_list, kpts2_m_list, fm_list, metadata_lists,
        ...     patch_size, colorspace)
        >>> aid1_list_, aid2_list_, warped_patch1_list, warped_patch2_list, flat_metadata = tup
        >>> ut.quit_if_noshow()
        >>> label_list = get_aidpair_training_labels(ibs, aid1_list_, aid2_list_)
        >>> draw_results.interact_view_patches(label_list, warped_patch1_list, warped_patch2_list, flat_metadata)
        >>> ut.show_if_requested()
    """
    import vtool as vt
    import itertools
    # Flatten to only apply chip operations once
    print('get_aidpair_patchmatch_training_data num_pairs = %r' % (len(aid1_list)))
    assert len(aid1_list) == len(aid2_list)
    assert len(aid1_list) == len(kpts1_m_list)
    assert len(aid1_list) == len(kpts2_m_list)
    assert len(aid1_list) == len(fm_list)
    print('geting_unflat_chips')
    flat_unique, reconstruct_tup = ut.inverable_unique_two_lists(aid1_list, aid2_list)
    print('grabbing %d unique chips' % (len(flat_unique)))
    chip_list = ibs.get_annot_chips(flat_unique)  # TODO config2_
    # convert to approprate colorspace
    chip_list = vt.convert_image_list_colorspace(chip_list, colorspace)
    ut.print_object_size(chip_list, 'chip_list')
    chip1_list, chip2_list = ut.uninvert_unique_two_lists(chip_list, reconstruct_tup)
    print('warping')

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
        fx1_list = [fm.T[0] for fm in fm_list]
        fx2_list = [fm.T[1] for fm in fm_list]
        warp_iter1 = ut.ProgressIter(zip(aid1_list, fx1_list, chip1_list,
                                         kpts1_m_list),
                                     nTotal=len(kpts1_m_list), lbl='warp1',
                                     adjust=True)
        warp_iter2 = ut.ProgressIter(zip(aid2_list, fx2_list, chip2_list,
                                         kpts2_m_list),
                                     nTotal=len(kpts2_m_list), lbl='warp2',
                                     adjust=True)
        warped_patches1_list = list(itertools.starmap(cacheget_wraped_patches, warp_iter1))
        warped_patches2_list = list(itertools.starmap(cacheget_wraped_patches, warp_iter2))
    else:
        warp_iter1 = ut.ProgressIter(zip(chip1_list, kpts1_m_list),
                                     nTotal=len(kpts1_m_list), lbl='warp1',
                                     adjust=True)
        warp_iter2 = ut.ProgressIter(zip(chip2_list, kpts2_m_list),
                                     nTotal=len(kpts2_m_list), lbl='warp2',
                                     adjust=True)
        warped_patches1_list = [vt.get_warped_patches(chip1, kpts1, patch_size=patch_size)[0]
                                for chip1, kpts1 in warp_iter1]
        warped_patches2_list = [vt.get_warped_patches(chip2, kpts2, patch_size=patch_size)[0]
                                for chip2, kpts2 in warp_iter2]

    del chip_list
    del chip1_list
    del chip2_list
    ut.print_object_size(warped_patches1_list, 'warped_patches1_list')
    ut.print_object_size(warped_patches2_list, 'warped_patches2_list')
    len1_list = list(map(len, warped_patches1_list))
    assert len1_list == list(map(len, warped_patches2_list))
    print('flattening')
    aid1_list_ = np.array(ut.flatten([[aid1] * len1 for len1, aid1 in zip(len1_list, aid1_list)]))
    aid2_list_ = np.array(ut.flatten([[aid2] * len1 for len1, aid2 in zip(len1_list, aid2_list)]))
    # Flatten metadata
    flat_metadata = {key: np.array(ut.flatten(val)) for key, val in metadata_lists.items()}
    flat_metadata['aid_pair'] = np.hstack(
        (np.array(aid1_list_)[:, None],
         np.array(aid2_list_)[:, None]))
    flat_metadata['fm'] = np.vstack(fm_list)
    flat_metadata['kpts1_m'] = np.vstack(kpts1_m_list)
    flat_metadata['kpts2_m'] = np.vstack(kpts2_m_list)

    #flat_metadata = ut.map_dict_vals(np.array, flat_metadata)

    warped_patch1_list = ut.flatten(warped_patches1_list)
    del warped_patches1_list
    warped_patch2_list = ut.flatten(warped_patches2_list)
    del warped_patches2_list
    return aid1_list_, aid2_list_, warped_patch1_list, warped_patch2_list, flat_metadata


def get_patchmetric_training_data_and_labels(ibs, aid1_list, aid2_list,
                                             kpts1_m_list, kpts2_m_list,
                                             fm_list, metadata_lists,
                                             patch_size, colorspace):
    """
    Notes:
        # FIXME: THERE ARE INCORRECT CORRESPONDENCES LABELED AS CORRECT THAT
        # NEED MANUAL CORRECTION EITHER THROUGH EXPLICIT LABLEING OR
        # SEGMENTATION MASKS

    CommandLine:
        python -m ibeis_cnn.ingest_ibeis --test-get_patchmetric_training_data_and_labels --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.ingest_ibeis import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb(defaultdb='PZ_MTEST')
        >>> (aid1_list, aid2_list, kpts1_m_list, kpts2_m_list, fm_list, metadata_lists) = get_aidpairs_and_matches(ibs, 10, 3)
        >>> pmcfg = PatchMetricDataConfig()
        >>> patch_size = pmcfg.patch_size
        >>> colorspace = pmcfg.colorspace
        >>> data, labels, flat_metadata = get_patchmetric_training_data_and_labels(ibs,
        ...     aid1_list, aid2_list, kpts1_m_list, kpts2_m_list, fm_list,
        ...     metadata_lists, patch_size, colorspace)
        >>> ut.quit_if_noshow()
        >>> draw_results.interact_siamsese_data_patches(labels, data)
        >>> ut.show_if_requested()
    """
    # To the removal of unknown pairs before computing the data

    tup = get_aidpair_patchmatch_training_data(
        ibs, aid1_list, aid2_list, kpts1_m_list, kpts2_m_list, fm_list,
        metadata_lists, patch_size, colorspace)
    aid1_list_, aid2_list_, warped_patch1_list, warped_patch2_list, flat_metadata = tup
    labels = get_aidpair_training_labels(ibs, aid1_list_, aid2_list_)
    img_list = ut.flatten(list(zip(warped_patch1_list, warped_patch2_list)))
    data = np.array(img_list)
    del img_list
    #data_per_label = 2
    assert labels.shape[0] == data.shape[0] // 2
    from ibeis import const
    assert np.all(labels != const.TRUTH_UNKNOWN)
    return data, labels, flat_metadata


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
        >>> from ibeis_cnn.ingest_ibeis import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb(defaultdb='PZ_MTEST')
        >>> tup = get_aidpairs_and_matches(ibs)
        >>> (aid1_list, aid2_list) = tup[0:2]
        >>> aid1_list = aid1_list[0:min(100, len(aid1_list))]
        >>> aid2_list = aid2_list[0:min(100, len(aid2_list))]
        >>> # execute function
        >>> labels = get_aidpair_training_labels(ibs, aid1_list, aid2_list)
        >>> result = ('labels = %s' % (ut.numpy_str(labels, threshold=10),))
        >>> print(result)
        labels = np.array([1, 0, 0, ..., 0, 1, 0], dtype=np.int32)
    """
    from ibeis import const
    truth_list = ibs.get_aidpair_truths(aid1_list, aid2_list)
    labels = truth_list
    # Mark different viewpoints as unknown for training
    isinconsistent_list = mark_inconsistent_viewpoints(ibs, aid1_list, aid2_list)
    labels[isinconsistent_list] = const.TRUTH_UNKNOWN
    return labels


class NewConfigBase(object):
    #def update(self, **kwargs):
    #    self.__dict__.update(**kwargs)

    def kw(self):
        return ut.KwargsWrapper(self)


class PatchMetricDataConfig(NewConfigBase):
    def __init__(pmcfg, **kwargs):
        pmcfg.patch_size = 64
        #pmcfg.colorspace = 'bgr'
        pmcfg.colorspace = 'gray'
        ut.update_existing(pmcfg.__dict__, kwargs)
        unhandled_keys = set(kwargs.keys()) - set(pmcfg.__dict__.keys())
        if len(unhandled_keys) > 0:
            raise AssertionError('[ConfigBaseError] unhandled_keys=%r' % (unhandled_keys,))

    def get_cfgstr(pmcfg):
        cfgstr_list = [
            'patch_size=%d' % (pmcfg.patch_size,),
        ]
        #if pmcfg.colorspace != 'bgr':
        cfgstr_list.append(pmcfg.colorspace)
        return ','.join(cfgstr_list)

    def get_data_shape(pmcfg):
        channels = 1 if pmcfg.colorspace == 'gray' else 3
        return (pmcfg.patch_size, pmcfg.patch_size, channels)


def cached_patchmetric_training_data_fpaths(ibs, aid1_list, aid2_list,
                                            kpts1_m_list, kpts2_m_list,
                                            fm_list, metadata_lists, **kwargs):
    """
    todo use size in cfgstrings
    kwargs is used for PatchMetricDataConfig

    from ibeis_cnn.ingest_ibeis import *
    """
    import utool as ut
    pmcfg = PatchMetricDataConfig(**kwargs)
    data_shape = pmcfg.get_data_shape()

    NOCACHE_TRAIN = ut.get_argflag('--nocache-train')

    semantic_uuids1 = ibs.get_annot_semantic_uuids(aid1_list)
    semantic_uuids2 = ibs.get_annot_semantic_uuids(aid2_list)
    aidpair_hashstr_list = list(map(ut.hashstr27, zip(semantic_uuids1, semantic_uuids2)))
    training_dname = ut.hashstr_arr27(aidpair_hashstr_list, pathsafe=True, lbl='patchmatch')

    nets_dir = ibs.get_neuralnet_dir()
    training_dpath = ut.unixjoin(nets_dir, training_dname)

    ut.ensuredir(nets_dir)
    ut.ensuredir(training_dpath)
    view_train_dir = ut.get_argflag('--vtd')
    if view_train_dir:
        ut.view_directory(training_dpath)

    fm_hashstr = ut.hashstr_arr27(np.vstack(fm_list), pathsafe=True, lbl='fm')
    cfgstr = fm_hashstr + '_' + pmcfg.get_cfgstr()
    #data_fpath = ut.unixjoin(training_dpath, 'data_' + cfgstr + '.pkl')
    #labels_fpath = ut.unixjoin(training_dpath, 'labels_'  + cfgstr + '.pkl')
    #metadata_fpath = ut.unixjoin(training_dpath, 'metadata_'  + cfgstr + '.pkl')
    data_fpath = ut.unixjoin(training_dpath, 'data_' + cfgstr + '.hdf5')
    labels_fpath = ut.unixjoin(training_dpath, 'labels_'  + cfgstr + '.hdf5')
    metadata_fpath = ut.unixjoin(training_dpath, 'metadata_'  + cfgstr + '.hdf5')

    # Change to use path friendlier hashes
    #if FIX_HASH:
    #    # Old hashes
    #    data_fpath_ = data_fpath
    #    labels_fpath_ = labels_fpath
    #    # New hashes
    #    fm_hashstr = ut.hashstr_arr27(fm_list, lbl='fm')
    #    cfgstr = fm_hashstr + '_' + pmcfg.get_cfgstr()
    #    data_fpath = ut.unixjoin(training_dpath, 'data_' + cfgstr + '.pkl')
    #    labels_fpath = ut.unixjoin(training_dpath, 'labels_'  + cfgstr + '.pkl')
    #    if ut.checkpath(data_fpath_, verbose=True):
    #        ut.move(data_fpath_, data_fpath)
    #    if ut.checkpath(labels_fpath_, verbose=True):
    #        ut.move(labels_fpath_, labels_fpath)

    if NOCACHE_TRAIN or not (
            ut.checkpath(data_fpath, verbose=True)
            and ut.checkpath(labels_fpath, verbose=True)
            and ut.checkpath(metadata_fpath, verbose=True)
    ):
        # Estimate how big the patches will be
        def estimate_data_bytes():
            data_per_label = 2
            num_data = sum(list(map(len, fm_list)))
            item_shape = pmcfg.get_data_shape()
            dtype_bytes = 1
            estimated_bytes = np.prod(item_shape) * num_data * data_per_label * dtype_bytes
            print('Estimated data size: ' + ut.byte_str2(estimated_bytes))
        estimate_data_bytes()
        # Extract the data and labels
        data, labels, flat_metadata = get_patchmetric_training_data_and_labels(
            ibs, aid1_list, aid2_list, kpts1_m_list, kpts2_m_list, fm_list,
            metadata_lists, **pmcfg.kw())
        # Save the data to cache
        ut.assert_eq(data.shape[1], pmcfg.patch_size)
        ut.assert_eq(data.shape[2], pmcfg.patch_size)
        # TODO; save metadata
        print('[write_data_and_labels] np.shape(data) = %r' % (np.shape(data),))
        print('[write_data_and_labels] np.shape(labels) = %r' % (np.shape(labels),))
        # TODO hdf5 for large data
        ut.save_hdf5(data_fpath, data)
        ut.save_hdf5(labels_fpath, labels)
        ut.save_hdf5(metadata_fpath, flat_metadata)
        #ut.save_cPkl(data_fpath, data)
        #ut.save_cPkl(labels_fpath, labels)
        #ut.save_cPkl(metadata_fpath, flat_metadata)
    else:
        print('data and labels cache hit')
    return data_fpath, labels_fpath, training_dpath, data_shape


def remove_unknown_training_pairs(ibs, aid1_list, aid2_list):
    return aid1_list, aid2_list


def get_aidpairs_and_matches(ibs, max_examples=None, num_top=3,
                             controlled=True, min_featweight=None,
                             acfg_name=None):
    """

    Args:
        ibs (IBEISController):  ibeis controller object
        max_examples (None): (default = None)
        num_top (int): (default = 3)
        controlled (bool): (default = True)

    Returns:
        tuple : patchmatch_tup = (aid1_list, aid2_list, kpts1_m_list,
                                   kpts2_m_list, fm_list, metadata_lists)
            aid pairs and matching keypoint pairs as well as the original index
            of the feature matches

    CommandLine:
        python -m ibeis_cnn.ingest_ibeis --test-get_aidpairs_and_matches --db PZ_Master0
        python -m ibeis_cnn.ingest_ibeis --test-get_aidpairs_and_matches --db PZ_MTEST --acfg ctrl:qindex=0:10 --show
        python -m ibeis_cnn.ingest_ibeis --test-get_aidpairs_and_matches --db NNP_Master3

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.ingest_ibeis import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb(defaultdb='PZ_MTEST')
        >>> acfg_name = ut.get_argval(('--aidcfg', '--acfg', '-a'),
        ...                             type_=str,
        ...                             default='ctrl:qindex=0:10')
        >>> max_examples = None
        >>> num_top = None
        >>> controlled = True
        >>> min_featweight = .99
        >>> patchmatch_tup = get_aidpairs_and_matches(ibs,
        >>>                                           max_examples=max_examples,
        >>>                                           num_top=num_top,
        >>>                                           controlled=controlled,
        >>>                                           min_featweight=min_featweight,
        >>>                                           acfg_name=acfg_name)
        >>> (aid1_list, aid2_list, kpts1_m_list, kpts2_m_list, fm_list, metadata_lists) = patchmatch_tup
        >>> ut.quit_if_noshow()
        >>> _iter = list(zip(aid1_list, aid2_list, kpts1_m_list, kpts2_m_list, fm_list))
        >>> _iter = ut.InteractiveIter(_iter, display_item=False)
        >>> import plottool as pt
        >>> import ibeis.viz
        >>> for aid1, aid2, kpts1, kpts2, fm in _iter:
        >>>     pt.reset()
        >>>     print('aid2 = %r' % (aid2,))
        >>>     print('aid1 = %r' % (aid1,))
        >>>     print('len(fm) = %r' % (len(fm),))
        >>>     ibeis.viz.viz_matches.show_matches2(ibs, aid1, aid2, fm=None, kpts1=kpts1, kpts2=kpts2)
        >>>     pt.update()
        >>> ut.show_if_requested()
    """

    def get_query_results():
        if acfg_name is None:
            print('OLD WAY OF FILTERING')
            from ibeis import ibsfuncs
            if controlled:
                # TODO: use acfg config
                qaid_list = ibsfuncs.get_two_annots_per_name_and_singletons(ibs, onlygt=True)
                daid_list = ibsfuncs.get_two_annots_per_name_and_singletons(ibs, onlygt=False)
            else:
                qaid_list = ibs.get_valid_aids()
                #from ibeis.model.hots import chip_match
                qaid_list = ut.list_compress(qaid_list, ibs.get_annot_has_groundtruth(qaid_list))
                daid_list = qaid_list
                if max_examples is not None:
                    daid_list = daid_list[0:min(max_examples, len(daid_list))]
        else:
            print('NEW WAY OF FILTERING')
            from ibeis.experiments import experiment_helpers
            acfg_list, expanded_aids_list = experiment_helpers.get_annotcfg_list(ibs, [acfg_name])
            #acfg = acfg_list[0]
            expanded_aids = expanded_aids_list[0]
            qaid_list, daid_list = expanded_aids

        if max_examples is not None:
            qaid_list = qaid_list[0:min(max_examples, len(qaid_list))]

        cfgdict = {
            'affine_invariance': False,
        }

        #import ibeis.other.dbinfo
        ibs.print_annotconfig_stats(qaid_list, daid_list, bigstr=True)
        #ibeis.other.dbinfo.print_qd_info(ibs, qaid_list, daid_list, verbose=False)
        qres_list, qreq_ = ibs.query_chips(
            qaid_list, daid_list, return_request=True, cfgdict=cfgdict)
        # TODO: Use ChipMatch2 instead of QueryResult
        #cm_list = [chip_match.ChipMatch2.from_qres(qres) for qres in qres_list]
        #for cm in cm_list:
        #    cm.evaluate_nsum_score(qreq_=qreq_)
        #aids1_list = [[cm.qaid] * num_top for cm in cm_list]
        #aids2_list = [[cm.qaid] * num_top for cm in cm_list]
        return qres_list, qreq_
    qres_list, qreq_ = get_query_results()

    def get_matchdata1():
        # Get aid pairs and feature matches
        if num_top is None:
            aids2_list = [qres.get_top_aids() for qres in qres_list]
        else:
            aids2_list = [qres.get_top_aids()[0:num_top] for qres in qres_list]
        aids1_list = [[qres.qaid] * len(aids2)
                      for qres, aids2 in zip(qres_list, aids2_list)]
        aid1_list_all = np.array(ut.flatten(aids1_list))
        aid2_list_all = np.array(ut.flatten(aids2_list))
        def take_qres_list_attr(attr):
            attrs_list = [ut.dict_take(getattr(qres, attr), aids2)
                          for qres, aids2 in zip(qres_list, aids2_list)]
            attr_list = ut.flatten(attrs_list)
            return attr_list
        fm_list_all = take_qres_list_attr(attr='aid2_fm')
        metadata_all = {}
        filtkey_lists = ut.unique_unordered([tuple(qres.filtkey_list) for qres in qres_list])
        assert len(filtkey_lists) == 1, 'multiple fitlers used in this query'
        filtkey_list = filtkey_lists[0]
        fsv_list = take_qres_list_attr('aid2_fsv')
        for index, key in enumerate(filtkey_list):
            metadata_all[key] = [fsv.T[index] for fsv in fsv_list]
        metadata_all['fs'] = take_qres_list_attr('aid2_fs')
        # extract metadata (like feature scores and whatnot)
        return aid1_list_all, aid2_list_all, fm_list_all, metadata_all

    def get_matchdata2():
        aid1_list_all, aid2_list_all, fm_list_all, metadata_all = get_matchdata1()
        # Filter out bad training examples
        # (we are currently in annot-vs-annot format, not yet in patch-vs-patch)
        labels_all = get_aidpair_training_labels(ibs, aid1_list_all, aid2_list_all)
        has_gt = (labels_all != ibs.const.TRUTH_UNKNOWN)
        nonempty = [len(fm) > 0 for fm in fm_list_all]
        isvalid = np.logical_and(has_gt, nonempty)
        aid1_list_uneq = ut.list_compress(aid1_list_all, isvalid)
        aid2_list_uneq = ut.list_compress(aid2_list_all, isvalid)
        labels_uneq    = ut.list_compress(labels_all, isvalid)
        fm_list_uneq   = ut.list_compress(fm_list_all, isvalid)
        metadata_uneq  = {key: ut.list_compress(vals, isvalid)
                          for key, vals in metadata_all.items()}
        return aid1_list_uneq, aid2_list_uneq, labels_uneq, fm_list_uneq, metadata_uneq

    def get_matchdata3():
        # Filters in place
        aid1_list_uneq, aid2_list_uneq, labels_uneq, fm_list_uneq, metadata_uneq = get_matchdata2()

        #min_featweight = None
        if min_featweight is not None:
            print('filter by featweight')
            # Remove feature matches where the foreground weight is under a threshold
            flags_list = []
            for index in range(len(aid1_list_uneq)):
                aid1 = aid1_list_uneq[index]
                aid2 = aid2_list_uneq[index]
                fm = fm_list_uneq[index]

                fgweight1 = ibs.get_annot_fgweights(
                    [aid1], config2_=qreq_.get_internal_query_config2())[0][fm.T[0]]
                fgweight2 = ibs.get_annot_fgweights(
                    [aid2], config2_=qreq_.get_internal_data_config2())[0][fm.T[1]]
                flags = np.logical_and(fgweight1 > min_featweight,
                                       fgweight2 > min_featweight)
                flags_list.append(flags)

            import vtool as vt
            num_keep = sum([x.sum() for x in flags_list])
            num_total = sum([len(x) for x in flags_list])
            print('Kept ' + str(100 * num_keep / num_total) + '% of matches')
            fm_list_uneq2 = vt.zipcompress_safe(fm_list_uneq, flags_list, axis=0)
            metadata_uneq2  = {key: vt.zipcompress_safe(vals, flags_list, axis=0)
                               for key, vals in metadata_uneq.items()}
        else:
            fm_list_uneq2 = fm_list_uneq
            metadata_uneq2 = metadata_uneq

        return aid1_list_uneq, aid2_list_uneq, labels_uneq, fm_list_uneq2, metadata_uneq2

    def equalize_flat_flags(flat_labels, flat_scores):
        import vtool as vt  # NOQA

        labelhist = ut.dict_hist(flat_labels)
        # Print input distribution of labels
        print('[ingest_ibeis] original label histogram = \n' + ut.dict_str(labelhist))
        print('[ingest_ibeis] total = %r' % (sum(list(labelhist.values()))))

        pref_method = 'rand'
        #pref_method = 'scores'
        seed = 0
        rng = np.random.RandomState(seed)

        def pref_rand(type_indicies, min_, rng=rng):
            return rng.choice(type_indicies, size=min_, replace=False)

        def pref_first(type_indicies, min_):
            return type_indicies[:min_]

        def pref_scores(type_indicies, min_, flat_scores=flat_scores):
            sortx = flat_scores.take(type_indicies).argsort()[::-1]
            return type_indicies.take(sortx[:min_])

        sample_func = {
            'rand': pref_rand,
            'scores': pref_scores,
            'first': pref_first,
        }[pref_method]

        # Figure out how much of each label needs to be removed
        # record the indicies that will not be filtered in keep_indicies_list
        allowed_ratio = ut.PHI * .8
        #allowed_ratio = 1.0
        # Find the maximum and minimum number of labels over all types
        true_max_ = max(labelhist.values())
        true_min_ = min(labelhist.values())
        # Allow for some window around the minimum
        min_ = min(int(true_min_ * allowed_ratio), true_max_)
        print('Equalizing label distribution with method=%r' % (pref_method,))
        print('Allowing at most %d labels of a type' % (min_,))
        key_list, type_indicies_list = vt.group_indices(flat_labels)
        #type_indicies_list = [np.where(flat_labels == key)[0]
        #                      for key in six.iterkeys(labelhist)]
        keep_indicies_list = []
        for type_indicies in type_indicies_list:
            if min_ >= len(type_indicies):
                keep_indicies = type_indicies
            else:
                keep_indicies = sample_func(type_indicies, min_)
            keep_indicies_list.append(keep_indicies)
        # Create a flag for each flat label (patch-pair)
        flat_keep_idxs = np.hstack(keep_indicies_list)
        flat_flag_list = vt.index_to_boolmask(flat_keep_idxs, maxval=len(flat_labels))
        return flat_flag_list

    def equalize_labels():
        (aid1_list_uneq, aid2_list_uneq, labels_uneq, fm_list_uneq2,
         metadata_uneq2) = get_matchdata3()
        import vtool as vt
        print('flattening')
        # Find out how many examples each source holds
        len1_list = list(map(len, fm_list_uneq2))
        # Expand source labels so one exists for each datapoint
        flat_labels = ut.flatten([
            [label] * len1
            for len1, label in zip(len1_list, labels_uneq)
        ])
        flat_labels = np.array(flat_labels)
        flat_scores = np.hstack(metadata_uneq2['fs'])
        flat_flag_list = equalize_flat_flags(flat_labels, flat_scores)

        # Unflatten back into source-vs-source pairs (annot-vs-annot)
        flags_list = ut.unflatten2(flat_flag_list, np.cumsum(len1_list))

        assert ut.depth_profile(flags_list) == ut.depth_profile(metadata_uneq2['fs'])

        fm_list_ = vt.zipcompress_safe(fm_list_uneq2, flags_list, axis=0)
        metadata_ = dict([
            (key, vt.zipcompress_safe(vals, flags_list))
            for key, vals in metadata_uneq2.items()
        ])

        # remove empty aids
        isnonempty_list = [len(fm) > 0 for fm in fm_list_]
        fm_list_eq = ut.list_compress(fm_list_, isnonempty_list)
        aid1_list_eq = ut.list_compress(aid1_list_uneq, isnonempty_list)
        aid2_list_eq = ut.list_compress(aid2_list_uneq, isnonempty_list)
        labels_eq    = ut.list_compress(labels_uneq, isnonempty_list)
        metadata_eq = dict([
            (key, ut.list_compress(vals, isnonempty_list))
            for key, vals in metadata_.items()
        ])

        # PRINT NEW LABEL STATS
        len1_list = list(map(len, fm_list_eq))
        flat_labels_eq = ut.flatten([[label] * len1 for len1, label in zip(len1_list, labels_eq)])
        labelhist_eq = {
            key: len(val)
            for key, val in six.iteritems(ut.group_items(flat_labels_eq, flat_labels_eq))}
        print('[ingest_ibeis] equalized label histogram = \n' + ut.dict_str(labelhist_eq))
        print('[ingest_ibeis] total = %r' % (sum(list(labelhist_eq.values()))))
        # --
        return aid1_list_eq, aid2_list_eq, fm_list_eq, labels_eq, metadata_eq

    #EQUALIZE_LABELS = True
    #if EQUALIZE_LABELS:
    aid1_list_eq, aid2_list_eq, fm_list_eq, labels_eq, metadata_eq = equalize_labels()

    # Convert annot-vs-annot pairs into raw feature-vs-feature pairs

    print('Building feature indicies')

    fx1_list = [fm.T[0] for fm in fm_list_eq]
    fx2_list = [fm.T[1] for fm in fm_list_eq]
    # Hack: use the ibeis cache to make quick lookups
    with ut.Timer('Reading keypoint sets (caching unique keypoints)'):
        ibs.get_annot_kpts(list(set(aid1_list_eq + aid2_list_eq)),
                           config2_=qreq_.get_internal_query_config2())
    with ut.Timer('Reading keypoint sets from cache'):
        kpts1_list = ibs.get_annot_kpts(aid1_list_eq,
                                        config2_=qreq_.get_internal_query_config2())
        kpts2_list = ibs.get_annot_kpts(aid2_list_eq,
                                        config2_=qreq_.get_internal_query_config2())

    # Save some memory
    ibs.print_cachestats_str()
    ibs.clear_table_cache(ibs.const.FEATURE_TABLE)
    print('Taking matching keypoints')
    kpts1_m_list = [kpts1.take(fx1, axis=0) for kpts1, fx1 in zip(kpts1_list, fx1_list)]
    kpts2_m_list = [kpts2.take(fx2, axis=0) for kpts2, fx2 in zip(kpts2_list, fx2_list)]

    (aid1_list, aid2_list, fm_list, metadata_lists) = (
        aid1_list_eq, aid2_list_eq, fm_list_eq, metadata_eq
    )
    #assert ut.get_list_column(ut.depth_profile(kpts1_m_list), 0) ==
    #ut.depth_profile(metadata_lists['fs'])
    patchmatch_tup = aid1_list, aid2_list, kpts1_m_list, kpts2_m_list, fm_list, metadata_lists
    return patchmatch_tup


# def get_background_training_patches(ibs, min_percent=0.80, max_percent=2.0,
#                                     dest_path=None, patch_size=96, patches_per_image=200):
#     """
#     """
#     import random
#     from os.path import join, expanduser

#     def resize_target(image, target_height=None, target_width=None):
#         assert target_height is not None or target_width is not None
#         height, width = image.shape[:2]
#         if target_height is not None and target_width is not None:
#             h = target_height
#             w = target_width
#         elif target_height is not None:
#             h = target_height
#             w = (width / height) * h
#         elif target_width is not None:
#             w = target_width
#             h = (height / width) * w
#         w, h = int(w), int(h)
#         return cv2.resize(image, (w, h))

#     def point_inside((x, y), (x0, y0, w, h)):
#         x1 = x0 + w
#         y1 = y0 + h
#         return x0 <= x and x <= x1 and y0 <= y and y <= y1

#     if dest_path is None:
#         dest_path = expanduser(join('~', 'Desktop', 'extracted'))

#     name = 'background_patches'
#     raw_path = join(dest_path, 'raw', name)
#     labels_path = join(dest_path, 'labels', name)

#     ut.remove_dirs(dest_path)
#     ut.ensuredir(dest_path)
#     ut.ensuredir(raw_path)
#     ut.ensuredir(labels_path)

#     gid_list = ibs.get_valid_gids()
#     size_list = ibs.get_image_sizes(gid_list)
#     aids_list = ibs.get_image_aids(gid_list)
#     bboxes_list = [ ibs.get_annot_bboxes(aid_list) for aid_list in aids_list ]

#     min_size = int(min_percent * patch_size)
#     max_size = int(max_percent * patch_size)

#     patch_size = (patch_size, patch_size)
#     zipped = zip(gid_list, size_list, aids_list, bboxes_list)
#     label_list = []
#     for gid, (width, height), aid_list, bbox_list in zipped:
#         image = ibs.get_images(gid)

#         # smallest_size = min(width, height)
#         # min_size = int(min_percent * smallest_size)
#         # max_size = int(max_percent * smallest_size)
#         # min_size = max(min_size, patch_size)

#         print('Processing GID: %r' % (gid, ))
#         print('\tAIDS  : %r' % (aid_list, ))
#         print('\tBBOXES: %r' % (bbox_list, ))

#         for (x0, y0, w, h) in bbox_list:
#             x1 = x0 + w
#             y1 = y0 + h
#             cv2.rectangle(image, (x0, y0), (x1, y1), (255, 0, 0))

#         for _ in range(patches_per_image):
#             x0 = random.randint(0, width)
#             y0 = random.randint(0, height)

#             x1, y1 = None, None
#             while x1 is None or y1 is None:
#                 x1_ = random.randint(0, width)
#                 y1_ = random.randint(0, height)
#                 dist_x = abs(x0 - x1_)
#                 dist_y = abs(y0 - y1_)
#                 if dist_x < min_size or max_size < dist_x:
#                     continue
#                 if dist_y < min_size or max_size < dist_y:
#                     continue
#                 x1 = x1_
#                 y1 = y1_

#             if x1 < x0:
#                 x0, x1 = x1, x0
#             if y1 < y0:
#                 y0, y1 = y1, y0

#             size = min(x1 - x0, y1 - y0)
#             x1 = x0 + size
#             y1 = y0 + size
#             center = ( int(np.mean((x0, x1))), int(np.mean((y0, y1))) )

#             inside = False
#             for bbox in bbox_list:
#                 if point_inside(center, bbox):
#                     inside = True
#                     break

#             patch = image[y0: y1, x0: x1]
#             patch = cv2.resize(patch, patch_size, interpolation=cv2.INTER_LANCZOS4)
#             patch_filename = 'patch_gid_%s_bbox_%d_%d_%d_%d.png' % (gid, x0, y0, size, size, )
#             patch_class = 'positive' if inside else 'negative'
#             patch_color = (0, 0, 255) if inside else (0, 255, 0)

#             patch_filepath = join(raw_path, patch_filename)
#             cv2.imwrite(patch_filepath, patch)
#             cv2.rectangle(image, (x0, y0), (x1, y1), patch_color)

#             label = '%s,%s' % (patch_filename, patch_class)
#             label_list.append(label)

#         image = resize_target(image, target_width=1000)
#         cv2.imshow('', image)
#         cv2.waitKey(0)

#     with open(join(labels_path, 'labels.csv'), 'w') as labels:
#         label_str = '\n'.join(label_list)
#         labels.write(label_str)


def get_background_training_patches2(ibs, dest_path=None, patch_size=48,
                                     patch_size_min=0.80, patch_size_max=1.25,
                                     annot_size=300, patience=20,
                                     patches_per_annotation=30):
    import random
    from os.path import join, expanduser

    def resize_target(image, target_height=None, target_width=None):
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
        return cv2.resize(image, (w, h))

    def point_inside((x, y), (x0, y0, w, h)):
        x1 = x0 + w
        y1 = y0 + h
        return x0 <= x and x <= x1 and y0 <= y and y <= y1

    dbname_mapping = {
        'ELPH_Master'    : 'elephant_savanna',
        'GIR_Master'     : 'giraffe_reticulated',
        'GZ_Master'      : 'zebra_grevys',
        'NNP_MasterGIRM' : 'giraffe_masai',
        'PZ_Master1'     : 'zebra_plains',
    }

    if dest_path is None:
        dest_path = expanduser(join('~', 'Desktop', 'extracted'))

    dbname = ibs.dbname
    positive_category = dbname_mapping.get(dbname, 'positive')
    negative_category = 'negative'

    name = 'background_patches'
    raw_path = join(dest_path, 'raw', name)
    labels_path = join(dest_path, 'labels', name)

    # ut.remove_dirs(dest_path)
    ut.ensuredir(dest_path)
    ut.ensuredir(raw_path)
    ut.ensuredir(labels_path)

    gid_list = ibs.get_valid_gids()
    aids_list = ibs.get_image_aids(gid_list)
    bboxes_list = [ ibs.get_annot_bboxes(aid_list) for aid_list in aids_list ]

    zipped = zip(gid_list, aids_list, bboxes_list)
    label_list = []
    global_positives = 0
    global_negatives = 0
    for gid, aid_list, bbox_list in zipped:
        image = ibs.get_images(gid)
        h, w, c = image.shape

        args = (gid, global_positives, global_negatives, len(label_list), )
        print('Processing GID: %r [ %r / %r = %r]' % args)
        print('\tAIDS  : %r' % (aid_list, ))
        print('\tBBOXES: %r' % (bbox_list, ))

        if len(aid_list) == 0 and len(bbox_list) == 0:
            aid_list = [None]
            bbox_list = [None]

        for aid, bbox in zip(aid_list, bbox_list):
            positives = 0
            negatives = 0

            if aid is not None:
                xtl, ytl, w_, h_ = bbox
                xbr, ybr = xtl + w_, ytl + h_

                # cv2.rectangle(image, (xtl, ytl), (xbr, ybr), (255, 0, 0))

                if min(w_, h_) / max(w_, h_) <= 0.25:
                    continue

                modifier = w_ / annot_size
                patch_size_ = patch_size * modifier
                patch_size_min_ = patch_size_ * patch_size_min
                patch_size_max_ = patch_size_ * patch_size_max

                for index in range(patches_per_annotation):
                    counter = 0
                    found = False
                    while not found and counter < patience:
                        counter += 1
                        patch_size_random = random.uniform(patch_size_min_, patch_size_max_)
                        patch_size_final = int(round(patch_size_random))

                        if patch_size_final > w_ or patch_size_final > h_:
                            continue

                        radius = patch_size_final // 2
                        centerx = random.randint(xtl + radius, xbr - radius)
                        centery = random.randint(ytl + radius, ybr - radius)

                        x0 = centerx - radius
                        y0 = centery - radius
                        x1 = centerx + radius
                        y1 = centery + radius

                        if x0 < 0 or x0 >= w or x1 < 0 or x1 >= w:
                            continue
                        if y0 < 0 or y0 >= w or y1 < 0 or y1 >= w:
                            continue

                        found = True

                    # Sanity checks
                    try:
                        assert x1 > x0
                        assert y1 > y0
                        assert x1 - x0 >= patch_size // 2
                        assert y1 - y0 >= patch_size // 2
                        assert x0 >= 0 and x0 < w and x1 >= 0 and x1 < w
                        assert y0 >= 0 and y0 < h and y1 >= 0 and y1 < h
                    except AssertionError:
                        found = False

                    if found:
                        positives += 1
                        # cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0))
                        chip = image[y0: y1, x0: x1]
                        chip = cv2.resize(chip, (patch_size, patch_size),
                                          interpolation=cv2.INTER_LANCZOS4)

                        values = (dbname, gid, positive_category, x0, y0, x1, y1, )
                        patch_filename = '%s_patch_gid_%s_%s_bbox_%d_%d_%d_%d.png' % values
                        patch_filepath = join(raw_path, patch_filename)
                        cv2.imwrite(patch_filepath, chip)
                        label = '%s,%s' % (patch_filename, positive_category)
                        label_list.append(label)

                positives_ = positives
            else:
                modifier = 4.0
                patch_size_ = patch_size * modifier
                patch_size_min_ = patch_size_ * patch_size_min
                patch_size_max_ = patch_size_ * patch_size_max
                positives_ = patches_per_annotation

            delta = global_positives - global_negatives
            if delta >= 2 * patches_per_annotation:
                print('SUPERCHARGE NEGATIVES')
                positives_ = int(positives_ * 1.5)
            elif delta <= -2 * patches_per_annotation:
                print('UNDERCHARGE NEGATIVES')
                positives_ = int(positives_ * 0.5)

            for index in range(positives_):
                counter = 0
                found = False
                while not found and counter < patience:
                    counter += 1
                    patch_size_random = random.uniform(patch_size_min_, patch_size_max_)
                    patch_size_final = int(round(patch_size_random))

                    radius = patch_size_final // 2
                    centerx = random.randint(radius, w - radius)
                    centery = random.randint(radius, h - radius)

                    inside = False
                    for bbox in bbox_list:
                        if bbox is None:
                            continue
                        if point_inside((centerx, centery), bbox):
                            inside = True
                            break

                    if inside:
                        continue

                    x0 = centerx - radius
                    y0 = centery - radius
                    x1 = centerx + radius
                    y1 = centery + radius

                    if x0 < 0 or x0 >= w or x1 < 0 or x1 >= w:
                        continue
                    if y0 < 0 or y0 >= w or y1 < 0 or y1 >= w:
                        continue

                    found = True

                # Sanity checks
                try:
                    assert x1 > x0
                    assert y1 > y0
                    assert x1 - x0 >= patch_size // 2
                    assert y1 - y0 >= patch_size // 2
                    assert x0 >= 0 and x0 < w and x1 >= 0 and x1 < w
                    assert y0 >= 0 and y0 < h and y1 >= 0 and y1 < h
                except AssertionError:
                    found = False

                if found:
                    negatives += 1
                    # cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255))
                    chip = image[y0: y1, x0: x1]
                    chip = cv2.resize(chip, (patch_size, patch_size),
                                      interpolation=cv2.INTER_LANCZOS4)

                    values = (dbname, gid, negative_category, x0, y0, x1, y1, )
                    patch_filename = '%s_patch_gid_%s_%s_bbox_%d_%d_%d_%d.png' % values
                    patch_filepath = join(raw_path, patch_filename)
                    cv2.imwrite(patch_filepath, chip)
                    label = '%s,%s' % (patch_filename, negative_category)
                    label_list.append(label)

            global_positives += positives
            global_negatives += negatives

        # image = resize_target(image, target_width=1000)
        # cv2.imshow('', image)
        # cv2.waitKey(0)

    args = (global_positives, global_negatives, len(label_list), )
    print('Final Split: [ %r / %r = %r]' % args)

    with open(join(labels_path, 'labels.csv'), 'a') as labels:
        label_str = '\n'.join(label_list) + '\n'
        labels.write(label_str)

    return args


def get_cnn_detector_training_images(ibs, dest_path=None, image_size=256):
    from os.path import join, expanduser

    def resize_target(image, target_height=None, target_width=None):
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
        return cv2.resize(image, (w, h))

    dbname_mapping = {
        'ELPH_Master'    : 'elephant_savanna',
        'GIR_Master'     : 'giraffe_reticulated',
        'GZ_Master'      : 'zebra_grevys',
        'NNP_MasterGIRM' : 'giraffe_masai',
        'PZ_Master1'     : 'zebra_plains',
    }

    if dest_path is None:
        dest_path = expanduser(join('~', 'Desktop', 'extracted'))

    dbname = ibs.dbname
    positive_category = dbname_mapping.get(dbname, 'positive')

    dbname = ibs.dbname
    name = 'saliency_detector'
    raw_path = join(dest_path, 'raw', name)
    labels_path = join(dest_path, 'labels', name)

    # ut.remove_dirs(dest_path)
    ut.ensuredir(dest_path)
    ut.ensuredir(raw_path)
    ut.ensuredir(labels_path)

    gid_list = ibs.get_valid_gids()
    aids_list = ibs.get_image_aids(gid_list)
    bboxes_list = [ ibs.get_annot_bboxes(aid_list) for aid_list in aids_list ]

    label_list = []
    zipped_list = zip(gid_list, aids_list, bboxes_list)
    global_bbox_list = []
    for gid, aid_list, bbox_list in zipped_list:
        image = ibs.get_images(gid)
        height, width, channels = image.shape

        args = (gid, )
        print('Processing GID: %r' % args)
        print('\tAIDS  : %r' % (aid_list, ))
        print('\tBBOXES: %r' % (bbox_list, ))

        image_ = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LANCZOS4)

        values = (dbname, gid, )
        patch_filename = '%s_image_gid_%s.png' % values
        patch_filepath = join(raw_path, patch_filename)
        cv2.imwrite(patch_filepath, image_)

        bbox_list_ = []
        for aid, (xtl, ytl, w, h) in zip(aid_list, bbox_list):
            xr = round(w / 2)
            yr = round(h / 2)
            xc = xtl + xr
            yc = ytl + yr

            # Normalize to unit box
            xr /= width
            xc /= width
            yr /= height
            yc /= height

            xr = min(1.0, max(0.0, xr))
            xc = min(1.0, max(0.0, xc))
            yr = min(1.0, max(0.0, yr))
            yc = min(1.0, max(0.0, yc))

            args = (xc, yc, xr, yr, )
            bbox_str = '%s:%s:%s:%s' % args
            bbox_list_.append(bbox_str)
            global_bbox_list.append(args)

            # xtl_ = int((xc - xr) * image_size)
            # ytl_ = int((yc - yr) * image_size)
            # xbr_ = int((xc + xr) * image_size)
            # ybr_ = int((yc + yr) * image_size)
            # cv2.rectangle(image_, (xtl_, ytl_), (xbr_, ybr_), (0, 255, 0))

        # cv2.imshow('', image_)
        # cv2.waitKey(0)

        aid_list_str = ';'.join(map(str, aid_list))
        bbox_list_str = ';'.join(map(str, bbox_list_))
        label = '%s,%s,%s,%s' % (patch_filename, positive_category, aid_list_str, bbox_list_str)
        label_list.append(label)

    with open(join(labels_path, 'labels.csv'), 'a') as labels:
        label_str = '\n'.join(label_list) + '\n'
        labels.write(label_str)

    return global_bbox_list


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis_cnn.ingest_ibeis
        python -m ibeis_cnn.ingest_ibeis --allexamples
        python -m ibeis_cnn.ingest_ibeis --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
