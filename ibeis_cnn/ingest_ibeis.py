# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import utool as ut
import numpy as np
import six
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
    aid1_list_ = np.array(ut.flatten([[aid1] * len1 for len1, aid1 in zip(len1_list, aid1_list)]))
    aid2_list_ = np.array(ut.flatten([[aid2] * len1 for len1, aid2 in zip(len1_list, aid2_list)]))
    # Flatten metadata
    flat_metadata = {key: ut.flatten(val) for key, val in metadata_lists.items()}
    flat_metadata['aid_pair'] = np.hstack((np.array(aid1_list_)[:, None], np.array(aid2_list_)[:, None]))
    flat_metadata['fm'] = np.vstack(fm_list)
    flat_metadata['kpts1_m'] = np.vstack(kpts1_m_list)
    flat_metadata['kpts2_m'] = np.vstack(kpts2_m_list)

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
    def update(self, **kwargs):
        self.__dict__.update(**kwargs)

    def kw(self):
        return ut.KwargsWrapper(self)


class PatchMetricDataConfig(NewConfigBase):
    def __init__(pmcfg):
        pmcfg.patch_size = 64
        #pmcfg.colorspace = 'bgr'
        pmcfg.colorspace = 'gray'

    def get_cfgstr(pmcfg):
        cfgstr_list = [
            'patch_size=%d' % (pmcfg.patch_size,),
        ]
        if pmcfg.colorspace != 'bgr':
            cfgstr_list.append(pmcfg.colorspace)
        return ','.join(cfgstr_list)

    def get_data_shape(pmcfg):
        channels = 1 if pmcfg.colorspace == 'gray' else 3
        return (pmcfg.patch_size, pmcfg.patch_size, channels)


def cached_patchmetric_training_data_fpaths(ibs, aid1_list, aid2_list, kpts1_m_list, kpts2_m_list, fm_list, metadata_lists):
    """
    todo use size in cfgstrings

    from ibeis_cnn.ingest_ibeis import *
    """
    import utool as ut
    pmcfg = PatchMetricDataConfig()
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
    data_fpath = ut.unixjoin(training_dpath, 'data_' + cfgstr + '.pkl')
    labels_fpath = ut.unixjoin(training_dpath, 'labels_'  + cfgstr + '.pkl')
    metadata_fpath = ut.unixjoin(training_dpath, 'metadata_'  + cfgstr + '.pkl')

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
        ut.save_cPkl(data_fpath, data)
        ut.save_cPkl(labels_fpath, labels)
        ut.save_cPkl(metadata_fpath, flat_metadata)
    else:
        print('data and labels cache hit')
    return data_fpath, labels_fpath, training_dpath, data_shape


def remove_unknown_training_pairs(ibs, aid1_list, aid2_list):
    return aid1_list, aid2_list


def get_aidpairs_and_matches(ibs, max_examples=None, num_top=3, controlled=True):
    """
    Returns:
        aid pairs and matching keypoint pairs as well as the original index of the feature matches

    CommandLine:
        python -m ibeis_cnn.ingest_ibeis --test-get_aidpairs_and_matches --db PZ_Master0
        python -m ibeis_cnn.ingest_ibeis --test-get_aidpairs_and_matches --db PZ_MTEST
        python -m ibeis_cnn.ingest_ibeis --test-get_aidpairs_and_matches --db NNP_Master3

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.ingest_ibeis import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb(defaultdb='PZ_MTEST')
        >>> max_examples = None
        >>> num_top = None
        >>> patchmatch_tup = get_aidpairs_and_matches(ibs, max_examples=None, num_top=num_top, controlled=True)
    """

    from ibeis import ibsfuncs
    if controlled:
        qaid_list = ibsfuncs.get_two_annots_per_name_and_singletons(ibs, onlygt=True)
        daid_list = ibsfuncs.get_two_annots_per_name_and_singletons(ibs, onlygt=False)
    else:
        qaid_list = ibs.get_valid_aids()
        #from ibeis.model.hots import chip_match
        qaid_list = ut.list_compress(qaid_list, ibs.get_annot_has_groundtruth(qaid_list))
        daid_list = qaid_list
        if max_examples is not None:
            daid_list = daid_list[0:min(max_examples, len(daid_list))]

    if max_examples is not None:
        qaid_list = qaid_list[0:min(max_examples, len(qaid_list))]

    cfgdict = {
        #'affine_invariance': False,
    }

    import ibeis.dev.dbinfo
    ibeis.dev.dbinfo.print_qd_info(ibs, qaid_list, daid_list, verbose=False)
    qres_list, qreq_ = ibs.query_chips(
        qaid_list, daid_list, return_request=True, cfgdict=cfgdict)
    # TODO: Use ChipMatch2 instead of QueryResult
    #cm_list = [chip_match.ChipMatch2.from_qres(qres) for qres in qres_list]
    #for cm in cm_list:
    #    cm.evaluate_nsum_score(qreq_=qreq_)
    #aids1_list = [[cm.qaid] * num_top for cm in cm_list]
    #aids2_list = [[cm.qaid] * num_top for cm in cm_list]

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
    # extract metadata (like feature scores and whatnot)
    metadata_all = {}
    filtkey_lists = ut.unique_unordered([tuple(qres.filtkey_list) for qres in qres_list])
    assert len(filtkey_lists) == 1, 'multiple fitlers used in this query'
    filtkey_list = filtkey_lists[0]
    fsv_list = take_qres_list_attr('aid2_fsv')
    for index, key in enumerate(filtkey_list):
        metadata_all[key] = [fsv.T[index] for fsv in fsv_list]
    metadata_all['fs'] = take_qres_list_attr('aid2_fs')

    # Filter out bad training examples
    # (we are currently in annot-vs-annot format, not yet in patch-vs-patch)
    labels_all = get_aidpair_training_labels(ibs, aid1_list_all, aid2_list_all)
    isvalid = (labels_all != ibs.const.TRUTH_UNKNOWN)
    aid1_list_uneq = ut.list_compress(aid1_list_all, isvalid)
    aid2_list_uneq = ut.list_compress(aid2_list_all, isvalid)
    fm_list_uneq   = ut.list_compress(fm_list_all, isvalid)
    labels_uneq    = ut.list_compress(labels_all, isvalid)
    metadata_uneq  = {key: ut.list_compress(vals, isvalid) for key, vals in metadata_all.items()}

    def equalize_labels():
        import vtool as vt
        print('flattening')
        # Find out how many examples each source holds
        len1_list = list(map(len, fm_list_uneq))
        # Expand source labels so one exists for each datapoint
        flat_labels = ut.flatten([[label] * len1 for len1, label in zip(len1_list, labels_uneq)])
        #aid1_list_ = ut.flatten([[aid1] * len1 for len1, aid1 in zip(len1_list, aid1_list)])
        #aid2_list_ = ut.flatten([[aid2] * len1 for len1, aid2 in zip(len1_list, aid2_list)])
        labelhist = ut.dict_hist(flat_labels)
        # Print input distribution of labels
        print('[ingest_ibeis] original label histogram = \n' + ut.dict_str(labelhist))
        print('[ingest_ibeis] total = %r' % (sum(list(labelhist.values()))))
        flat_labels = np.array(flat_labels)

        flat_fs = np.hstack(metadata_uneq['fs'])
        def preference_highscores(type_indicies, min_):
            sortx = flat_fs.take(type_indicies).argsort()[::-1]
            keep_indicies = type_indicies.take(sortx[:min_])
            return keep_indicies
        preference_strat = 'rand'
        #preference_strat = preference_highscores

        # Figure out how much of each label needs to be removed
        # record the indicies that will not be filtered in keep_indicies_list
        allowed_ratio = ut.PHI * .8
        #allowed_ratio = 1.0
        def compute_keep_indicies(flat_labels, labelhist, allowed_ratio, preference_strat='rand', seed=0):
            # Find the maximum and minimum number of labels over all types
            true_max_ = max(labelhist.values())
            true_min_ = min(labelhist.values())
            # Allow for some window around the minimum
            min_ = min(int(true_min_ * allowed_ratio), true_max_)
            print('Allowing at most %d labels of a type' % (min_,))
            keep_indicies_list = []
            randstate = np.random.RandomState(seed)
            type_indicies_list = [np.where(flat_labels == key)[0] for key in six.iterkeys(labelhist)]
            for type_indicies in type_indicies_list:
                size = min(min_, len(type_indicies))
                if size == len(type_indicies):
                    # no need to filter
                    keep_indicies = type_indicies
                else:
                    if preference_strat == 'rand':
                        keep_indicies = randstate.choice(type_indicies, size=min_, replace=False)
                    elif preference_strat == 'first':
                        # Be stupid and grab the first few labels of each type
                        keep_indicies = type_indicies[:min_]
                    elif ut.is_funclike(preference_strat):
                        # custom function
                        keep_indicies = preference_highscores(type_indicies, min_)
                    else:
                        raise NotImplementedError('preference_strat = %r' % (preference_strat,))

                keep_indicies_list.append(keep_indicies)
            # Create a flag for each flat label (patch-pair)
            flag_list = vt.index_to_boolmask(np.hstack(keep_indicies_list), maxval=len(flat_labels))
            return flag_list

        flag_list = compute_keep_indicies(flat_labels, labelhist, allowed_ratio, preference_strat)
        #fm_flat, cumsum = ut.invertible_flatten2_numpy(fm_list)
        # Unflatten back into source-vs-source pairs (annot-vs-annot)
        flags_list = ut.unflatten2(flag_list, np.cumsum(len1_list))
        fm_list_ = vt.zipcompress(fm_list_uneq, flags_list, axis=0)
        metadata_ = {key: vt.zipcompress(vals, flags_list) for key, vals in metadata_uneq.items()}

        # remove empty aids
        isnonempty_list = [len(fm) != 0 for fm in fm_list_]
        fm_list_eq = ut.list_compress(fm_list_, isnonempty_list)
        aid1_list_eq = ut.list_compress(aid1_list_uneq, isnonempty_list)
        aid2_list_eq = ut.list_compress(aid2_list_uneq, isnonempty_list)
        labels_eq    = ut.list_compress(labels_uneq, isnonempty_list)
        metadata_eq = {key: ut.list_compress(vals, isnonempty_list) for key, vals in metadata_.items()}

        # PRINT NEW LABEL STATS
        len1_list = list(map(len, fm_list_eq))
        flat_labels_eq = ut.flatten([[label] * len1 for len1, label in zip(len1_list, labels_eq)])
        labelhist_eq = {key: len(val) for key, val in six.iteritems(ut.group_items(flat_labels_eq, flat_labels_eq))}
        print('[ingest_ibeis] equalized label histogram = \n' + ut.dict_str(labelhist_eq))
        print('[ingest_ibeis] total = %r' % (sum(list(labelhist_eq.values()))))
        # --
        return aid1_list_eq, aid2_list_eq, fm_list_eq, labels_eq, metadata_eq

    EQUALIZE_LABELS = True
    if EQUALIZE_LABELS:
        aid1_list_eq, aid2_list_eq, fm_list_eq, labels_eq, metadata_eq = equalize_labels()
        pass

    # Convert annot-vs-annot pairs into raw feature-vs-feature pairs

    print('Building feature indicies')

    fx1_list = [fm.T[0] for fm in fm_list_eq]
    fx2_list = [fm.T[1] for fm in fm_list_eq]
    # Hack: use the ibeis cache to make quick lookups
    with ut.Timer('Reading keypoint sets (caching unique keypoints)'):
        ibs.get_annot_kpts(list(set(aid1_list_eq + aid2_list_eq)), config2_=qreq_.get_internal_query_config2())
    with ut.Timer('Reading keypoint sets from cache'):
        kpts1_list = ibs.get_annot_kpts(aid1_list_eq, config2_=qreq_.get_internal_query_config2())
        kpts2_list = ibs.get_annot_kpts(aid2_list_eq, config2_=qreq_.get_internal_query_config2())

    # Save some memory
    ibs.print_cachestats_str()
    ibs.clear_table_cache(ibs.const.FEATURE_TABLE)
    print('Taking matching keypoints')
    kpts1_m_list = [kpts1.take(fx1, axis=0) for kpts1, fx1 in zip(kpts1_list, fx1_list)]
    kpts2_m_list = [kpts2.take(fx2, axis=0) for kpts2, fx2 in zip(kpts2_list, fx2_list)]

    (aid1_list, aid2_list, fm_list, metadata_lists) = (
        aid1_list_eq, aid2_list_eq, fm_list_eq, metadata_eq
    )
    #assert ut.get_list_column(ut.depth_profile(kpts1_m_list), 0) == ut.depth_profile(metadata_lists['fs'])
    patchmatch_tup = aid1_list, aid2_list, kpts1_m_list, kpts2_m_list, fm_list, metadata_lists
    return patchmatch_tup


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
