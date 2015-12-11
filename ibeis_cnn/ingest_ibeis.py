# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import utool as ut
import numpy as np
import vtool as vt
import six
import cv2
import itertools
from six.moves import zip, map, range
from functools import partial
#from ibeis_cnn import utils
from ibeis_cnn import draw_results  # NOQA
print, rrr, profile = ut.inject2(__name__, '[ibeis_cnn.ingest_ibeis]')


FIX_HASH = True


def get_aidpairs_partmatch(ibs, acfg_name):
    """

    CommandLine:
        python -m ibeis_cnn.ingest_ibeis --exec-get_aidpairs_partmatch

    SeeAlso:
        extract_annotpair_training_chips

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.ingest_ibeis import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb(defaultdb='PZ_Master1')
        >>> #ibs = ibeis.opendb(defaultdb='PZ_MTEST')
        >>> #ibs = ibeis.opendb(defaultdb='PZ_FlankHack')
        >>> acfg_name = ut.get_argval(('--aidcfg', '--acfg', '-a'),
        ...                             type_=str,
        ...                             default='ctrl:pername=None,excluderef=False,contrib_contains=FlankHack')
        >>> aid_pairs, label_list, flat_metadata = get_aidpairs_partmatch(ibs, acfg_name)
    """
    print('NEW WAY OF FILTERING')
    from ibeis.expt import experiment_helpers
    acfg_list, expanded_aids_list = experiment_helpers.get_annotcfg_list(
        ibs, [acfg_name])
    #acfg = acfg_list[0]
    expanded_aids = expanded_aids_list[0]
    qaid_list, daid_list = expanded_aids
    available_aids = np.unique(ut.flatten([qaid_list, daid_list]))
    tup = ibs.partition_annots_into_corresponding_groups(qaid_list, daid_list)
    aids1_list, aids2_list, other_aids1, other_aids2 = tup
    multiton_aids = np.unique(ut.flatten(aids1_list + aids2_list))
    #singletons = ut.flatten(other_aids2 + other_aids1)

    ibs.print_annotconfig_stats(qaid_list, daid_list, bigstr=True)

    # Positive Examples
    print('Sampling positive examples')
    nested_pairs = list(
        map(list, itertools.starmap(ut.iprod, zip(aids1_list, aids2_list))))
    pos_aid_pairs = np.vstack(nested_pairs)
    # Filter Self
    flag_list = pos_aid_pairs.T[0] != pos_aid_pairs.T[1]
    pos_aid_pairs = pos_aid_pairs.compress(flag_list, axis=0)
    # Filter bad viewpoints
    TAU = np.pi * 2
    _np_get_annot_yaws = ut.accepts_numpy(ibs.get_annot_yaws_asfloat.im_func)
    yaw_pairs = _np_get_annot_yaws(ibs, pos_aid_pairs)
    yawdist = vt.ori_distance(yaw_pairs.T[0], yaw_pairs.T[1])
    flag_list = np.logical_or(np.isnan(yawdist), yawdist < TAU / 8.)
    pos_aid_pairs = pos_aid_pairs.compress(flag_list, axis=0)
    #pos_aid_pairs = vt.unique_rows(pos_aid_pairs)  # should be unncessary
    assert len(vt.unique_rows(pos_aid_pairs)) == len(pos_aid_pairs)
    print('pos_aid_pairs.shape = %r' % (pos_aid_pairs.shape,))

    # Hard Negative Examples
    print('Sampling hard negative examples')
    num_hard_neg_per_aid = max(1, len(pos_aid_pairs) // len(multiton_aids))
    cfgdict = {
        'affine_invariance': False,
        'fg_on': not ut.WIN32,
    }
    cm_list, qreq_ = ibs.query_chips(
        qaid_list, daid_list, return_request=True, cfgdict=cfgdict,
        return_cm=True)
    hardneg_aids1 = [[cm.qaid] for cm in (cm_list)]
    hardneg_aids2 = [cm.get_top_gf_aids(ibs, ntop=num_hard_neg_per_aid)
                     for cm in cm_list]
    hardneg_aid_pairs = np.array(ut.flatten(
        itertools.starmap(ut.iprod, zip(hardneg_aids1, hardneg_aids2))))

    # Random Negative Examples
    # TODO: may be able to say not a match from viewpoint?
    print('Sampling random negative examples')
    num_rand_neg_per_aid = max(1, len(pos_aid_pairs) // len(multiton_aids))
    rng = np.random.RandomState(0)
    randneg_aid_pairs = []
    neg_aid_pool = available_aids
    neg_nid_pool = np.array(ibs.get_annot_nids(neg_aid_pool))
    for aid, nid in zip(neg_aid_pool, neg_nid_pool):
        is_valid = np.not_equal(neg_nid_pool, nid)
        p = is_valid / (is_valid.sum())
        chosen = rng.choice(neg_aid_pool, size=num_rand_neg_per_aid,
                            replace=False, p=p)
        chosen_pairs = list(ut.iprod([aid], chosen))
        randneg_aid_pairs.extend(chosen_pairs)
    randneg_aid_pairs = np.array(randneg_aid_pairs)

    # Concatenate both types of negative examples
    #print('Building negative examples')
    #_neg_aid_pairs = np.vstack((hardneg_aid_pairs, randneg_aid_pairs))
    #neg_aid_pairs = vt.unique_rows(_neg_aid_pairs)

    # Unsure Examples
    # TODO: use quality and viewoint labelings to determine this
    # as well as metadata from the annotmatch table
    print('hardneg_aid_pairs.shape = %r' % (hardneg_aid_pairs.shape,))
    print('randneg_aid_pairs.shape = %r' % (randneg_aid_pairs.shape,))
    print('pos_aid_pairs.shape = %r' % (pos_aid_pairs.shape,))

    print('Building labels')
    const = ibs.const
    unflat_pairs = (pos_aid_pairs, hardneg_aid_pairs, randneg_aid_pairs)
    type_labels = (const.TRUTH_MATCH, const.TRUTH_NOT_MATCH,
                   const.TRUTH_NOT_MATCH)
    type_meta_labels = ('pos', 'hardneg', 'randneg')

    def _expand(type_list):
        return [[item] * len(pairs)
                for pairs, item in zip(unflat_pairs, type_list)]

    _aid_pairs = np.vstack(unflat_pairs)
    _labels = np.hstack(_expand(type_labels))

    flat_metadata = {
        'meta_label': ut.flatten(_expand(type_meta_labels))
    }

    print('Filtering Duplicates')
    nonunique_flags = vt.nonunique_row_flags(_aid_pairs)
    print('Filtered %d duplicate pairs' % (nonunique_flags.sum()))
    print('Nonunique stats:')
    for key, val in flat_metadata.items():
        print(ut.dict_hist(ut.list_compress(val, nonunique_flags)))
    unique_flags = ~nonunique_flags
    # Do filtering
    aid_pairs = _aid_pairs.compress(unique_flags, axis=0)
    label_list = _labels.compress(unique_flags, axis=0)
    for key, val in flat_metadata.items():
        flat_metadata[key] = ut.list_compress(val, unique_flags)
    print('Final Stats')
    for key, val in flat_metadata.items():
        print(ut.dict_hist(val))

    groupsizes = map(len, vt.group_indices(aid_pairs.T[0])[1])
    ut.print_dict(ut.dict_hist(groupsizes), 'groupsize freq')
    flat_metadata['aid_pairs'] = aid_pairs
    return aid_pairs, label_list, flat_metadata


def extract_annotpair_training_chips(ibs, aid_pairs, **kwargs):
    """

    CommandLine:
        python -m ibeis_cnn.ingest_ibeis --exec-extract_annotpair_training_chips --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.ingest_ibeis import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> if False:
        >>>     ibs = ibeis.opendb(defaultdb='PZ_Master1')
        >>>     acfg_name = ut.get_argval(('--aidcfg', '--acfg', '-a'),
        >>>                                 type_=str,
        >>>                                 default='ctrl:pername=None,excluderef=False,contrib_contains=FlankHack')
        >>> else:
        >>>     ibs = ibeis.opendb(defaultdb='PZ_MTEST')
        >>>     acfg_name = ut.get_argval(('--aidcfg', '--acfg', '-a'),
        >>>                                  type_=str,
        >>>                                  default='ctrl:pername=None,excluderef=False')
        >>> #ibs = ibeis.opendb(defaultdb='PZ_FlankHack')
        >>> aid_pairs, label_list, flat_metadata = get_aidpairs_partmatch(ibs, acfg_name)
        >>> s = slice(2, len(aid_pairs), len(aid_pairs) // ut.get_argval('--x', type_=int, default=8))
        >>> #aid_pairs = aid_pairs[s]
        >>> #label_list = label_list[s]
        >>> #flat_metadata = dict([(key, val[s]) for key, val in flat_metadata.items()])
        >>> rchip1_list, rchip2_list = extract_annotpair_training_chips(ibs, aid_pairs)
        >>> ut.quit_if_noshow()
        >>> from ibeis_cnn import draw_results  # NOQA
        >>> interact = draw_results.interact_patches(label_list, (rchip1_list, rchip2_list), flat_metadata, chunck_sizes=(2, 2), ibs=ibs)
        >>> ut.show_if_requested()
    """

    # TODO extract chips in a sane manner
    import ibeis.model.hots.vsone_pipeline
    kwargs = kwargs.copy()
    part_chip_width  = kwargs.pop('part_chip_width', 256)
    part_chip_height = kwargs.pop('part_chip_height', 128)
    colorspace = kwargs.pop('colorspace', 'gray')
    assert len(kwargs) == 0, 'unhandled arguments %r' % (kwargs,)

    size = (part_chip_width, part_chip_height)

    #cfgdict = {}
    import ibeis.control.IBEISControl
    import ibeis.model.hots.query_request
    assert isinstance(ibs, ibeis.control.IBEISControl.IBEISController)
    qreq_ = ibs.new_query_request(aid_pairs.T[0][0:1], aid_pairs.T[1][0:1])
    assert isinstance(qreq_, ibeis.model.hots.query_request.QueryRequest)
    qconfig2_ = qreq_.extern_query_config2
    dconfig2_ = qreq_.extern_data_config2

    warped = True

    def compute_alignment(pair_metadata, qreq_=qreq_):
        aid1, aid2 = pair_metadata['aid1'], pair_metadata['aid2']
        print('Computing alignment aidpair=(%r, %r)' % (aid1, aid2))
        matches, match_metadata = ibeis.model.hots.vsone_pipeline.vsone_single(
            aid1, aid2, qreq_, verbose=False)
        match_metadata.clear_stored(['dlen_sqrd2', 'vecs2', 'kpts2', 'kpts1', 'vecs1'])
        return match_metadata

    def make_warped_chips(pair_metadata, size=size):
        match_metadata = pair_metadata['match_metadata']
        print('Warping Chips aidpair=(%r, %r)' % (
            pair_metadata['aid1'], pair_metadata['aid2']))
        rchip1 = match_metadata['rchip1']
        rchip2 = match_metadata['rchip2']
        if warped:
            H1 = match_metadata['H_RAT']
            #print('WARPING')
            # Initial Warping
            wh2 = vt.get_size(rchip2)
            rchip1_ = vt.warpHomog(rchip1, H1, wh2) if H1 is not None else rchip1
            # Cropping to remove parts of the image that (probably) cannot match
            if True:
                isfill = vt.get_pixel_dist(rchip1_, np.array([0, 0, 0])) == 0
                rowslice, colslice = vt.get_crop_slices(isfill)
                rchip1_crop = rchip1_[rowslice, colslice]
                rchip2_crop = rchip2[rowslice, colslice]
            else:
                rchip1_crop = rchip1_
                rchip2_crop = rchip2
            rchip1 = rchip1_crop
            rchip2 = rchip2_crop
            # Make sure match_metadata doesn't take up too much memory
            match_metadata.clear_evaluated()
        # Resize to fit into a neural network
        rchip1_sz = cv2.resize(rchip1, size, interpolation=cv2.INTER_LANCZOS4)
        rchip2_sz = cv2.resize(rchip2, size, interpolation=cv2.INTER_LANCZOS4)
        # hack
        rchip1_sz = vt.convert_image_list_colorspace([rchip1_sz], colorspace)[0]
        rchip2_sz = vt.convert_image_list_colorspace([rchip2_sz], colorspace)[0]
        return (rchip1_sz, rchip2_sz)

    def make_lazy_resize_funcs(pair_metadata):
        tmp_meta = ut.LazyDict(verbose=False)
        tmp_meta['warped_chips'] = partial(make_warped_chips, pair_metadata)
        def lazy_rchip1_sz(tmp_meta=tmp_meta):
            return tmp_meta['warped_chips'][0]
        def lazy_rchip2_sz(tmp_meta=tmp_meta):
            return tmp_meta['warped_chips'][1]
        return lazy_rchip1_sz, lazy_rchip2_sz

    # Compute alignments

    pairmetadata_list = []
    for aid1, aid2 in ut.ProgressIter(aid_pairs, lbl='Align Info', adjust=True):
        pair_metadata = ibs.get_annot_pair_lazy_dict(
            ibs, aid1, aid2, qconfig2_, dconfig2_)
        pair_metadata['match_metadata'] = partial(compute_alignment, pair_metadata)
        pairmetadata_list.append(pair_metadata)

    # Warp the Chips

    rchip1_list = ut.LazyList(verbose=False)
    rchip2_list = ut.LazyList(verbose=False)
    for pair_metadata in ut.ProgressIter(pairmetadata_list, lbl='Building Warped Chips', adjust=True):
        #rchip1_sz, rchip2_sz = make_warped_chips(pair_metadata)
        rchip1_sz, rchip2_sz = make_lazy_resize_funcs(pair_metadata)
        rchip1_list.append(rchip1_sz)
        rchip2_list.append(rchip2_sz)

    return rchip1_list, rchip2_list
    """
    import plottool as pt
    pt.imshow(vt.stack_images(rchip1, rchip2)[0])
    pt.imshow(vt.stack_images(rchip1_sz, rchip2_sz)[0])
    """


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
        >>> draw_results.interact_patches(label_list, (warped_patch1_list, warped_patch2_list), flat_metadata)
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
    chip_list = ibs.get_annot_chips(flat_unique)  # TODO config2_
    # convert to approprate colorspace
    chip_list = vt.convert_image_list_colorspace(chip_list, colorspace)
    ut.print_object_size(chip_list, 'chip_list')
    chip1_list, chip2_list = ut.uninvert_unique_two_lists(chip_list, reconstruct_tup)
    print('warping')

    class PatchExtractCache(object):

        def __init__(self, patch_size):
            self.patch_size = patch_size
            self.cache_ = {}

        def idcache_find_misses(self, id_list, cache_):
            # Generalize?
            val_list = [cache_.get(id_, None) for id_ in id_list]
            ismiss_list = [val is None for val in val_list]
            return val_list, ismiss_list

        def idcache_save(self, ismiss_list, miss_vals, id_list, val_list, cache_):
            # Generalize?
            miss_indices = ut.list_where(ismiss_list)
            miss_ids  = ut.filter_items(id_list, ismiss_list)
            # overwrite missed output
            for index, val in zip(miss_indices, miss_vals):
                val_list[index] = val
            # cache save
            for id_, val in zip(miss_ids, miss_vals):
                cache_[id_] = val

        def cacheget_wraped_patches(self, aid, fxs, chip, kpts):
            # +-- Custom ids
            id_list = [(aid, fx) for fx in fxs]
            # L__
            val_list, ismiss_list = self.idcache_find_misses(id_list, self.cache_)
            if any(ismiss_list):
                # +-- Custom evaluate misses
                kpts_miss = kpts.compress(ismiss_list, axis=0)
                miss_vals = vt.get_warped_patches(chip, kpts_miss, patch_size=self.patch_size)[0]
                # L__
                self.idcache_save(ismiss_list, miss_vals, id_list, val_list, self.cache_)
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
    pec = PatchExtractCache(patch_size=patch_size)
    warped_patches1_list = list(itertools.starmap(pec.cacheget_wraped_patches, warp_iter1))
    warped_patches2_list = list(itertools.starmap(pec.cacheget_wraped_patches, warp_iter2))
    #warp_iter1 = ut.ProgressIter(zip(chip1_list, kpts1_m_list),
    #                             nTotal=len(kpts1_m_list), lbl='warp1',
    #                             adjust=True)
    #warp_iter2 = ut.ProgressIter(zip(chip2_list, kpts2_m_list),
    #                             nTotal=len(kpts2_m_list), lbl='warp2',
    #                             adjust=True)
    #warped_patches1_list = [vt.get_warped_patches(chip1, kpts1, patch_size=patch_size)[0]
    #                        for chip1, kpts1 in warp_iter1]
    #warped_patches2_list = [vt.get_warped_patches(chip2, kpts2, patch_size=patch_size)[0]
    #                        for chip2, kpts2 in warp_iter2]

    #del chip_list
    #del chip1_list
    #del chip2_list
    ut.print_object_size(warped_patches1_list, 'warped_patches1_list')
    ut.print_object_size(warped_patches2_list, 'warped_patches2_list')
    len1_list = list(map(len, fm_list))
    assert ut.lmap(len, warped_patches1_list) == ut.lmap(len, warped_patches2_list), 'bug'
    assert ut.lmap(len, warped_patches1_list) == len1_list, 'bug'
    print('flattening')
    aid1_list_ = np.array(ut.flatten([[aid1] * len1 for len1, aid1 in zip(len1_list, aid1_list)]))
    aid2_list_ = np.array(ut.flatten([[aid2] * len1 for len1, aid2 in zip(len1_list, aid2_list)]))
    # Flatten metadata
    flat_metadata = {key: np.array(ut.flatten(val)) for key, val in metadata_lists.items()}
    flat_metadata['aid_pairs'] = np.hstack(
        (np.array(aid1_list_)[:, None],
         np.array(aid2_list_)[:, None]))
    flat_metadata['fm'] = np.vstack(fm_list)
    flat_metadata['kpts1_m'] = np.vstack(kpts1_m_list)
    flat_metadata['kpts2_m'] = np.vstack(kpts2_m_list)

    #flat_metadata = ut.map_dict_vals(np.array, flat_metadata)

    warped_patch1_list = ut.flatten(warped_patches1_list)
    warped_patch2_list = ut.flatten(warped_patches2_list)
    #del warped_patches1_list
    #del warped_patches2_list
    return aid1_list_, aid2_list_, warped_patch1_list, warped_patch2_list, flat_metadata


def flatten_patch_data(ibs, aid1_list, aid2_list, kpts1_m_list, kpts2_m_list, fm_list, metadata_lists):
    # TODO; rectify
    len1_list = list(map(len, fm_list))
    print('flattening')
    aid1_list_ = np.array(ut.flatten([[aid1] * len1 for len1, aid1 in zip(len1_list, aid1_list)]))
    aid2_list_ = np.array(ut.flatten([[aid2] * len1 for len1, aid2 in zip(len1_list, aid2_list)]))
    # Flatten metadata
    flat_metadata = {key: np.array(ut.flatten(val)) for key, val in metadata_lists.items()}
    flat_metadata['aid_pairs'] = np.hstack(
        (np.array(aid1_list_)[:, None],
         np.array(aid2_list_)[:, None]))
    flat_metadata['fm'] = np.vstack(fm_list)
    flat_metadata['kpts1_m'] = np.vstack(kpts1_m_list)
    flat_metadata['kpts2_m'] = np.vstack(kpts2_m_list)
    labels = get_aidpair_training_labels(ibs, aid1_list_, aid2_list_)
    return aid1_list_, aid2_list_, labels, flat_metadata
    # TEMP


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
    (aid1_list_, aid2_list_, warped_patch1_list, warped_patch2_list,
     flat_metadata) = tup
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
    yaw1_list = np.array(ut.replace_nones(ibs.get_annot_yaws(aid2_list), np.nan))
    yaw2_list = np.array(ut.replace_nones(ibs.get_annot_yaws(aid1_list), np.nan))
    yawdist_list = vt.ori_distance(yaw1_list, yaw2_list)
    TAU = np.pi * 2
    isinconsistent_list = yawdist_list > TAU / 8
    return isinconsistent_list


def get_aidpair_training_labels(ibs, aid1_list_, aid2_list_):
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
    truth_list = ibs.get_aidpair_truths(aid1_list_, aid2_list_)
    labels = truth_list
    # Mark different viewpoints as unknown for training
    isinconsistent_list = mark_inconsistent_viewpoints(ibs, aid1_list_, aid2_list_)
    labels[isinconsistent_list] = ibs.const.TRUTH_UNKNOWN
    return labels


# Estimate how big the patches will be
def estimate_data_bytes(num_data, item_shape):
    data_per_label = 2
    dtype_bytes = 1
    estimated_bytes = (
        np.prod(item_shape) * num_data * data_per_label *
        dtype_bytes)
    print('Estimated data size: ' + ut.byte_str2(estimated_bytes))


class NewConfigBase(object):
    def __init__(self, **kwargs):
        self.update(**kwargs)

    def update(self, **kwargs):
        self.__dict__.update(**kwargs)
        ut.update_existing(self.__dict__, kwargs)
        unhandled_keys = set(kwargs.keys()) - set(self.__dict__.keys())
        if len(unhandled_keys) > 0:
            raise AssertionError(
                '[ConfigBaseError] unhandled_keys=%r' % (unhandled_keys,))

    def kw(self):
        return ut.KwargsWrapper(self)


class PartMatchDataConfig(NewConfigBase):
    def __init__(pmcfg, **kwargs):
        pmcfg.part_chip_width = 256
        pmcfg.part_chip_height = 128
        pmcfg.colorspace = 'gray'
        super(PartMatchDataConfig, pmcfg).__init__(**kwargs)

    def get_cfgstr(pmcfg):
        cfgstr_list = [
            'sz=(%d,%d)' % (pmcfg.part_chip_width, pmcfg.part_chip_height),
        ]
        cfgstr_list.append(pmcfg.colorspace)
        return ','.join(cfgstr_list)

    def get_data_shape(pmcfg):
        channels = 1 if pmcfg.colorspace == 'gray' else 3
        return (pmcfg.part_chip_height, pmcfg.part_chip_width, channels)


def cached_part_match_training_data_fpaths(ibs, aid_pairs, label_list,
                                            flat_metadata, **kwargs):
    """
    CommandLine:
        python -m ibeis_cnn --tf netrun --db PZ_MTEST --acfg ctrl:pername=None,excluderef=False --ensuredata --show --datatype=siam-part --nocache-train --nocache-cnn

    """
    NOCACHE_TRAIN = ut.get_argflag('--nocache-train')

    pmcfg = PartMatchDataConfig(**kwargs)
    data_shape = pmcfg.get_data_shape()

    semantic_uuids1 = ibs.get_annot_semantic_uuids(aid_pairs.T[0])
    semantic_uuids2 = ibs.get_annot_semantic_uuids(aid_pairs.T[1])
    aidpair_hashstr_list = list(map(ut.hashstr27, zip(semantic_uuids1, semantic_uuids2)))
    training_dname = ut.hashstr_arr27(
        aidpair_hashstr_list, pathsafe=True, lbl='part_match')

    nets_dir = ibs.get_neuralnet_dir()
    training_dpath = ut.unixjoin(nets_dir, training_dname)

    ut.ensuredir(nets_dir)
    ut.ensuredir(training_dpath)
    view_train_dir = ut.get_argflag('--vtd')
    if view_train_dir:
        ut.view_directory(training_dpath)

    cfgstr = pmcfg.get_cfgstr()
    data_fpath = ut.unixjoin(training_dpath, 'data_' + cfgstr + '.hdf5')
    labels_fpath = ut.unixjoin(training_dpath, 'labels_'  + cfgstr + '.hdf5')
    metadata_fpath = ut.unixjoin(training_dpath, 'metadata_'  + cfgstr + '.hdf5')

    if NOCACHE_TRAIN or not (
            ut.checkpath(data_fpath, verbose=True) and
            ut.checkpath(labels_fpath, verbose=True) and
            ut.checkpath(metadata_fpath, verbose=True)
    ):
        estimate_data_bytes(len(aid_pairs), pmcfg.get_data_shape())
        # Extract the data and labels
        rchip1_list, rchip2_list = extract_annotpair_training_chips(
            ibs, aid_pairs, **pmcfg.kw())

        datagen_ = zip(rchip1_list, rchip2_list)
        datagen = ut.ProgressIter(datagen_,
                                  nTotal=len(rchip1_list),
                                  lbl='Evaluating', adjust=False)
        data = np.array(list(ut.flatten(datagen)))

        flat_metadata = ut.map_dict_vals(np.array, flat_metadata)

        #img_list = ut.flatten(list(zip(rchip1_list,
        #                               rchip2_list)))
        #data = np.array(img_list)
        #del img_list
        labels = label_list
        #data_per_label = 2
        assert labels.shape[0] == data.shape[0] // 2
        #data, labels, flat_metadata
        # Save the data to cache
        ut.assert_eq(data.shape[1], pmcfg.part_chip_height)
        ut.assert_eq(data.shape[2], pmcfg.part_chip_width)
        # TODO; save metadata
        print('[write_part_data] np.shape(data) = %r' % (np.shape(data),))
        print('[write_part_labels] np.shape(labels) = %r' % (np.shape(labels),))
        # TODO hdf5 for large data
        ut.save_hdf5(data_fpath, data)
        ut.save_hdf5(labels_fpath, labels)
        ut.save_hdf5(metadata_fpath, flat_metadata)
        #ut.save_cPkl(data_fpath, data)
        #ut.save_cPkl(labels_fpath, labels)
        #ut.save_cPkl(metadata_fpath, flat_metadata)
    else:
        print('data and labels cache hit')
    return data_fpath, labels_fpath, metadata_fpath, training_dpath, data_shape


class PatchMetricDataConfig(NewConfigBase):
    def __init__(pmcfg, **kwargs):
        pmcfg.patch_size = 64
        #pmcfg.colorspace = 'bgr'
        pmcfg.colorspace = 'gray'
        super(PatchMetricDataConfig, pmcfg).__init__(**kwargs)

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

    NOCACHE_TRAIN = ut.get_argflag(('--nocache-train', '--nocache-cnn'))

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
            ut.checkpath(data_fpath, verbose=True) and
            ut.checkpath(labels_fpath, verbose=True) and
            ut.checkpath(metadata_fpath, verbose=True)
    ):
        estimate_data_bytes(sum(list(map(len, fm_list))),
                            pmcfg.get_data_shape())
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
    return data_fpath, labels_fpath, metadata_fpath, training_dpath, data_shape


def remove_unknown_training_pairs(ibs, aid1_list, aid2_list):
    return aid1_list, aid2_list


def get_aidpairs_and_matches(ibs, max_examples=None, num_top=3,
                             controlled=True, min_featweight=None,
                             acfg_name=None):
    """
    Gets data for training a patch match network.

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
        python -m ibeis_cnn.ingest_ibeis --test-get_aidpairs_and_matches --db PZ_Master1 --acfg default --show
        python -m ibeis_cnn.ingest_ibeis --test-get_aidpairs_and_matches --db PZ_MTEST --acfg ctrl:qindex=0:10 --show
        python -m ibeis_cnn.ingest_ibeis --test-get_aidpairs_and_matches --db NNP_Master3

        python -m ibeis_cnn.ingest_ibeis --test-get_aidpairs_and_matches --db PZ_MTEST --acfg default:is_known=True,qmin_pername=2,view=primary,species=primary,minqual=ok --show

        python -m ibeis_cnn.ingest_ibeis --test-get_aidpairs_and_matches --db PZ_Master1 --acfg default:is_known=True,qmin_pername=2,view=primary,species=primary,minqual=ok --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.ingest_ibeis import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb(defaultdb='PZ_MTEST')
        >>> acfg_name = ut.get_argval(('--aidcfg', '--acfg', '-a', '--acfg-name'),
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
        >>> print('Visualizing')
        >>> # Visualize feature scores
        >>> tup = flatten_patch_data(
        >>>     ibs, aid1_list, aid2_list, kpts1_m_list, kpts2_m_list, fm_list,
        >>>     metadata_lists)
        >>> labels, flat_metadata = tup[2:]
        >>> fs = flat_metadata['fs']
        >>> encoder = vt.ScoreNormalizer(adjust=1)
        >>> encoder.fit(fs, labels)
        >>> encoder.visualize(bin_width=.001, fnum=None)
        >>> # Visualize parent matches
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

    Ignore:


    """

    def get_query_results():
        if acfg_name is not None:
            print('NEW WAY OF FILTERING')
            from ibeis.expt import experiment_helpers
            acfg_list, expanded_aids_list = experiment_helpers.get_annotcfg_list(ibs, [acfg_name])
            #acfg = acfg_list[0]
            expanded_aids = expanded_aids_list[0]
            qaid_list, daid_list = expanded_aids
        else:
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

        if max_examples is not None:
            qaid_list = qaid_list[0:min(max_examples, len(qaid_list))]

        cfgdict = {
            'affine_invariance': False,
            'fg_on': not ut.WIN32,
        }

        #import ibeis.other.dbinfo
        ibs.print_annotconfig_stats(qaid_list, daid_list, bigstr=True)
        #ibeis.other.dbinfo.print_qd_info(ibs, qaid_list, daid_list, verbose=False)
        qres_list, qreq_ = ibs.query_chips(
            qaid_list, daid_list, return_request=True, cfgdict=cfgdict)
        # TODO: Use ChipMatch2 instead of QueryResult
        #cm_list = [chip_match.ChipMatch2.from_qres(cm) for cm in qres_list]
        #for cm in cm_list:
        #    cm.evaluate_nsum_score(qreq_=qreq_)
        #aids1_list = [[cm.qaid] * num_top for cm in cm_list]
        #aids2_list = [[cm.qaid] * num_top for cm in cm_list]
        return qres_list, qreq_
    qres_list, qreq_ = get_query_results()

    def get_matchdata1():
        # Get aid pairs and feature matches
        if num_top is None:
            aids2_list = [cm.get_top_aids() for cm in qres_list]
        else:
            aids2_list = [cm.get_top_aids()[0:num_top] for cm in qres_list]
        aids1_list = [[cm.qaid] * len(aids2)
                      for cm, aids2 in zip(qres_list, aids2_list)]
        aid1_list_all = np.array(ut.flatten(aids1_list))
        aid2_list_all = np.array(ut.flatten(aids2_list))

        def take_qres_list_attr(attr):
            attrs_list = [ut.dict_take(getattr(cm, attr), aids2)
                          for cm, aids2 in zip(qres_list, aids2_list)]
            attr_list = ut.flatten(attrs_list)
            return attr_list
        fm_list_all = take_qres_list_attr(attr='aid2_fm')
        metadata_all = {}
        filtkey_lists = ut.unique_unordered([tuple(cm.filtkey_list) for cm in qres_list])
        assert len(filtkey_lists) == 1, 'multiple fitlers used in this query'
        filtkey_list = filtkey_lists[0]
        fsv_list = take_qres_list_attr('aid2_fsv')
        for index, key in enumerate(filtkey_list):
            metadata_all[key] = [fsv.T[index] for fsv in fsv_list]
        metadata_all['fs'] = take_qres_list_attr('aid2_fs')

        if True:
            neg_aid_pool = np.unique(ut.flatten([aid1_list_all, aid2_list_all]))
            randneg_aid1 = []
            randneg_aid2 = []
            randneg_fm = []
            rand_meta = {key: [] for key in metadata_all.keys()}
            neg_nid_pool = np.array(ibs.get_annot_nids(neg_aid_pool))
            rng = np.random.RandomState(0)
            num_rand_neg_per_aid = 3
            num_rand_fm = 30
            for aid, nid in ut.ProgIter(list(zip(neg_aid_pool, neg_nid_pool)), 'sample aid rand'):
                #is_valid = get_badtag_flags(ibs, [aid] * len(neg_aid_pool), neg_aid_pool)
                is_valid = np.not_equal(neg_nid_pool, nid)
                #is_valid = np.logical_and(, is_valid)
                p = is_valid / (is_valid.sum())
                chosen = rng.choice(neg_aid_pool, size=num_rand_neg_per_aid,
                                    replace=False, p=p)
                #chosen_pairs = list(ut.iprod([aid], chosen))
                randneg_aid1.extend([aid] * len(chosen))
                randneg_aid2.extend(chosen)

            neg_fws1 = ibs.get_annot_fgweights(randneg_aid1, config2_=qreq_.get_internal_query_config2())
            neg_fws2 = ibs.get_annot_fgweights(randneg_aid2, config2_=qreq_.get_internal_data_config2())

            for fw1, fw2 in ut.ProgIter(list(zip(neg_fws1, neg_fws2)), 'sample fm rand'):
                valid_fx1s = np.where(fw1 > min_featweight)[0]
                valid_fx2s = np.where(fw2 > min_featweight)[0]
                size = min(num_rand_fm, len(valid_fx1s), len(valid_fx2s))
                if size > 0:
                    chosen_fx1 = rng.choice(valid_fx1s, size=size, replace=False)
                    chosen_fx2 = rng.choice(valid_fx2s, size=size, replace=False)
                    fm = np.vstack([chosen_fx1, chosen_fx2]).T
                else:
                    fm = np.empty((0, 2), dtype=np.int)
                randneg_fm.append(fm)
                for key in rand_meta.keys():
                    rand_meta[key].append(np.array([0] * len(fm)))

            prev_total = sum(map(len, fm_list_all))
            adding = sum(map(len, randneg_fm))
            print('prev_total = %r' % (prev_total,))
            print('adding     = %r' % (adding,))

            metadata_all = ut.dict_isect_combine(metadata_all, rand_meta)
            fm_list_all = fm_list_all + randneg_fm
            aid1_list_all = np.append(aid1_list_all, randneg_aid1)
            aid2_list_all = np.append(aid2_list_all, randneg_aid2)

        # extract metadata (like feature scores and whatnot)
        return aid1_list_all, aid2_list_all, fm_list_all, metadata_all

    def get_badtag_flags(ibs, aid1_list, aid2_list):
        from ibeis import tag_funcs
        tag_filter_kw = dict(has_none=['photobomb', 'scenerymatch', 'joincase', 'splitcase'])
        am_rowids1 = ibs.get_annotmatch_rowid_from_superkey(aid1_list, aid2_list)
        am_rowids2 = ibs.get_annotmatch_rowid_from_superkey(aid2_list, aid1_list)
        case_tags1 = ibs.get_annotmatch_case_tags(am_rowids1)
        case_tags2 = ibs.get_annotmatch_case_tags(am_rowids2)
        flags1 = tag_funcs.filterflags_general_tags(case_tags1, **tag_filter_kw)
        flags2 = tag_funcs.filterflags_general_tags(case_tags2, **tag_filter_kw)
        flags_tag = ut.and_lists(flags1, flags2)
        return flags_tag

    def get_matchdata2():
        aid1_list_all, aid2_list_all, fm_list_all, metadata_all = get_matchdata1()
        # Filter out bad training examples
        # (we are currently in annot-vs-annot format, not yet in patch-vs-patch)
        labels_all = get_aidpair_training_labels(ibs, aid1_list_all, aid2_list_all)
        has_gt = (labels_all != ibs.const.TRUTH_UNKNOWN)
        nonempty = [len(fm) > 0 for fm in fm_list_all]
        # Filter pairs bad pairs of aids
        # using case tags

        flags_tag = get_badtag_flags(ibs, aid1_list_all, aid2_list_all)
        print(ut.filtered_infostr(flags_tag, 'annots', 'tag filters'))
        flags = ut.and_lists(flags_tag, flags_tag)
        #
        MIN_TD = 5 * 60  # 5 minutes at least
        timedelta_list = np.abs(ibs.get_annot_pair_timdelta(aid1_list_all, aid2_list_all))
        #isnan = np.isnan(timedelta_list)
        gf_tdflags = np.logical_or(labels_all == ibs.const.TRUTH_MATCH, timedelta_list > MIN_TD)
        print(ut.filtered_infostr(gf_tdflags, 'gf annots', 'timestamp'))
        flags = ut.and_lists(flags, gf_tdflags)
        # Remove small time deltas
        # --
        print(ut.filtered_infostr(flags, 'total invalid annots'))
        isvalid = np.logical_and(np.logical_and(has_gt, nonempty), flags)
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
            for index in ut.ProgIter(range(len(aid1_list_uneq)), 'filt fw', adjust=True):
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

            print(ut.filtered_infostr(ut.flatten(flags_list), 'feat matches', 'featweight'))
            fm_list_uneq2 = vt.zipcompress_safe(fm_list_uneq, flags_list, axis=0)
            metadata_uneq2  = {key: vt.zipcompress_safe(vals, flags_list, axis=0)
                               for key, vals in metadata_uneq.items()}
        else:
            fm_list_uneq2 = fm_list_uneq
            metadata_uneq2 = metadata_uneq

        return aid1_list_uneq, aid2_list_uneq, labels_uneq, fm_list_uneq2, metadata_uneq2

    def equalize_flat_flags(flat_labels, flat_scores):
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
    #with ut.Timer('Reading keypoint sets (caching unique keypoints)'):
    #    ibs.get_annot_kpts(list(set(aid1_list_eq + aid2_list_eq)),
    #                       config2_=qreq_.get_internal_query_config2())
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


def get_background_training_patches2(ibs, dest_path=None, patch_size=48,
                                     patch_size_min=0.80, patch_size_max=1.25,
                                     annot_size=300, patience=20,
                                     patches_per_annotation=30):
    """
    Get data for bg
    """
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
        return cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)

    def point_inside(tup1, tup2):
        (x, y) = tup1
        (x0, y0, w, h) = tup2
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


def get_cnn_detector_training_images(ibs, dest_path=None, image_size=128):
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

        # if gid > 20:
        #     continue

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
