# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from os.path import join
import numpy as np
import utool as ut
from ibeis_cnn import utils
from six.moves import range, zip
from os.path import dirname, exists, basename
print, rrr, profile = ut.inject2(__name__, '[ibeis_cnn.ingest_helpers]')


NOCACHE_DATA_SPLIT = ut.get_argflag('--nocache-datasplit')


def ondisk_data_split(data_fpath, labels_fpath, data_per_label, split_names=['train', 'valid', 'test'], fraction_list=[.2, .1], nocache=None):
    """
    splits into train / validation datasets on disk

    # TODO: metadata fpath

    split_names=['train', 'valid', 'test'], fraction_list=[.2, .1]

    CommandLine:
        python -m ibeis_cnn.ingest_helpers --test-ondisk_data_split

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.ingest_helpers import *  # NOQA
        >>> from ibeis_cnn import ingest_data
        >>> trainset = ingest_data.testdata_trainset()
        >>> data_fpath = trainset.data_fpath
        >>> labels_fpath = trainset.labels_fpath
        >>> data_per_label = trainset.data_per_label
        >>> split_names = ['train', 'valid', 'test']
        >>> fraction_list = [0.2, 0.1]
        >>> nocache = True
        >>> (data_fpath_dict, label_fpath_dict) = ondisk_data_split(data_fpath, labels_fpath, data_per_label, split_names, fraction_list, nocache)
        >>> from os.path import basename
        >>> data_bytes = ut.dict_map_apply_vals(data_fpath_dict, ut.get_file_nBytes_str)
        >>> label_bytes = ut.dict_map_apply_vals(label_fpath_dict, ut.get_file_nBytes_str)
        >>> data_fpath_dict = ut.dict_map_apply_vals(data_fpath_dict, basename)
        >>> label_fpath_dict = ut.dict_map_apply_vals(label_fpath_dict, basename)
        >>> print('(data_bytes, label_bytes) = %s' % (ut.list_str((data_bytes, label_bytes), nl=True),))
        >>> result = ('(data_fpath_dict, label_fpath_dict) = %s' % (ut.list_str((data_fpath_dict, label_fpath_dict), nl=True),))
        >>> print(result)
    """
    assert len(split_names) == len(fraction_list) + 1, 'must have one less fraction then split names'
    USE_FILE_UUIDS = False
    if USE_FILE_UUIDS:
        # Get uuid based on the data, so different data makes different validation paths
        data_uuid   = ut.get_file_uuid(data_fpath)
        labels_uuid = ut.get_file_uuid(labels_fpath)
        split_uuid = ut.augment_uuid(data_uuid, labels_uuid)
        hashstr_ = ut.hashstr(str(split_uuid), alphabet=ut.ALPHABET_16)
    else:
        # Faster to base on the data fpath if that already has a uuid in it
        hashstr_ = ut.hashstr(basename(data_fpath), alphabet=ut.ALPHABET_16)

    splitdir = join(dirname(data_fpath), 'data_splits')
    ut.ensuredir(splitdir)

    # Get the total fraction of data for each subset
    totalfrac_list = [1.0]
    for fraction in fraction_list:
        total = totalfrac_list[-1]
        right = total * fraction
        left = total * (1 - fraction)
        totalfrac_list[-1] = left
        totalfrac_list.append(right)

    split_data_fpaths = [join(splitdir, name + '_data_%.3f_' % (frac,) + hashstr_ + '.pkl')
                         for name, frac in zip(split_names, totalfrac_list)]
    split_labels_fpaths = [join(splitdir, name + '_labels_%.3f_' % (frac,) + hashstr_ + '.pkl')
                           for name, frac in zip(split_names, totalfrac_list)]

    is_cache_hit = (all(map(exists, split_data_fpaths)) and all(map(exists, split_labels_fpaths)))

    if nocache is None:
        nocache = NOCACHE_DATA_SPLIT

    if not is_cache_hit or nocache:
        print('Writing data splits')
        X_left, y_left = utils.load(data_fpath, labels_fpath)
        _iter = zip(fraction_list, split_data_fpaths, split_labels_fpaths)
        for fraction, x_fpath, y_fpath in _iter:
            _tup = utils.train_test_split(X_left, y_left, eval_size=fraction,
                                          data_per_label=data_per_label,
                                          shuffle=True)
            X_left, y_left, X_right, y_right = _tup
            #print('-----------')
            #print(x_fpath)
            #print(y_fpath)
            #print(X_right.shape[0] / X_left.shape[0])
            #print(y_right.shape[0] / y_left.shape[0])
            #print('-----------')
            utils.write_data_and_labels(X_left, y_left, x_fpath, y_fpath)
            X_left = X_right
            y_left = y_right
        x_fpath  = split_data_fpaths[-1]
        y_fpath = split_labels_fpaths[-1]
        utils.write_data_and_labels(X_left, y_left, x_fpath, y_fpath)

    data_fpath_dict = dict(zip(split_names, split_data_fpaths))
    label_fpath_dict = dict(zip(split_names, split_labels_fpaths))

    data_fpath_dict['all']  = data_fpath
    label_fpath_dict['all'] = labels_fpath
    return data_fpath_dict, label_fpath_dict


def open_mnist_files(labels_fpath, data_fpath):
    """
    References:
        http://g.sweyla.com/blog/2012/mnist-numpy/
    """
    import struct
    #import os
    import numpy as np
    from array import array as pyarray
    with open(labels_fpath, 'rb') as flbl:
        magic_nr, size = struct.unpack(">II", flbl.read(8))
        lbl = pyarray("b", flbl.read())

    with open(data_fpath, 'rb') as fimg:
        magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = pyarray("B", fimg.read())
    digits = np.arange(10)

    ind = [k for k in range(size) if lbl[k] in digits]
    N = len(ind)

    images = np.zeros((N, rows, cols), dtype=np.uint8)
    labels = np.zeros((N, 1), dtype=np.uint8)
    for i in range(len(ind)):
        images[i] = np.array(
            img[ind[i] * rows * cols: (ind[i] + 1) * rows * cols]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]
    return images, labels


def extract_liberty_style_patches(ds_path, pairs):
    """
    CommandLine:
        python -m ibeis_cnn.ingest_data --test-grab_cached_liberty_data --show

    """
    from itertools import product
    import numpy as np
    from PIL import Image
    import subprocess

    patch_x = 64
    patch_y = 64
    rows = 16
    cols = 16

    def _available_patches(ds_path):
        """
        Number of patches in _dataset_ (a path).

        Only available through the line count
        in info.txt -- use unix 'wc'.

        _path_ is supposed to be a path to
        a directory with bmp patchsets.
        """
        fname = join(ds_path, "info.txt")
        p = subprocess.Popen(['wc', '-l', fname],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        result, err = p.communicate()
        if p.returncode != 0:
            raise IOError(err)
        return int(result.strip().split()[0])

    #def read_patch_ids():
    #    pass

    def matches(ds_path, pairs):
        """Return _pairs_ many match/non-match pairs for _dataset_.
        _dataset_ is one of "liberty", "yosemite", "notredame".
        Every dataset has a number of match-files, that
        have _pairs_ many matches and non-matches (always
        the same number).
        The naming of these files is confusing, e.g. if there are 500 matching
        pairs and 500 non-matching pairs the file name is
        'm50_1000_1000_0.txt' -- in total 1000 patch-ids are used for matches,
        and 1000 patch-ids for non-matches. These patch-ids need not be
        unique.

        Also returns the used patch ids in a list.

        Extract all matching and non matching pairs from _fname_.
        Every line in the matchfile looks like:
            patchID1 3DpointID1 unused1 patchID2 3DpointID2 unused2
        'matches' have the same 3DpointID.

        Every file has the same number of matches and non-matches.
        """
        #pairs = 500
        match_fname = ''.join(
            ["m50_", str(2 * pairs), "_", str(2 * pairs), "_0.txt"])
        match_fpath = join(ds_path, match_fname)

        #print(pairs, "pairs each (matching/non_matching) from", match_fpath)

        with open(match_fpath) as match_file:
            # collect patches (id), and match/non-match pairs
            patch_ids, match, non_match = [], [], []

            for line in match_file:
                match_info = line.split()
                p1_id, p1_3d, p2_id, p2_3d = int(match_info[0]), int(
                    match_info[1]), int(match_info[3]), int(match_info[4])
                if p1_3d == p2_3d:
                    match.append((p1_id, p2_id))
                else:
                    non_match.append((p1_id, p2_id))
                patch_ids.append(p1_id)
                patch_ids.append(p2_id)

            patch_ids = list(set(patch_ids))
            patch_ids.sort()

            assert len(match) == len(
                non_match), "Different number of matches and non-matches."

        return match, non_match, patch_ids

    def _crop_to_numpy(patchfile, requested_indicies):
        """
        Convert _patchfile_ to a numpy array with patches per row.
        A _patchfile_ is a .bmp file.
        """
        pil_img = Image.open(patchfile)
        ptch_iter = (
            pil_img.crop(
                (col * patch_x, row * patch_y, (col + 1) * patch_x, (row + 1) * patch_y))
            for index, (row, col) in enumerate(product(range(rows), range(cols)))
            if index in requested_indicies
        )
        patches = [np.array(ptch) for ptch in ptch_iter]
        pil_img.close()
        return patches

    num_patch_per_bmp = rows * cols
    total_num_patches = _available_patches(ds_path)
    num_bmp_files, mod = divmod(total_num_patches, num_patch_per_bmp)

    patchfile_list = [join(ds_path, ''.join(['patches', str(i).zfill(4), '.bmp']))
                      for i in range(num_bmp_files)]

    # Build matching labels
    match_pairs, non_match_pairs, all_requested_patch_ids = matches(ds_path, pairs)
    all_requested_patch_ids = np.array(all_requested_patch_ids)
    print('len(match_pairs) = %r' % (len(match_pairs,)))
    print('len(non_match_pairs) = %r' % (len(non_match_pairs,)))
    print('len(all_requested_patch_ids) = %r' % (len(all_requested_patch_ids,)))

    assert len(list(set(ut.flatten(match_pairs) + ut.flatten(non_match_pairs)))) == len(all_requested_patch_ids)
    assert max(all_requested_patch_ids) <= total_num_patches

    # Read all patches out of the bmp file store
    all_patches = {}
    for pfx, patchfile in ut.ProgressIter(list(enumerate(patchfile_list)), lbl='Reading Patches'):
        patch_id_offset = pfx * num_patch_per_bmp
        # get local patch ids in this bmp file
        patch_ids_ = np.arange(num_patch_per_bmp) + patch_id_offset
        requested_patch_ids_ = np.intersect1d(patch_ids_, all_requested_patch_ids)
        requested_indicies = requested_patch_ids_ - patch_id_offset
        if len(requested_indicies) == 0:
            continue
        patches = _crop_to_numpy(patchfile, requested_indicies)
        for idx, patch in zip(requested_patch_ids_, patches):
            all_patches[idx] = patch

    # Read the last patches
    if mod > 0:
        pfx += 1
        patch_id_offset = pfx * num_patch_per_bmp
        patchfile = join(
            ds_path, ''.join(['patches', str(num_bmp_files).zfill(4), '.bmp']))
        patch_ids_ = np.arange(mod) + patch_id_offset
        requested_patch_ids_ = np.intersect1d(patch_ids_, all_requested_patch_ids)
        requested_indicies = requested_patch_ids_ - patch_id_offset
        patches = _crop_to_numpy(patchfile, requested_indicies)
        for idx, patch in zip(requested_patch_ids_, patches):
            all_patches[idx] = patch

    print('read %d patches ' % (len(all_patches)))
    #patches_list += [patches]

    #all_patches = np.concatenate(patches_list, axis=0)

    matching_patches1 = [all_patches[idx1] for idx1, idx2 in match_pairs]
    matching_patches2 = [all_patches[idx2] for idx1, idx2 in match_pairs]
    nonmatching_patches1 = [all_patches[idx1] for idx1, idx2 in non_match_pairs]
    nonmatching_patches2 = [all_patches[idx2] for idx1, idx2 in non_match_pairs]

    labels = np.array(([True] * len(matching_patches1)) + ([False] * len(nonmatching_patches1)))
    warped_patch1_list = matching_patches1 + nonmatching_patches1
    warped_patch2_list = matching_patches2 + nonmatching_patches2

    img_list = ut.flatten(list(zip(warped_patch1_list, warped_patch2_list)))
    data = np.array(img_list)
    del img_list
    #data_per_label = 2
    assert labels.shape[0] == data.shape[0] // 2
    return data, labels


def get_juction_dpath():
    junction_dpath = ut.ensure_app_resource_dir('ibeis_cnn', 'training_junction')
    return junction_dpath


def register_training_dpath(training_dpath, alias_key=None):
    junction_dpath = get_juction_dpath()
    training_dname = basename(training_dpath)
    if alias_key is not None:
        # hack for a bit more readable pathname
        prefix = alias_key.split(';')[0].replace(' ', '')
        training_dname = prefix + '_' + training_dname
    training_dlink = join(junction_dpath, training_dname)
    ut.symlink(training_dpath, training_dlink)


def convert_category_to_siam_data(category_data, category_labels):
    # CONVERT CATEGORY LABELS TO PAIR LABELS
    # Make genuine imposter pairs
    import vtool as vt
    unique_labels, groupxs_list = vt.group_indices(category_labels)

    num_categories = len(unique_labels)

    num_geninue  = 10000 * num_categories
    num_imposter = 10000 * num_categories

    num_gen_per_category = int(num_geninue / len(unique_labels))
    num_imp_per_category = int(num_imposter / len(unique_labels))

    np.random.seed(0)
    groupxs = groupxs_list[0]

    def find_fix_flags(pairxs):
        is_dup = vt.nonunique_row_flags(pairxs)
        is_eye = pairxs.T[0] == pairxs.T[1]
        needs_fix = np.logical_or(is_dup, is_eye)
        #print(pairxs[needs_fix])
        return needs_fix

    def swap_undirected(pairxs):
        """ ensure left indicies are lower """
        needs_swap = pairxs.T[0] > pairxs.T[1]
        arr = pairxs[needs_swap]
        tmp = arr.T[0].copy()
        arr.T[0, :] = arr.T[1]
        arr.T[1, :] = tmp
        pairxs[needs_swap] = arr
        return pairxs

    def sample_pairs(left_list, right_list, size):
        # Sample initial random left and right indices
        _index1 = np.random.choice(left_list, size=size, replace=True)
        _index2 = np.random.choice(right_list, size=size, replace=True)
        # stack
        _pairxs = np.vstack((_index1, _index2)).T
        # make undiractional
        _pairxs = swap_undirected(_pairxs)
        # iterate until feasible
        needs_fix = find_fix_flags(_pairxs)
        while np.any(needs_fix):
            num_fix = needs_fix.sum()
            print('fixing: %d' % num_fix)
            _pairxs.T[1][needs_fix] = np.random.choice(right_list, size=num_fix, replace=True)
            _pairxs = swap_undirected(_pairxs)
            needs_fix = find_fix_flags(_pairxs)
        return _pairxs

    print('sampling genuine pairs')
    genuine_pairx_list = []
    for groupxs in groupxs_list:
        left_list = groupxs
        right_list = groupxs
        size = num_gen_per_category
        _pairxs = sample_pairs(left_list, right_list, size)
        genuine_pairx_list.extend(_pairxs.tolist())

    print('sampling imposter pairs')
    imposter_pairx_list = []
    for index in range(len(groupxs_list)):
        # Pick random pairs of false matches
        groupxs = groupxs_list[index]
        bar_groupxs = np.hstack(groupxs_list[:index] + groupxs_list[index + 1:])
        left_list = groupxs
        right_list = bar_groupxs
        size = num_imp_per_category
        _pairxs = sample_pairs(left_list, right_list, size)
        imposter_pairx_list.extend(_pairxs.tolist())

    # We might have added duplicate imposters, just remove them for now
    imposter_pairx_list = ut.list_take(imposter_pairx_list, vt.unique_row_indexes(np.array(imposter_pairx_list)))

    # structure data for output
    flat_data_pairxs = np.array(genuine_pairx_list + imposter_pairx_list)
    assert np.all(flat_data_pairxs.T[0] < flat_data_pairxs.T[1])
    assert find_fix_flags(flat_data_pairxs).sum() == 0
    # TODO: batch should use indicies into data
    flat_index_list = np.array(ut.flatten(list(zip(flat_data_pairxs.T[0], flat_data_pairxs.T[1]))))
    data = np.array(category_data.take(flat_index_list, axis=0))
    labels = np.array([True] * len(genuine_pairx_list) + [False] * len(imposter_pairx_list))
    return data, labels


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis_cnn.ingest_helpers
        python -m ibeis_cnn.ingest_helpers --allexamples
        python -m ibeis_cnn.ingest_helpers --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
