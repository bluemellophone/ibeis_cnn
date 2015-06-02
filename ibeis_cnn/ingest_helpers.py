# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from os.path import join
import utool as ut
print, rrr, profile = ut.inject2(__name__, '[ibeis_cnn.ingest_helpers]')


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

    def _crop_to_numpy(patchfile):
        """
        Convert _patchfile_ to a numpy array with patches per row.
        A _patchfile_ is a .bmp file.
        """
        pil_img = Image.open(patchfile)
        ptch_list = [
            pil_img.crop(
                (col * patch_x, row * patch_y, (col + 1) * patch_x, (row + 1) * patch_y))
            for row, col in product(range(rows), range(cols))
        ]
        patches = np.array([np.array(ptch) for ptch in ptch_list])
        pil_img.close()
        return patches

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

    per_bmp = rows * cols
    total_num_patches = _available_patches(ds_path)
    bmps, mod = divmod(total_num_patches, per_bmp)

    patchfile_list = [join(ds_path, ''.join(["patches", str(i).zfill(4), '.bmp']))
                      for i in range(bmps)]

    # Build matching labels
    match_pairs, non_match_pairs, patch_ids = matches(ds_path, pairs)
    assert max(patch_ids) <= total_num_patches

    # Read all patches out of the bmp file store
    patches_list = [_crop_to_numpy(patchfile) for patchfile in ut.ProgressIter(patchfile_list, lbl='Reading Patches')]
    # Read the last patches
    if mod > 0:
        patchfile = join(
            ds_path, ''.join(["patches", str(bmps).zfill(4), ".bmp"]))
        patches = _crop_to_numpy(patchfile)[:mod]
        patches_list += [patches]

    all_patches = np.concatenate(patches_list, axis=0)

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
