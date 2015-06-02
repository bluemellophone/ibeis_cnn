# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from ibeis_cnn import utils
from ibeis_cnn import ingest_helpers
from ibeis_cnn import ingest_ibeis
from os.path import join
import utool as ut
print, rrr, profile = ut.inject2(__name__, '[ibeis_cnn.ingest]')


def grab_cached_mist_data(nets_dir):
    import numpy as np
    data_fpath = join(nets_dir, 'mnist_data.cPkl')
    labels_fpath = join(nets_dir, 'mnist_labels.cPkl')
    if not ut.checkpath(data_fpath):
        train_imgs_fpath = ut.grab_zipped_url(
            'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz')
        train_lbls_fpath = ut.grab_zipped_url(
            'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz')
        test_imgs_fpath = ut.grab_zipped_url(
            'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz')
        test_lbls_fpath = ut.grab_zipped_url(
            'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz')

        train_images, train_labels = ingest_helpers.open_mnist_files(
            train_lbls_fpath, train_imgs_fpath)
        test_images, test_labels = ingest_helpers.open_mnist_files(
            test_lbls_fpath, test_imgs_fpath)
        data = np.vstack((train_images, test_images))
        labels = np.append(train_labels, test_labels)
        utils.write_data_and_labels(data, labels, data_fpath, labels_fpath)
    return data_fpath, labels_fpath


def grab_cached_liberty_data(nets_dir):
    """
    References:
        http://www.cs.ubc.ca/~mbrown/patchdata/patchdata.html
        https://github.com/osdf/datasets/blob/master/patchdata/dataset.py

    Notes:
        "info.txt" contains the match information Each row of info.txt
        corresponds corresponds to a separate patch, with the patches ordered
        from left to right and top to bottom in each bitmap image.

        3 types of metadata files

        info.txt - contains patch ids that correspond with the order of patches
          in the bmp images
          In the format:
              pointid, unused

        interest.txt -
            interest points corresponding to patches with patchids
            has same number of rows as info.txt
            In the format:
                reference image id, x, y, orientation, scale (in log2 units)

        m50_<d>_<d>_0.txt -
             matches files
             patchID1  3DpointID1  unused1  patchID2  3DpointID2  unused2

    CommandLine:
        python -m ibeis_cnn.ingest_data --test-grab_cached_liberty_data --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_cnn.ingest_data import *  # NOQA
        >>> nets_dir = ut.get_app_resource_dir('ibeis_cnn', 'training', 'liberty')
        >>> data_fpath, labels_fpath, data_shape = grab_cached_liberty_data(nets_dir)
        >>> ut.quit_if_noshow()
        >>> from ibeis_cnn import ibsplugin
        >>> #ibsplugin.rrr()
        >>> flat_metadata = {}
        >>> data, labels = utils.load(data_fpath, labels_fpath)
        >>> warped_patch1_list = data[::2]
        >>> warped_patch2_list = data[1::2]
        >>> ibsplugin.interact_view_patches(labels, warped_patch1_list, warped_patch2_list, flat_metadata, rand=True)
        >>> print(result)
        >>> ut.show_if_requested()
    """
    liberty_dog_fpath = ut.grab_zipped_url(
        'http://www.cs.ubc.ca/~mbrown/patchdata/liberty.zip')
    #liberty_harris_fpath = ut.grab_zipped_url(  # NOQA
    #    'http://www.cs.ubc.ca/~mbrown/patchdata/liberty_harris.zip')
    ds_path = liberty_dog_fpath

    #nets_dir = '.'
    #pairs = 500
    #pairs = 100000
    pairs = 250000
    #pairs = 50000
    data_fpath = join(nets_dir, 'liberty_data_%d.cPkl' % (pairs,))
    labels_fpath = join(nets_dir, 'liberty_labels_%d.cPkl' % (pairs,))
    if not ut.checkpath(data_fpath):
        data, labels = ingest_helpers.extract_liberty_style_patches(ds_path, pairs)
        #data = np.vstack((train_images, test_images))
        #labels = np.append(train_labels, test_labels)
        utils.write_data_and_labels(data, labels, data_fpath, labels_fpath)
    data_shape = (64, 64, 1)
    return data_fpath, labels_fpath, data_shape


def get_patchmetric_training_fpaths(ibs, **kwargs):
    """
    CommandLine:
        python -m ibeis_cnn.ingest_data --test-get_patchmetric_training_fpaths --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.ingest_data import *  # NOQA
        >>> from ibeis_cnn import ingest_ibeis
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='PZ_MTEST')
        >>> kwargs = ut.argparse_dict({'max_examples': None, 'num_top': 3})
        >>> (data_fpath, labels_fpath, training_dpath, data_shape) = get_patchmetric_training_fpaths(ibs, **kwargs)
        >>> ut.quit_if_noshow()
        >>> ingest_ibeis.interact_view_data_fpath_patches(data_fpath, labels_fpath, {})
    """
    print('\n\n[get_patchmetric_training_fpaths] START')
    max_examples = kwargs.get('max_examples', None)
    num_top = kwargs.get('num_top', None)
    # Get training data pairs
    patchmatch_tup = ingest_ibeis.get_aidpairs_and_matches(ibs, max_examples, num_top)
    aid1_list, aid2_list, kpts1_m_list, kpts2_m_list, fm_list, metadata_lists = patchmatch_tup
    # Extract and cache the data
    data_fpath, labels_fpath, training_dpath, data_shape = ingest_ibeis.cached_patchmetric_training_data_fpaths(
        ibs, aid1_list, aid2_list, kpts1_m_list, kpts2_m_list, fm_list, metadata_lists)
    print('\n[get_patchmetric_training_fpaths] FINISH\n\n')
    return data_fpath, labels_fpath, training_dpath, data_shape


def testdata_patchmatch():
    """
        >>> from ibeis_cnn.ingest_data import *  # NOQA
    """
    with ut.Indenter('[LOAD IBEIS DB]'):
        import ibeis
        ibs = ibeis.opendb(defaultdb='PZ_MTEST')

    with ut.Indenter('[ENSURE TRAINING DATA]'):
        pathtup = get_patchmetric_training_fpaths(ibs, max_examples=5)
        data_fpath, labels_fpath, training_dpath, data_shape = pathtup
    data_cv2, labels = utils.load(data_fpath, labels_fpath)
    data = utils.convert_cv2_images_to_theano_images(data_cv2)
    return data, labels


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis_cnn.ingest_data
        python -m ibeis_cnn.ingest_data --allexamples
        python -m ibeis_cnn.ingest_data --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
