# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from ibeis_cnn import utils
from ibeis_cnn import ingest_helpers
from ibeis_cnn import ingest_ibeis
from os.path import join
import utool as ut
print, rrr, profile = ut.inject2(__name__, '[ibeis_cnn.ingest]')


def grab_cached_mist_data():
    import numpy as np
    training_dpath = ut.get_app_resource_dir('ibeis_cnn', 'training', 'mnist')
    ut.ensuredir(training_dpath)
    data_fpath = join(training_dpath, 'mnist_data.cPkl')
    labels_fpath = join(training_dpath, 'mnist_labels.cPkl')
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
    data_shape = (28, 28, 1)
    output_dims = 10
    return data_fpath, labels_fpath, training_dpath, data_shape, output_dims


def grab_cached_liberty_data(pairs=250000):
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
        >>> pairs = 500
        >>> data_fpath, labels_fpath, data_shape = grab_cached_liberty_data(pairs)
        >>> ut.quit_if_noshow()
        >>> from ibeis_cnn import ingest_ibeis
        >>> #ibsplugin.rrr()
        >>> flat_metadata = {}
        >>> data, labels = utils.load(data_fpath, labels_fpath)
        >>> warped_patch1_list = data[::2]
        >>> warped_patch2_list = data[1::2]
        >>> ingest_ibeis.interact_view_patches(labels, warped_patch1_list, warped_patch2_list, flat_metadata, rand=True)
        >>> print(result)
        >>> ut.show_if_requested()
    """
    training_dpath = ut.ensure_app_resource_dir('ibeis_cnn', 'training', 'liberty')
    if ut.get_argflag('--vtd'):
        ut.vd(training_dpath)
    ut.ensuredir(training_dpath)

    liberty_dog_fpath = ut.grab_zipped_url(
        'http://www.cs.ubc.ca/~mbrown/patchdata/liberty.zip')
    #liberty_harris_fpath = ut.grab_zipped_url(  # NOQA
    #    'http://www.cs.ubc.ca/~mbrown/patchdata/liberty_harris.zip')
    ds_path = liberty_dog_fpath

    #training_dpath = '.'
    assert pairs in [500, 50000, 100000, 250000]
    #pairs = 500
    #pairs = 100000
    #pairs = 250000
    #pairs = 50000
    data_fpath = join(training_dpath, 'liberty_data_%d.cPkl' % (pairs,))
    labels_fpath = join(training_dpath, 'liberty_labels_%d.cPkl' % (pairs,))
    if not ut.checkpath(data_fpath, verbose=True):
        data, labels = ingest_helpers.extract_liberty_style_patches(ds_path, pairs)
        #data = np.vstack((train_images, test_images))
        #labels = np.append(train_labels, test_labels)
        utils.write_data_and_labels(data, labels, data_fpath, labels_fpath)
    data_shape = (64, 64, 1)
    return data_fpath, labels_fpath, training_dpath, data_shape


def get_patchmetric_training_fpaths(**kwargs):
    """
    CommandLine:
        python -m ibeis_cnn.ingest_data --test-get_patchmetric_training_fpaths --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.ingest_data import *  # NOQA
        >>> from ibeis_cnn import ingest_ibeis
        >>> import ibeis
        >>> kwargs = {}  # ut.argparse_dict({'max_examples': None, 'num_top': 3})
        >>> (data_fpath, labels_fpath, training_dpath, data_shape) = get_patchmetric_training_fpaths(**kwargs)
        >>> ut.quit_if_noshow()
        >>> ingest_ibeis.interact_view_data_fpath_patches(data_fpath, labels_fpath, {})
    """
    #ut.embed()
    datakw = ut.argparse_dict(
        {
            #'db': 'PZ_MTEST',
            'max_examples': None,
            'num-top': 3,
        }
    )
    with ut.Indenter('[LOAD IBEIS DB]'):
        import ibeis
        ibs = ibeis.opendb(defaultdb='PZ_MTEST')
    # Nets dir is the root dir for all training
    training_dpath = ibs.get_neuralnet_dir()
    ut.ensuredir(training_dpath)
    datakw.update(kwargs)
    print('\n\n[get_patchmetric_training_fpaths] START')
    max_examples = datakw.get('max_examples', None)
    num_top = datakw.get('num_top', None)
    #log_dir = join(training_dpath, 'logs')
    #ut.start_logging(log_dir=log_dir)

    with ut.Indenter('[CHECKDATA]'):
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
    pathtup = get_patchmetric_training_fpaths(max_examples=5)
    data_fpath, labels_fpath, training_dpath, data_shape = pathtup
    data_cv2, labels = utils.load(data_fpath, labels_fpath)
    data = utils.convert_cv2_images_to_theano_images(data_cv2)
    return data, labels


def testdata_patchmatch2():
    """
        >>> from ibeis_cnn.ingest_data import *  # NOQA
    """
    pathtup = get_patchmetric_training_fpaths(max_examples=5)
    data_fpath, labels_fpath, training_dpath, data_shape = pathtup
    data, labels = utils.load(data_fpath, labels_fpath)
    return data, labels


def ondisk_data_split(model, data_fpath, labels_fpath, split_names=['train', 'valid', 'test'], fraction_list=[.2, .1]):
    """
    splits into train / validation datasets on disk

    # TODO: metadata fpath

    split_names=['train', 'valid', 'test'], fraction_list=[.2, .1]
    """
    from os.path import dirname, join, exists, basename
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

    split_data_fpaths = [join(splitdir, name + '_data_%.2f_' % (frac,) + hashstr_ + '.pkl') for name, frac in zip(split_names, totalfrac_list)]
    split_labels_fpaths = [join(splitdir, name + '_labels_%.2f_' % (frac,) + hashstr_ + '.pkl') for name, frac in zip(split_names, totalfrac_list)]

    is_cache_hit = (all(map(exists, split_data_fpaths)) and all(map(exists, split_labels_fpaths)))

    data_per_label = model.data_per_label

    if not is_cache_hit:
        print('Writing data splits')
        X_left, y_left = utils.load(data_fpath, labels_fpath)
        _iter = zip(fraction_list, split_data_fpaths, split_labels_fpaths)
        for fraction, x_fpath, y_fpath in _iter:
            _tup = utils.train_test_split(X_left, y_left, eval_size=fraction,
                                          data_per_label=data_per_label,
                                          shuffle=True)
            X_left, y_left, X_right, y_right = _tup
            utils.write_data_and_labels(X_left, y_left, x_fpath, y_fpath)
        x_fpath  = split_data_fpaths[-1]
        y_fpath = split_labels_fpaths[-1]
        utils.write_data_and_labels(X_right, y_right, x_fpath, y_fpath)

    data_fpath_dict = dict(zip(split_names, split_data_fpaths))
    label_fpath_dict = dict(zip(split_names, split_labels_fpaths))

    label_fpath_dict['all'] = labels_fpath
    data_fpath_dict['all'] = data_fpath
    return data_fpath_dict, label_fpath_dict


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
