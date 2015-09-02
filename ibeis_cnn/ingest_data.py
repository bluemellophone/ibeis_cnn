# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from ibeis_cnn import utils
from ibeis_cnn import ingest_helpers
from ibeis_cnn import ingest_ibeis
from ibeis_cnn.dataset import DataSet
from os.path import join, basename, splitext
import utool as ut
print, rrr, profile = ut.inject2(__name__, '[ibeis_cnn.ingest]')


def testdata_dataset():
    dataset = get_ibeis_siam_dataset(max_examples=5, controlled=False)
    return dataset


def testdata_patchmatch():
    """
        >>> from ibeis_cnn.ingest_data import *  # NOQA
    """
    dataset = get_ibeis_siam_dataset(max_examples=5)
    data_fpath = dataset.data_fpath
    labels_fpath = dataset.labels_fpath
    data_cv2, labels = utils.load(data_fpath, labels_fpath)
    data = utils.convert_cv2_images_to_theano_images(data_cv2)
    return data, labels


def testdata_patchmatch2():
    """
        >>> from ibeis_cnn.ingest_data import *  # NOQA
    """
    dataset = get_ibeis_siam_dataset(max_examples=5)
    data_fpath = dataset.data_fpath
    labels_fpath = dataset.labels_fpath
    data, labels = utils.load(data_fpath, labels_fpath)
    return data, labels


def get_extern_training_dpath(alias_key):
    return DataSet.from_alias_key(alias_key).training_dpath


def view_training_directories():
    r"""
    CommandLine:
        python -m ibeis_cnn.ingest_data --test-view_training_directories

    Example:
        >>> # UTILITY_SCRIPT
        >>> from ibeis_cnn.ingest_data import *  # NOQA
        >>> result = view_training_directories()
        >>> print(result)
    """
    ut.vd(ingest_ibeis.get_juction_dpath())


def merge_datasets(dataset_list):
    """
    Merges a list of dataset objects into a single combined dataset.
    """

    def consensus_check_factory():
        """
        Returns a temporary function used to check that all incoming values
        with the same key are consistent
        """
        from collections import defaultdict
        past_values = defaultdict(lambda: None)
        def consensus_check(value, key):
            assert past_values[key] is None or past_values[key] == value, (
                'key=%r with value=%r does not agree with past_value=%r' %
                (key, value, past_values[key]))
            past_values[key] = value
            return value
        return consensus_check

    total_num_labels = 0
    total_num_data = 0

    input_alias_list = [dataset.alias_key for dataset in dataset_list]

    alias_key = 'combo_' + ut.hashstr27(repr(input_alias_list), hashlen=8)
    training_dpath = ut.ensure_app_resource_dir('ibeis_cnn', 'training', alias_key)
    data_fpath = ut.unixjoin(training_dpath, alias_key + '_data.hdf5')
    labels_fpath = ut.unixjoin(training_dpath, alias_key + '_labels.hdf5')

    try:
        # Try and short circut cached loading
        merged_dataset = DataSet.from_alias_key(alias_key)
        return merged_dataset
    except Exception as ex:
        ut.printex(ex, 'alias definitions have changed. alias_key=%r' % (alias_key,), iswarning=True)

    # Build the dataset
    consensus_check = consensus_check_factory()

    for dataset in dataset_list:
        print(ut.get_file_nBytes_str(dataset.data_fpath))
        print(dataset.data_fpath_dict['all'])
        print(dataset.num_labels)
        print(dataset.data_per_label)
        total_num_labels += dataset.num_labels
        total_num_data += (dataset.data_per_label * dataset.num_labels)
        # check that all data_dims agree
        data_shape = consensus_check(dataset.data_shape, 'data_shape')
        data_per_label = consensus_check(dataset.data_per_label, 'data_per_label')

    # hack record this
    import numpy as np
    data_dtype = np.uint8
    label_dtype = np.int32
    data = np.empty((total_num_data,) + data_shape, dtype=data_dtype)
    labels = np.empty(total_num_labels, dtype=label_dtype)

    #def iterable_assignment():
    #    pass
    data_left = 0
    data_right = None
    labels_left = 0
    labels_right = None
    for dataset in ut.ProgressIter(dataset_list, lbl='combining datasets', freq=1):
        X_all, y_all = dataset.load_subset('all')
        labels_right = labels_left + y_all.shape[0]
        data_right = data_left + X_all.shape[0]
        data[data_left:data_right] = X_all
        labels[labels_left:labels_right] = y_all
        data_left = data_right
        labels_left = labels_right

    utils.write_data_and_labels(data, labels, data_fpath, labels_fpath)

    labels = ut.load_data(labels_fpath)
    num_labels = len(labels)

    merged_dataset = DataSet.new_training_set(
        alias_key=alias_key,
        data_fpath=data_fpath,
        labels_fpath=labels_fpath,
        training_dpath=training_dpath,
        data_shape=data_shape,
        data_per_label=data_per_label,
        output_dims=1,
        num_labels=num_labels,
    )
    return merged_dataset


def grab_siam_dataset(ds_tag=None):
    """
    Will build the dataset using the command line if it doesnt exist

    CommandLine:
        python -m ibeis_cnn.ingest_data --test-grab_siam_dataset --db PZ_MTEST --show
        python -m ibeis_cnn.ingest_data --test-grab_siam_dataset --db NNP_Master3 --show
        python -m ibeis_cnn.ingest_data --test-grab_siam_dataset --db PZ_Master0 --show
        python -m ibeis_cnn.ingest_data --test-grab_siam_dataset --db mnist --show
        python -m ibeis_cnn.ingest_data --test-grab_siam_dataset --db liberty --show

        python -m ibeis_cnn.ingest_data --test-grab_siam_dataset --db PZ_MTEST

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_cnn.ingest_data import *  # NOQA
        >>> ds_tag = None
        >>> dataset = grab_siam_dataset(ds_tag=ds_tag)
        >>> ut.quit_if_noshow()
        >>> from ibeis_cnn import draw_results
        >>> #ibsplugin.rrr()
        >>> flat_metadata = {}
        >>> data, labels = dataset.load_subset('all')
        >>> ut.quit_if_noshow()
        >>> warped_patch1_list = data[::2]
        >>> warped_patch2_list = data[1::2]
        >>> draw_results.interact_patches(labels, warped_patch1_list, warped_patch2_list, flat_metadata, sortby='rand')
        >>> print(result)
        >>> ut.show_if_requested()
    """
    if ds_tag is not None:
        try:
            return DataSet.from_alias_key(ds_tag)
        except Exception as ex:
            ut.printex(ex, 'Could not resolve alias. Need to rebuild dataset', keys=['ds_tag'])
            raise

    dbname = ut.get_argval('--db')
    if dbname == 'liberty':
        pairs = 250000
        dataset = grab_liberty_siam_dataset(pairs)
    elif dbname == 'mnist':
        dataset = grab_mnist_siam_dataset()
    else:
        dataset = get_ibeis_siam_dataset()
    return dataset


def grab_mnist_category_dataset():
    import numpy as np
    training_dpath = ut.get_app_resource_dir('ibeis_cnn', 'training', 'mnist')
    ut.ensuredir(training_dpath)
    data_fpath = join(training_dpath, 'mnist_data.hdf5')
    labels_fpath = join(training_dpath, 'mnist_labels.hdf5')
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

    alias_key = 'mnist'

    # hack for caching num_labels
    labels = ut.load_data(labels_fpath)
    num_labels = len(labels)

    dataset = DataSet.new_training_set(
        alias_key=alias_key,
        data_fpath=data_fpath,
        labels_fpath=labels_fpath,
        training_dpath=training_dpath,
        data_per_label=1,
        data_shape=(28, 28, 1),
        output_dims=10,
        num_labels=num_labels,
    )
    return dataset


def grab_mnist_siam_dataset():
    r"""

    CommandLine:
        python -m ibeis_cnn.ingest_data --test-grab_mnist_siam_dataset --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_cnn.ingest_data import *  # NOQA
        >>> dataset = grab_mnist_siam_dataset()
        >>> ut.quit_if_noshow()
        >>> from ibeis_cnn import draw_results
        >>> #ibsplugin.rrr()
        >>> flat_metadata = {}
        >>> data, labels = dataset.load_subset('all')
        >>> ut.quit_if_noshow()
        >>> warped_patch1_list = data[::2]
        >>> warped_patch2_list = data[1::2]
        >>> draw_results.interact_patches(labels, warped_patch1_list, warped_patch2_list, flat_metadata, sortby='rand')
        >>> print(result)
        >>> ut.show_if_requested()
    """
    import numpy as np
    alias_key = 'mnist_pairs'

    training_dpath = ut.get_app_resource_dir('ibeis_cnn', 'training', alias_key)
    ut.ensuredir(training_dpath)

    data_fpath = join(training_dpath, alias_key + '_data.hdf5')
    labels_fpath = join(training_dpath, alias_key + '_labels.hdf5')
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

        category_data = np.vstack((train_images, test_images))
        category_labels = np.append(train_labels, test_labels)
        data, labels = ingest_helpers.convert_category_to_siam_data(category_data, category_labels)
        #metadata = {
        #    'flat_index_list': flat_index_list,
        #    'category_labels': category_labels.take(flat_index_list)
        #}
        utils.write_data_and_labels(data, labels, data_fpath, labels_fpath)
        #metadata = {}

    # hack for caching num_labels
    labels = ut.load_data(labels_fpath)
    num_labels = len(labels)

    dataset = DataSet.new_training_set(
        alias_key=alias_key,
        data_fpath=data_fpath,
        labels_fpath=labels_fpath,
        training_dpath=training_dpath,
        data_per_label=2,
        data_shape=(28, 28, 1),
        output_dims=1,
        num_labels=num_labels,
    )
    return dataset


def grab_liberty_siam_dataset(pairs=250000):
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
        python -m ibeis_cnn.ingest_data --test-grab_liberty_siam_dataset --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_cnn.ingest_data import *  # NOQA
        >>> pairs = 500
        >>> dataset = grab_liberty_siam_dataset(pairs)
        >>> ut.quit_if_noshow()
        >>> from ibeis_cnn import draw_results
        >>> #ibsplugin.rrr()
        >>> flat_metadata = {}
        >>> data, labels = dataset.load_subset('all')
        >>> ut.quit_if_noshow()
        >>> warped_patch1_list = data[::2]
        >>> warped_patch2_list = data[1::2]
        >>> draw_results.interact_patches(labels, warped_patch1_list, warped_patch2_list, flat_metadata, sortby='rand')
        >>> print(result)
        >>> ut.show_if_requested()
    """
    datakw = {
        'detector': 'dog',
        'pairs': pairs,
    }

    assert datakw['detector'] in ['dog', 'harris']
    assert pairs in [500, 50000, 100000, 250000]

    liberty_urls = {
        'dog': 'http://www.cs.ubc.ca/~mbrown/patchdata/liberty.zip',
        'harris': 'http://www.cs.ubc.ca/~mbrown/patchdata/liberty_harris.zip',
    }
    url = liberty_urls[datakw['detector']]
    ds_path = ut.grab_zipped_url(url)

    ds_name = splitext(basename(ds_path))[0]
    alias_key = 'liberty;' + ut.dict_str(datakw, nl=False, explicit=True)
    cfgstr = ','.join([str(val) for key, val in ut.iteritems_sorted(datakw)])

    training_dpath = ut.ensure_app_resource_dir('ibeis_cnn', 'training', ds_name)
    if ut.get_argflag('--vtd'):
        ut.vd(training_dpath)
    ut.ensuredir(training_dpath)

    data_fpath = join(training_dpath, 'liberty_data_' + cfgstr + '.hdf5')
    labels_fpath = join(training_dpath, 'liberty_labels_' + cfgstr  + '.hdf5')

    if not ut.checkpath(data_fpath, verbose=True):
        data, labels = ingest_helpers.extract_liberty_style_patches(ds_path, pairs)
        utils.write_data_and_labels(data, labels, data_fpath, labels_fpath)

    # hack for caching num_labels
    labels = ut.load_data(labels_fpath)
    num_labels = len(labels)

    dataset = DataSet.new_training_set(
        alias_key=alias_key,
        data_fpath=data_fpath,
        labels_fpath=labels_fpath,
        training_dpath=training_dpath,
        data_shape=(64, 64, 1),
        data_per_label=2,
        output_dims=1,
        num_labels=num_labels,
    )
    return dataset


def get_ibeis_siam_dataset(**kwargs):
    """
    CommandLine:
        python -m ibeis_cnn.ingest_data --test-get_ibeis_siam_dataset --show
        python -m ibeis_cnn.ingest_data --test-get_ibeis_siam_dataset --show --db PZ_Master0

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_cnn.ingest_data import *  # NOQA
        >>> from ibeis_cnn import draw_results
        >>> import ibeis
        >>> kwargs = {}  # ut.argparse_dict({'max_examples': None, 'num_top': 3})
        >>> dataset = get_ibeis_siam_dataset(**kwargs)
        >>> data_fpath = dataset.data_fpath
        >>> labels_fpath = dataset.labels_fpath
        >>> ut.quit_if_noshow()
        >>> draw_results.interact_siamese_data_fpath_patches(data_fpath, labels_fpath, {})
        >>> ut.show_if_requested()
    """
    datakw = ut.argparse_dict(
        {
            #'db': 'PZ_MTEST',
            'max_examples': None,
            #'num_top': 3,
            'num_top': None,
            'controlled': True,
            'colorspace': 'gray',
        }, verbose=True)
    with ut.Indenter('[LOAD IBEIS DB]'):
        import ibeis
        dbname = ut.get_argval('--db', default='PZ_MTEST')
        ibs = ibeis.opendb(dbname=dbname, defaultdb='PZ_MTEST')

    # Nets dir is the root dir for all training on this data
    training_dpath = ibs.get_neuralnet_dir()
    ut.ensuredir(training_dpath)
    datakw.update(kwargs)
    print('\n\n[get_ibeis_siam_dataset] START')
    #log_dir = join(training_dpath, 'logs')
    #ut.start_logging(log_dir=log_dir)

    alias_key = ibs.get_dbname() + ';' + ut.dict_str(datakw, nl=False, explicit=True)
    try:
        # Try and short circut cached loading
        dataset = DataSet.from_alias_key(alias_key)
        return dataset
    except Exception as ex:
        ut.printex(ex, 'alias definitions have changed. alias_key=%r' % (alias_key,), iswarning=True)

    with ut.Indenter('[CHECKDATA]'):
        # Get training data pairs
        colorspace = datakw.pop('colorspace')
        patchmatch_tup = ingest_ibeis.get_aidpairs_and_matches(ibs, **datakw)
        aid1_list, aid2_list, kpts1_m_list, kpts2_m_list, fm_list, metadata_lists = patchmatch_tup
        # Extract and cache the data
        # TODO: metadata
        data_fpath, labels_fpath, training_dpath, data_shape = ingest_ibeis.cached_patchmetric_training_data_fpaths(
            ibs, aid1_list, aid2_list, kpts1_m_list, kpts2_m_list, fm_list, metadata_lists, colorspace=colorspace)
        print('\n[get_ibeis_siam_dataset] FINISH\n\n')

    # hack for caching num_labels
    labels = ut.load_data(labels_fpath)
    num_labels = len(labels)

    dataset = DataSet.new_training_set(
        alias_key=alias_key,
        data_fpath=data_fpath,
        labels_fpath=labels_fpath,
        training_dpath=training_dpath,
        data_shape=data_shape,
        data_per_label=2,
        output_dims=1,
        num_labels=num_labels,
    )
    return dataset


def get_numpy_dataset(data_fpath, labels_fpath, training_dpath):
    """
    """
    # hack for caching num_labels
    data = ut.load_data(data_fpath)
    data_shape = data.shape
    labels = ut.load_data(labels_fpath)
    num_labels = len(labels)

    alias_key = 'temp'
    ut.ensuredir(training_dpath)

    dataset = DataSet.new_training_set(
        alias_key=alias_key,
        data_fpath=data_fpath,
        labels_fpath=labels_fpath,
        training_dpath=training_dpath,
        data_shape=data_shape,
        data_per_label=1,
        output_dims=1,
        num_labels=num_labels,
    )
    return dataset


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
