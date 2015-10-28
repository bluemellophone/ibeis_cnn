# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from ibeis_cnn import utils
from os.path import join, basename, exists, dirname, splitext
import six
import numpy as np
import utool as ut
print, rrr, profile = ut.inject2(__name__, '[ibeis_cnn.datset]')


#NOCACHE_ALIAS = True
NOCACHE_ALIAS = ut.get_argflag('--nocache-alias')
NOCACHE_DATA_SPLIT = ut.get_argflag('--nocache-datasplit')


class DummyDataSet(object):
    def __init__(dataset):
        dataset.data_fpath_dict = None
        dataset.label_fpath_dict = None


@six.add_metaclass(ut.ReloadingMetaclass)
class DataSet(object):
    """
    helper class for managing dataset paths and general metadata

    SeeAlso:
        python -m ibeis_cnn.ingest_data --test-get_ibeis_part_siam_dataset --show

    """
    def __init__(dataset, alias_key, training_dpath, data_fpath, labels_fpath,
                 metadata_fpath, data_per_label, data_shape, output_dims, num_labels):
        # Constructor args is primary data
        key_list = ut.get_func_argspec(dataset.__init__).args[1:]
        r"""
        Gen Explicit:
            import utool, ibeis_cnn.dataset
            key_list = ut.get_func_argspec(ibeis_cnn.dataset.DataSet.__init__).args[1:]
            autogen_block = ut.indent('\n'.join(['dataset.{key} = {key}'.format(key=key) for key in key_list]), ' ' * 12)
            ut.inject_python_code2(ibeis_cnn.dataset.__file__, autogen_block, 'AUTOGEN_INIT')
            #ut.copy_text_to_clipboard('\n' + autogen_block)

        Gen Explicit Commandline:
            python -Bc "import utool, ibeis_cnn.dataset; utool.inject_python_code2(utool.get_modpath_from_modname(ibeis_cnn.dataset.__name__), utool.indent('\n'.join(['dataset.{key} = {key}'.format(key=key) for key in utool.get_func_argspec(ibeis_cnn.dataset.DataSet.__init__).args[1:]]), ' ' * 12), 'AUTOGEN_INIT')"
        """
        EXPLICIT = True
        if EXPLICIT:
            pass
            # <AUTOGEN_INIT>
            dataset.alias_key = alias_key
            dataset.training_dpath = training_dpath
            dataset.data_fpath = data_fpath
            dataset.labels_fpath = labels_fpath
            dataset.metadata_fpath = metadata_fpath
            dataset.data_per_label = data_per_label
            dataset.data_shape = data_shape
            dataset.output_dims = output_dims
            dataset.num_labels = num_labels
            # </AUTOGEN_INIT>
        else:
            locals_ = locals()
            for key in key_list:
                setattr(dataset, key, locals_[key])
        # Define auxillary data
        dataset.build_auxillary_data()
        # Hacky dictionary for custom things
        # Probably should be refactored
        dataset._lazy_cache = ut.LazyDict()

    def hasprop(dataset, key):
        return key in dataset._lazy_cache.keys()

    def getprop(dataset, key):
        return dataset._lazy_cache[key]

    def setprop(dataset, key, val):
        dataset._lazy_cache[key] = val

    def build_auxillary_data(dataset):
        # Make test train validatation sets
        named_split_fpath_dict = ondisk_data_split(
            dataset.data_fpath,
            dataset.labels_fpath,
            dataset.metadata_fpath,
            dataset.data_per_label,
            split_names=['train', 'valid', 'test'],
            fraction_list=[.2, .1])
        dataset.named_split_fpath_dict = named_split_fpath_dict
        dataset.data_fpath_dict = named_split_fpath_dict['data']
        dataset.label_fpath_dict = named_split_fpath_dict['labels']
        if 'metadata' in named_split_fpath_dict:
            dataset.metadata_fpath_dict = named_split_fpath_dict['labels']
        else:
            dataset.metadata_fpath_dict = None

    def asdict(dataset):
        # save all args passed into constructor as a dict
        key_list = ut.get_func_argspec(dataset.__init__).args[1:]
        data_dict = ut.dict_subset(dataset.__dict__, key_list)
        return data_dict

    @classmethod
    def from_alias_key(cls, alias_key):
        if NOCACHE_ALIAS:
            raise Exception('Aliasing Disabled')
        # shortcut to the cached information so we dont need to
        # compute hotspotter matching hashes. There is a change data
        # can get out of date while this is enabled.
        alias_fpath = get_alias_dict_fpath()
        alias_dict = ut.text_dict_read(alias_fpath)
        if alias_key in alias_dict:
            data_dict = alias_dict[alias_key]
            ut.assert_exists(data_dict['training_dpath'])
            ut.assert_exists(data_dict['data_fpath'])
            ut.assert_exists(data_dict['labels_fpath'])
            dataset = cls(**data_dict)
            print('[dataset] Returning aliased data alias_key=%r' % (alias_key,))
            return dataset
        raise Exception('Alias cache miss:\n    alias_key=%r' % (alias_key,))

    @classmethod
    def new_training_set(cls, **kwargs):
        dataset = cls(**kwargs)
        # creates a symlink in the junction dir
        register_training_dpath(dataset.training_dpath, dataset.alias_key)
        dataset.save_alias(dataset.alias_key)
        return dataset

    def save_alias(dataset, alias_key):
        # shortcut to the cached information so we dont need to
        # compute hotspotter matching hashes. There is a change data
        # can get out of date while this is enabled.
        alias_fpath = get_alias_dict_fpath()
        alias_dict = ut.text_dict_read(alias_fpath)
        data_dict = dataset.asdict()
        alias_dict[alias_key] = data_dict
        ut.text_dict_write(alias_fpath, alias_dict)

    def load_subset_data(dataset, key='all'):
        data_fpath = dataset.named_split_fpath_dict['data'][key]
        data = ut.load_data(data_fpath, verbose=True)
        if len(data.shape) == 3:
            # add channel dimension for implicit grayscale
            data.shape = data.shape + (1,)
        return data

    def load_subset_labels(dataset, key='all'):
        labels_fpath = dataset.named_split_fpath_dict['labels'][key]
        labels = (None if labels_fpath is None
                  else ut.load_data(labels_fpath, verbose=True))
        return labels

    def load_subset_metadata(dataset, key='all'):
        if 'metadata' in dataset.named_split_fpath_dict:
            metadata_fpath = dataset.named_split_fpath_dict['metadata'][key]
            flat_metadata = ut.load_data(metadata_fpath, verbose=True)
        else:
            flat_metadata = None
        return flat_metadata

    def load_subset(dataset, key):
        """ loads a test/train/valid/all data subset """
        data = dataset.load_subset_data(key)
        labels = dataset.load_subset_labels(key)
        utils.print_data_label_info(data, labels, key)
        dataset.print_dataset_info(data, labels, key)
        return data, labels

    @staticmethod
    def print_dataset_info(data, labels, key):
        # print('[load] adding channels...')
        # data = utils.add_channels(data)
        #data   = dataset.load_subset_data(key)
        #labels = dataset.load_subset_labels(key)
        print('[dataset] %s_memory(data) = %r' % (key, ut.get_object_size_str(data),))
        print('[dataset] %s_data.shape   = %r' % (key, data.shape,))
        print('[dataset] %s_data.dtype   = %r' % (key, data.dtype,))
        print('[dataset] %s_labels.shape = %r' % (key, labels.shape,))
        print('[dataset] %s_labels.dtype = %r' % (key, labels.dtype,))
        labelhist = {key: len(val) for key, val in six.iteritems(ut.group_items(labels, labels))}
        print('[dataset] %s_label histogram = \n%s' % (key, ut.dict_str(labelhist)))
        print('[dataset] %s_label total = %d' % (key, sum(labelhist.values())))

    @property
    def labels(dataset):
        return dataset.load_subset_labels()

    @property
    def data(dataset):
        return dataset.load_subset_data()

    @property
    def metadata(dataset):
        return dataset.load_subset_metadata()

    def interact(dataset, **kwargs):
        """
        python -m ibeis_cnn --tf netrun --db mnist --ensuredata --show --datatype=category
        """
        from ibeis_cnn import draw_results
        #interact_func = draw_results.interact_siamsese_data_patches
        interact_func = draw_results.interact_dataset
        # Automatically infer which lazy properties are needed for the
        # interaction.
        kwargs_list = ut.recursive_parse_kwargs(interact_func)
        interact_kw = {key: dataset.getprop(key)
                       for key in kwargs_list if dataset.hasprop(key)}
        interact_kw.update(**kwargs)
        # TODO : generalize
        return interact_func(
            dataset.labels, dataset.data, dataset.metadata, dataset.data_per_label,
            **interact_kw)


def get_alias_dict_fpath():
    alias_fpath = join(get_juction_dpath(), 'alias_dict_v2.txt')
    return alias_fpath


def get_juction_dpath():
    r"""
    Returns:
        str: junction_dpath

    CommandLine:
        python -m ibeis_cnn --tf get_juction_dpath --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_cnn.dataset import *  # NOQA
        >>> junction_dpath = get_juction_dpath()
        >>> result = ('junction_dpath = %s' % (str(junction_dpath),))
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> ut.vd(junction_dpath)
    """
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


def ondisk_data_split(data_fpath, labels_fpath, metadata_fpath, data_per_label,
                      split_names=['train', 'valid', 'test'],
                      fraction_list=[.2, .1], nocache=None):
    """
    splits into train / validation datasets on disk

    # TODO: metadata fpath

    split_names=['train', 'valid', 'test'], fraction_list=[.2, .1]

    TODO: maybe use folds instead of fractions

    CommandLine:
        python -m ibeis_cnn.dataset --test-ondisk_data_split

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.dataset import *  # NOQA
        >>> from ibeis_cnn import ingest_data
        >>> dataset = ingest_data.testdata_dataset()
        >>> data_fpath = dataset.data_fpath
        >>> labels_fpath = dataset.labels_fpath
        >>> metadata_fpath = dataset.metadata_fpath
        >>> data_per_label = dataset.data_per_label
        >>> split_names = ['train', 'valid', 'test']
        >>> fraction_list = [0.2, 0.1]
        >>> nocache = True
        >>> named_split_fpath_dict = ondisk_data_split(data_fpath, labels_fpath,
        >>>                                            metadata_fpath,
        >>>                                            data_per_label,
        >>>                                            split_names,
        >>>                                            fraction_list, nocache)
        >>> from os.path import basename
        >>> data_bytes = ut.map_dict_vals(ut.get_file_nBytes_str, named_split_fpath_dict['data'])
        >>> label_bytes = ut.map_dict_vals(ut.get_file_nBytes_str, named_split_fpath_dict['labels'])
        >>> data_fpath_dict = ut.map_dict_vals(basename, named_split_fpath_dict['data'])
        >>> label_fpath_dict = ut.map_dict_vals(basename, named_split_fpath_dict['labels'])
        >>> print('(data_bytes, label_bytes) = %s' % (ut.list_str((data_bytes, label_bytes), nl=True),))
        >>> result = ('(data_fpath_dict, label_fpath_dict) = %s' % (ut.list_str((data_fpath_dict, label_fpath_dict), nl=True),))
        >>> print(result)
    """
    assert len(split_names) == len(fraction_list) + 1, (
        'must have one less fraction then split names')

    _fpath_dict = {
        'data': data_fpath,
        'labels': labels_fpath,
        'metadata': metadata_fpath,
    }
    #_rows_per_label_dict = {
    #    'data': data_per_label,
    #    'metadata': 1,
    #    'labels': 1,
    #}

    # Remove non-existing fpaths
    fpath_dict = {
        key: val
        for key, val in _fpath_dict.items()
        if val is not None
    }
    #rows_per_label_dict = {_rows_per_label_dict[key] for key in fpath_dict.keys()}

    assert 'data' in fpath_dict
    assert 'labels' in fpath_dict

    if 'metadata' not in fpath_dict:
        print('Warning no metadata')

    training_dir = dirname(fpath_dict['data'])
    splitdir = join(training_dir, 'data_splits')

    USE_FILE_UUIDS = False
    if USE_FILE_UUIDS:
        # Get uuid based on the data, so different data makes different validation paths
        fpath_uuids = [ut.get_file_uuid(fpath)
                       for fpath in fpath_dict.values()]
        split_uuid = ut.augment_uuid(*fpath_uuids)
        hashstr_ = ut.hashstr(str(split_uuid), alphabet=ut.ALPHABET_16)
    else:
        # Faster to base on the data fpath if that already has a uuid in it
        hashstr_ = ut.hashstr(basename(fpath_dict['data']), alphabet=ut.ALPHABET_16)

    # Get the total fraction of data for each subset
    totalfrac_list = [1.0]
    for fraction in fraction_list:
        total = totalfrac_list[-1]
        right = total * fraction
        left = total * (1 - fraction)
        totalfrac_list[-1] = left
        totalfrac_list.append(right)

    def make_split_fpaths(type_, fpath, splitdir):
        ext = splitext(fpath)[1]
        return [
            join(splitdir, '%s_%s_%.3f_%s%s' % (
                name, type_, frac, hashstr_, ext))
            for name, frac in zip(split_names, totalfrac_list)
        ]

    split_fpaths_dict = {
        type_: make_split_fpaths(type_, fpath, splitdir)
        for type_, fpath in fpath_dict.items()
    }

    is_cache_hit = all([
        all(map(exists, fpaths))
        for fpaths in split_fpaths_dict.values()
    ])

    if nocache is None:
        nocache = NOCACHE_DATA_SPLIT

    ut.ensuredir(splitdir)

    if not is_cache_hit or nocache:
        print('Writing data splits')

        def take_items(items, idx_list):
            if isinstance(items, dict):
                return {key: val.take(idx_list, axis=0)
                        for key, val in items.items()}
            else:
                return items.take(idx_list, axis=0)

        loaded_dict = {
            key: ut.load_data(fpath, verbose=True)
            for key, fpath in fpath_dict.items()
        }
        avail_dict = loaded_dict.copy()
        fraction_list_ = fraction_list + [None]
        for index in range(len(fraction_list_)):
            print('--------index = %r' % (index,))
            part_dict = {}
            remain_dict = {}
            fraction = fraction_list_[index]

            part_fpath_dict = {
                key: val[index]
                for key, val in split_fpaths_dict.items()
            }

            if fraction is None:
                label_indices1 = np.arange(len(avail_dict['labels']))
                label_indices2 = None
            else:
                label_indices1, label_indices2 = partition_label_indices(
                    avail_dict['labels'], fraction=fraction, shuffle=True)

            data_indicies1 = expand_data_indicies(label_indices1, data_per_label)

            part_dict['labels'] = take_items(avail_dict['labels'], label_indices1)
            part_dict['data']   = take_items(avail_dict['data'], data_indicies1)
            if 'metadata' in avail_dict:
                part_dict['metadata'] = take_items(avail_dict['metadata'], label_indices1)

            if label_indices2 is not None:
                data_indicies2 = expand_data_indicies(label_indices2, data_per_label)
                remain_dict['labels'] = take_items(avail_dict['labels'], label_indices2)
                remain_dict['data'] = take_items(avail_dict['data'], data_indicies2)
                if 'metadata' in avail_dict:
                    remain_dict['metadata'] = take_items(avail_dict['metadata'], label_indices2)

            for key in part_dict.keys():
                ut.save_data(part_fpath_dict[key], part_dict[key], verbose=2)

            avail_dict = remain_dict.copy()

    named_split_fpath_dict = {
        type_: dict(zip(split_names, split_fpaths))
        for type_, split_fpaths in split_fpaths_dict.items()
    }

    for type_, split_fpath_dict in named_split_fpath_dict.items():
        split_fpath_dict['all'] = fpath_dict[type_]

    return named_split_fpath_dict


#def
#split_labels_fpaths = split_fpaths_dict['labels']

#    if not is_cache_hit or nocache:
#        print('Writing data splits')
#        X_left, y_left = utils.load(data_fpath, labels_fpath)
#        _iter = zip(fraction_list, split_data_fpaths, split_labels_fpaths)
#        for fraction, x_fpath, y_fpath in _iter:
#            _tup = utils.train_test_split(X_left, y_left, eval_size=fraction,
#                                          data_per_label=data_per_label,
#                                          shuffle=True)
#            X_left, y_left, X_right, y_right = _tup
#            #print('-----------')
#            #print(x_fpath)
#            #print(y_fpath)
#            #print(X_right.shape[0] / X_left.shape[0])
#            #print(y_right.shape[0] / y_left.shape[0])
#            #print('-----------')
#            utils.write_data_and_labels(X_left, y_left, x_fpath, y_fpath)
#            X_left = X_right
#            y_left = y_right
#        x_fpath  = split_data_fpaths[-1]
#        y_fpath = split_labels_fpaths[-1]
#        utils.write_data_and_labels(X_left, y_left, x_fpath, y_fpath)

#    data_fpath_dict = dict(zip(split_names, split_data_fpaths))
#    label_fpath_dict = dict(zip(split_names, split_labels_fpaths))

#    data_fpath_dict['all']  = data_fpath
#    label_fpath_dict['all'] = labels_fpath
#    return data_fpath_dict, label_fpath_dict


def load(data_fpath, labels_fpath=None):
    # Load X matrix (data)
    data = ut.load_data(data_fpath)
    labels = ut.load_data(labels_fpath) if labels_fpath is not None else None
    ## TODO: This should be part of data preprocessing
    ## Ensure that data is 4-dimensional
    if len(data.shape) == 3:
        # add channel dimension for implicit grayscale
        data.shape = data.shape + (1,)
    # Return data
    return data, labels


def partition_label_indices(labels, fraction,  shuffle=True):
    r"""
    used to split datasets into two parts.
    Preserves class distributions using Stratified K-Fold sampling

    Args:
        labels (ndarray):

    Returns:
        tuple: (X_train, y_train, X_valid, y_valid)

    CommandLine:
        python -m ibeis_cnn.utils --test-train_test_split

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.dataset import *  # NOQA
        >>> # build test data
        >>> labels = [0, 0, 0, 0, 1, 1, 1, 1]
        >>> fraction = .5
        >>> indices1, indices2 = partition_label_indices(labels, fraction)
        >>> result = str((indices1, indices2))
        >>> print(result)
        (array([1, 2, 5, 7]), array([0, 3, 4, 6]))
    """
    # take the data and label arrays, split them preserving the class distributions
    import sklearn.cross_validation
    nfolds = int(round(1. / fraction))
    rng = np.random.RandomState(0)
    skf = sklearn.cross_validation.StratifiedKFold(labels, nfolds,
                                                   shuffle=shuffle,
                                                   random_state=rng)
    indices1, indices2 = six.next(iter(skf))
    return indices1, indices2


def expand_data_indicies(label_indices, data_per_label=1):
    """
    when data_per_label > 1, gives the corresponding data indicies for the data
    indicies
    """
    import numpy as np
    expanded_indicies = [label_indices * data_per_label + count
                         for count in range(data_per_label)]
    data_indices = np.vstack(expanded_indicies).T.flatten()
    return data_indices


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis_cnn.dataset
        python -m ibeis_cnn.dataset --allexamples
        python -m ibeis_cnn.dataset --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
