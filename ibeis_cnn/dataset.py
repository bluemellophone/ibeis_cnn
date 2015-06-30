# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from ibeis_cnn import utils
from os.path import join, basename, exists, dirname
import six
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

    TODO: metadata
    """
    def __init__(dataset, alias_key, training_dpath, data_fpath, labels_fpath,
                 data_per_label, data_shape, output_dims, num_labels):
        # Constructor args is primary data
        key_list = ut.get_func_argspec(dataset.__init__).args[1:]
        locals_ = locals()
        for key in key_list:
            setattr(dataset, key, locals_[key])
        # Define auxillary data
        dataset.build_auxillary_data()

    def build_auxillary_data(dataset):
        # Make test train validatation sets
        data_fpath_dict, label_fpath_dict = ondisk_data_split(
            dataset.data_fpath, dataset.labels_fpath, dataset.data_per_label,
            split_names=['train', 'valid', 'test'],
            fraction_list=[.2, .1])
        dataset.data_fpath_dict = data_fpath_dict
        dataset.label_fpath_dict = label_fpath_dict

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
            print('[get_ibeis_siam_dataset] Returning aliased data alias_key=%r' % (alias_key,))
            return dataset
        raise Exception('Alias cache miss: alias_key=%r' % (alias_key,))

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

    def load_subset(dataset, key):
        """ loads a test/train/valid/all data subset """
        data, labels = utils.load(dataset.data_fpath_dict[key], dataset.label_fpath_dict[key])
        utils.print_data_label_info(data, labels, key)
        #X, y = utils.load_from_fpath_dicts(dataset.data_fpath_dict, dataset.label_fpath_dict, key)
        return data, labels


def get_alias_dict_fpath():
    alias_fpath = join(get_juction_dpath(), 'alias_dict_v2.txt')
    return alias_fpath


def get_juction_dpath():
    r"""
    Returns:
        ?: junction_dpath

    CommandLine:
        python -m ibeis_cnn.dataset --test-get_juction_dpath

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_cnn.dataset import *  # NOQA
        >>> junction_dpath = get_juction_dpath()
        >>> ut.vd(junction_dpath)
        >>> result = ('junction_dpath = %s' % (str(junction_dpath),))
        >>> print(result)
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


def ondisk_data_split(data_fpath, labels_fpath, data_per_label, split_names=['train', 'valid', 'test'], fraction_list=[.2, .1], nocache=None):
    """
    splits into train / validation datasets on disk

    # TODO: metadata fpath

    split_names=['train', 'valid', 'test'], fraction_list=[.2, .1]

    CommandLine:
        python -m ibeis_cnn.dataset --test-ondisk_data_split

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.dataset import *  # NOQA
        >>> from ibeis_cnn import ingest_data
        >>> dataset = ingest_data.testdata_dataset()
        >>> data_fpath = dataset.data_fpath
        >>> labels_fpath = dataset.labels_fpath
        >>> data_per_label = dataset.data_per_label
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

    from os.path import splitext

    data_ext = splitext(data_fpath)[1]
    labels_ext = splitext(data_fpath)[1]

    split_data_fpaths = [join(splitdir, name + '_data_%.3f_' % (frac,) + hashstr_ + data_ext)
                         for name, frac in zip(split_names, totalfrac_list)]
    split_labels_fpaths = [join(splitdir, name + '_labels_%.3f_' % (frac,) + hashstr_ + labels_ext)
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
