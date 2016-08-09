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
                 metadata_fpath, data_per_label, data_shape, output_dims,
                 num_labels, dataset_dpath=None):
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
            python -Bc "import utool, ibeis_cnn.dataset; utool.inject_python_code2(utool.get_modpath(ibeis_cnn.dataset.__name__), utool.indent('\n'.join(['dataset.{key} = {key}'.format(key=key) for key in utool.get_func_argspec(ibeis_cnn.dataset.DataSet.__init__).args[1:]]), ' ' * 12), 'AUTOGEN_INIT')"
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
            dataset.dataset_dpath = dataset_dpath
            # </AUTOGEN_INIT>
        else:
            locals_ = locals()
            for key in key_list:
                setattr(dataset, key, locals_[key])
        # Dictionary for storing different data subsets
        dataset.fpath_dict = {
            'all' : {
                'data': data_fpath,
                'labels': labels_fpath,
                'metadata': metadata_fpath,
            }
        }
        # Hacky dictionary for custom things
        # Probably should be refactored
        dataset._lazy_cache = ut.LazyDict()

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
            ut.assert_exists(data_dict['metadata_fpath'])
            dataset = cls(**data_dict)
            # dataset.build_auxillary_data()
            print('[dataset] Returning aliased data alias_key=%r' % (alias_key,))
            return dataset
        raise Exception('Alias cache miss:\n    alias_key=%r' % (alias_key,))

    @classmethod
    def new_training_set(cls, **kwargs):
        dataset = cls(**kwargs)
        # Define auxillary data
        try:
            # dataset.build_auxillary_data()
            dataset.register_self()
            dataset.save_alias(dataset.alias_key)
        except Exception as ex:
            ut.printex(ex, 'WARNING was not able to generate splis or save alias')
        return dataset

    def hasprop(dataset, key):
        return key in dataset._lazy_cache.keys()

    def getprop(dataset, key, *d):
        if len(d) == 0:
            return dataset._lazy_cache[key]
        else:
            assert len(d) == 1
            if key in dataset._lazy_cache:
                return dataset._lazy_cache[key]
            else:
                return d[0]

    def setprop(dataset, key, val):
        dataset._lazy_cache[key] = val

    def load_subset(dataset, key):
        """ loads a test/train/valid/all data subset """
        data = dataset.load_subset_data(key)
        labels = dataset.load_subset_labels(key)
        return data, labels

    def print_subset_info(dataset, key='all'):
        data, labels = dataset.load_subset(key)
        dataset.print_dataset_info(data, labels, key)

    @property
    def labels(dataset):
        return dataset.load_subset_labels()

    @property
    def data(dataset):
        return dataset.load_subset_data()

    @property
    def metadata(dataset):
        return dataset.load_subset_metadata()

    def asdict(dataset):
        # save all args passed into constructor as a dict
        key_list = ut.get_func_argspec(dataset.__init__).args[1:]
        data_dict = ut.dict_subset(dataset.__dict__, key_list)
        return data_dict

    def register_self(dataset):
        # creates a symlink in the junction dir
        register_training_dpath(dataset.training_dpath, dataset.alias_key)

    def save_alias(dataset, alias_key):
        # shortcut to the cached information so we dont need to
        # compute hotspotter matching hashes. There is a change data
        # can get out of date while this is enabled.
        alias_fpath = get_alias_dict_fpath()
        alias_dict = ut.text_dict_read(alias_fpath)
        data_dict = dataset.asdict()
        alias_dict[alias_key] = data_dict
        ut.text_dict_write(alias_fpath, alias_dict)

    @ut.memoize
    def load_subset_data(dataset, key='all'):
        data_fpath = dataset.fpath_dict[key]['data']
        data = ut.load_data(data_fpath, verbose=True)
        if len(data.shape) == 3:
            # add channel dimension for implicit grayscale
            data.shape = data.shape + (1,)
        return data

    @ut.memoize
    def load_subset_labels(dataset, key='all'):
        labels_fpath = dataset.fpath_dict[key]['labels']
        labels = (None if labels_fpath is None
                  else ut.load_data(labels_fpath, verbose=True))
        return labels

    @ut.memoize
    def load_subset_metadata(dataset, key='all'):
        metadata_fpath = dataset.fpath_dict[key].get('metadata', None)
        if metadata_fpath is not None:
            flat_metadata = ut.load_data(metadata_fpath, verbose=True)
        else:
            flat_metadata = None
        return flat_metadata

    @staticmethod
    def print_dataset_info(data, labels, key):
        labelhist = {key: len(val) for key, val in ut.group_items(labels, labels).items()}
        stats_dict = ut.get_stats(data.ravel())
        ut.delete_keys(stats_dict, ['shape', 'nMax', 'nMin'])
        print('[dataset] Dataset Info: ')
        print('[dataset] * Data:')
        print('[dataset]     %s_data(shape=%r, dtype=%r)' % (key, data.shape, data.dtype))
        print('[dataset]     %s_memory(data) = %r' % (key, ut.get_object_size_str(data),))
        print('[dataset]     %s_stats(data) = %s' % (key, ut.repr2(stats_dict, precision=2),))
        print('[dataset] * Labels:')
        print('[dataset]     %s_labels(shape=%r, dtype=%r)' % (key, labels.shape, labels.dtype))
        print('[dataset]     %s_label histogram = %s' % (key, ut.repr2(labelhist)))

    def interact(dataset, key='all', **kwargs):
        """
        python -m ibeis_cnn --tf netrun --db mnist --ensuredata --show --datatype=category
        python -m ibeis_cnn --tf netrun --db PZ_MTEST --acfg ctrl --ensuredata --show
        """
        from ibeis_cnn import draw_results
        #interact_func = draw_results.interact_siamsese_data_patches
        interact_func = draw_results.interact_dataset
        # Automatically infer which lazy properties are needed for the
        # interaction.
        kwarg_items = ut.recursive_parse_kwargs(interact_func)
        kwarg_keys = ut.get_list_column(kwarg_items, 0)
        interact_kw = {key_: dataset.getprop(key_)
                       for key_ in kwarg_keys if dataset.hasprop(key_)}
        interact_kw.update(**kwargs)
        # TODO : generalize
        data     = dataset.load_subset_data(key)
        labels   = dataset.load_subset_labels(key)
        metadata = dataset.load_subset_metadata(key)
        return interact_func(
            labels, data, metadata, dataset.data_per_label,
            **interact_kw)

    def view_directory(dataset):
        ut.view_directory(dirname(dataset.data_fpath))

    def has_splitset(dataset, key):
        return key in dataset.fpath_dict

    @property
    def split_dpath(dataset):
        split_dpath = join(dataset.dataset_dpath, 'data_splits')
        return split_dpath

    def load_splitsets(dataset):
        import parse
        fmtstr_ = '{key}_{type_}_{size:d}{ext}'
        fpath_dict = {}
        for fpath in ut.ls(dataset.split_dpath):
            parsed = parse.parse(fmtstr_, basename(fpath))
            key = parsed['key']
            type_ = parsed['type_']
            splitset = fpath_dict.get(key, {})
            splitset[type_] = fpath
            fpath_dict[key] = splitset
        # check validity of loaded data
        for key, val in fpath_dict.items():
            assert 'data' in val, 'subset missing data'
        dataset.fpath_dict.update(**fpath_dict)

    def add_splitset(dataset, key, idxs):
        print('[dataset] adding splitset %r' % (key,))
        # Partition data into the subset
        part_dict = {
            'data': dataset.data.take(idxs, axis=0),
            'labels': dataset.labels.take(idxs, axis=0),
        }
        if dataset.metadata is not None:
            taker = ut.partial(ut.take, index_list=idxs)
            part_dict['metadata'] = ut.map_dict_vals(taker, dataset.metadata)
        # Build subset filenames
        ut.ensuredir(dataset.split_dpath)
        ext = '.pkl'
        fmtstr_ = '{key}_{type_}_{size}{ext}'
        fmtdict = dict(key=key, ext=ext, size=len(idxs))
        splitset = {
            type_: join(dataset.split_dpath, fmtstr_.format(type_=type_, **fmtdict))
            for type_ in part_dict.keys()
        }
        # Write splitset data to files
        for type_ in part_dict.keys():
            ut.save_data(splitset[type_], part_dict[type_])
        # Register filenames with dataset
        dataset.fpath_dict[key] = splitset

    def build_auxillary_data(dataset):
        # Make test train validatation sets
        data_fpath = dataset.data_fpath
        labels_fpath = dataset.labels_fpath
        metadata_fpath = dataset.metadata_fpath
        data_per_label = dataset.data_per_label
        split_names = ['train', 'test', 'valid']
        fractions = [.7, .2, .1]
        named_split_fpath_dict = ondisk_data_split(
            data_fpath, labels_fpath, metadata_fpath,
            data_per_label, split_names, fractions,
        )
        for key, val in named_split_fpath_dict.items():
            splitset = dataset.fpath_dict.get(key, {})
            splitset.update(**val)
            dataset.fpath_dict[key] = splitset


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
    # Hacks to keep junction clean
    home_dlink = ut.truepath('~/training_junction')
    if not exists(home_dlink):
        ut.symlink(junction_dpath, home_dlink)
    ut.remove_broken_links(junction_dpath)
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


def ondisk_data_split(data_fpath, labels_fpath, metadata_fpath,
                      data_per_label,
                      split_names=['train', 'test', 'valid'],
                      fractions=[.7, .2, .1], use_cache=None):
    """
    splits into train / test / validation datasets on disk

    # TODO: ENSURE THAT VALIDATION / TEST SET CONTAINS DISJOINT IMAGES FROM
    # TRAINING

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
        >>> fractions = [.7, 0.1, 0.2]
        >>> use_cache = False
        >>> named_split_fpath_dict = ondisk_data_split(
        >>>     data_fpath, labels_fpath, metadata_fpath, data_per_label,
        >>>     split_names, fractions, use_cache)
        >>> from os.path import basename
        >>> data_bytes = ut.map_dict_vals(ut.get_file_nBytes_str, named_split_fpath_dict['data'])
        >>> label_bytes = ut.map_dict_vals(ut.get_file_nBytes_str, named_split_fpath_dict['labels'])
        >>> data_fpath_dict = ut.map_dict_vals(basename, named_split_fpath_dict['data'])
        >>> label_fpath_dict = ut.map_dict_vals(basename, named_split_fpath_dict['labels'])
        >>> print('(data_bytes, label_bytes) = %s' % (ut.list_str((data_bytes, label_bytes), nl=True),))
        >>> result = ('(data_fpath_dict, label_fpath_dict) = %s' % (ut.list_str((data_fpath_dict, label_fpath_dict), nl=True),))
        >>> print(result)
    """
    print('ondisk_data_split')
    assert len(split_names) == len(fractions), ('names and fractions not aligned')
    assert np.isclose(sum(fractions), 1), 'fractions must sum to 1'
    _fpath_dict = {'data': data_fpath, 'labels': labels_fpath,
                   'metadata': metadata_fpath}
    # Remove non-existing fpaths
    fpath_dict = {key: val for key, val in _fpath_dict.items() if val is not None}
    assert 'data' in fpath_dict
    assert 'labels' in fpath_dict
    if 'metadata' not in fpath_dict:
        print('Warning no metadata')

    training_dir = dirname(fpath_dict['data'])
    splitdir = join(training_dir, 'data_splits')
    # base on the data fpath if that already has a uuid in it
    hashstr_ = ut.hashstr(basename(fpath_dict['data']), alphabet=ut.ALPHABET_16)

    def make_split_fpaths(type_, fpath, splitdir):
        ext = splitext(fpath)[1]
        return [
            join(splitdir, '%s_%s_%.3f_%s%s' % (
                name, type_, frac, hashstr_, ext))
            for name, frac in zip(split_names, fractions)
        ]

    split_fpaths_dict = {type_: make_split_fpaths(type_, fpath, splitdir)
                         for type_, fpath in fpath_dict.items()}
    is_cache_hit = all([all(map(exists, fpaths))
                        for fpaths in split_fpaths_dict.values()])

    if use_cache is None:
        use_cache = not NOCACHE_DATA_SPLIT

    ut.ensuredir(splitdir)
    if not is_cache_hit or not use_cache:
        print('Writing data splits')

        def take_items(items, idx_list):
            if isinstance(items, dict):
                return {key: val.take(idx_list, axis=0)
                        for key, val in items.items()}
            else:
                return items.take(idx_list, axis=0)

        labels = ut.load_data(fpath_dict['labels'], verbose=True)
        # Generate the split indicies
        rng = np.random.RandomState(0)
        sample_idxs = stratified_shuffle_split(labels, fractions, rng=rng)

        # Load and break the data
        loaded_dict = {
            key: ut.load_data(fpath, verbose=True)
            for key, fpath in fpath_dict.items() if key != 'labels'
        }
        loaded_dict['labels'] = labels

        for index in range(len(fractions)):
            print('--------index = %r' % (index,))
            part_dict = {}
            part_fpath_dict = {key: val[index] for key, val in
                               split_fpaths_dict.items()}
            label_indices1 = sample_idxs[index]
            data_indicies1 = expand_data_indicies(label_indices1, data_per_label)

            part_dict['labels'] = take_items(loaded_dict['labels'], label_indices1)
            part_dict['data']   = take_items(loaded_dict['data'], data_indicies1)
            if 'metadata' in loaded_dict:
                part_dict['metadata'] = take_items(loaded_dict['metadata'], label_indices1)
            for key in part_dict.keys():
                ut.save_data(part_fpath_dict[key], part_dict[key], verbose=2)

    named_split_fpath_dict = {
        type_: dict(zip(split_names, split_fpaths))
        for type_, split_fpaths in split_fpaths_dict.items()
    }

    for type_, split_fpath_dict in named_split_fpath_dict.items():
        split_fpath_dict['all'] = fpath_dict[type_]

    return named_split_fpath_dict


#def load(data_fpath, labels_fpath=None):
#    # Load X matrix (data)
#    data = ut.load_data(data_fpath)
#    labels = ut.load_data(labels_fpath) if labels_fpath is not None else None
#    ## TODO: This should be part of data preprocessing
#    ## Ensure that data is 4-dimensional
#    if len(data.shape) == 3:
#        # add channel dimension for implicit grayscale
#        data.shape = data.shape + (1,)
#    # Return data
#    return data, labels


def stratified_shuffle_split(y, fractions, rng=None, class_weights=None):
    """
    modified from sklearn to make n splits instaed of 2
    """

    n_samples = len(y)
    classes, y_indices = np.unique(y, return_inverse=True)
    n_classes = classes.shape[0]
    class_counts = np.bincount(y_indices)

    # TODO: weighted version
    #class_counts_ = np.array([sum([w.get(cx, 0) for w in class_weights]) for cx in classes])

    # Number of sets to split into
    num_sets = len(fractions)
    fractions = np.asarray(fractions)
    set_sizes = n_samples * fractions

    if np.min(class_counts) < 2:
        raise ValueError("The least populated class in y has only 1"
                         " member, which is too few. The minimum"
                         " number of labels for any class cannot"
                         " be less than 2.")
    for size in set_sizes:
        if size < n_classes:
            raise ValueError('The size = %d of all splits should be greater or '
                             'equal to the number of classes = %d' %
                             (size, n_classes))
    if rng is None:
        rng = np.random
    elif isinstance(rng, int):
        rng = np.random.RandomState(rng)

    # Probability sampling of class[i]
    p_i = class_counts / float(n_samples)

    # Get split points within each class sample
    set_perclass_nums = []
    count_remain = class_counts.copy()
    for x in range(num_sets):
        size = set_sizes[x]
        set_i = np.round(size * p_i).astype(int)
        set_i = np.minimum(count_remain, set_i)
        count_remain -= set_i
        set_perclass_nums.append(set_i)
    set_perclass_nums = np.array(set_perclass_nums)

    index_sets = [[] for _ in range(num_sets)]

    for i, class_i in enumerate(classes):
        # Randomly shuffle all members of class i
        permutation = rng.permutation(class_counts[i])
        perm_indices_class_i = np.where((y == class_i))[0][permutation]
        # Pick out members according to split points

        split_sample = np.split(perm_indices_class_i, set_perclass_nums.T[i].cumsum())
        assert len(split_sample) == num_sets + 1
        for x in range(num_sets):
            index_sets[x].extend(split_sample[x])
        missing_indicies = split_sample[-1]

        # Randomly missing assign indicies to a set
        set_idxs = rng.randint(0, num_sets, len(missing_indicies))
        for x in range(num_sets):
            idxs = np.where(set_idxs == x)[0]
            index_sets[x].extend(missing_indicies[idxs])

    for set_idx in range(num_sets):
        # shuffle the indicies again
        index_sets[set_idx] = rng.permutation(index_sets[set_idx])
    return index_sets


def stratified_label_shuffle_split(y, labels, fractions, rng=None):
    """
    modified from sklearn to make n splits instaed of 2.
    Also enforces that labels are not broken into separate groups.

    Args:
        y (ndarray):  labels
        labels (?):
        fractions (?):
        rng (RandomState):  random number generator(default = None)

    Returns:
        ?: index_sets

    CommandLine:
        python -m ibeis_cnn.dataset stratified_label_shuffle_split --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.dataset import *  # NOQA
        >>> y      = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        >>> labels = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 6, 0, 7, 7, 7, 7]
        >>> fractions = [.7, .3]
        >>> rng = np.random.RandomState(0)
        >>> index_sets = stratified_label_shuffle_split(y, labels, fractions, rng)
    """
    if rng is None:
        rng = np.random
    #orig_y = y
    unique_labels, groupxs = ut.group_indices(labels)
    grouped_ys = ut.apply_grouping(y, groupxs)
    # Assign each group a probabilistic class
    unique_ys = [ys[rng.randint(0, len(ys))] for ys in grouped_ys]
    # TODO: should weight the following selection based on size of group
    #class_weights = [ut.dict_hist(ys) for ys in grouped_ys]

    unique_idxs = stratified_shuffle_split(unique_ys, fractions, rng)
    index_sets = [ut.flatten(ut.take(groupxs, idxs)) for idxs in unique_idxs]
    return index_sets


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
