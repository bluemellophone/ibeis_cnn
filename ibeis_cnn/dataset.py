# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from os.path import join, basename, exists
import six
import numpy as np
import utool as ut
print, rrr, profile = ut.inject2(__name__, '[ibeis_cnn.datset]')


#NOCACHE_ALIAS = True
# NOCACHE_ALIAS = ut.get_argflag('--nocache-alias')
# NOCACHE_DATA_SPLIT = ut.get_argflag('--nocache-datasplit')


class DummyDataSet(object):
    def __init__(dataset):
        dataset.data_fpath_dict = None
        dataset.label_fpath_dict = None


@six.add_metaclass(ut.ReloadingMetaclass)
class DataSet(ut.NiceRepr):
    """
    helper class for managing dataset paths and general metadata

    SeeAlso:
        python -m ibeis_cnn.ingest_data --test-get_ibeis_part_siam_dataset --show

    """
    #def __init__(dataset, alias_key, training_dpath, data_fpath, labels_fpath,
    #             metadata_fpath, data_per_label, data_shape, output_dims,
    #             num_labels, xdata_dpath=None):
    #    # Constructor args is primary data
    #    key_list = ut.get_func_argspec(dataset.__init__).args[1:]
    #    r"""
    #    Gen Explicit:
    #        import utool, ibeis_cnn.dataset
    #        key_list = ut.get_func_argspec(ibeis_cnn.dataset.DataSet.__init__).args[1:]
    #        autogen_block = ut.indent('\n'.join(['dataset.{key} = {key}'.format(key=key) for key in key_list]), ' ' * 12)
    #        ut.inject_python_code2(ibeis_cnn.dataset.__file__, autogen_block, 'AUTOGEN_INIT')
    #        #ut.copy_text_to_clipboard('\n' + autogen_block)

    #    Gen Explicit Commandline:
    #        python -Bc "import utool, ibeis_cnn.dataset; utool.inject_python_code2(utool.get_modpath(ibeis_cnn.dataset.__name__), utool.indent('\n'.join(['dataset.{key} = {key}'.format(key=key) for key in utool.get_func_argspec(ibeis_cnn.dataset.DataSet.__init__).args[1:]]), ' ' * 12), 'AUTOGEN_INIT')"
    #    """
    #    EXPLICIT = True
    #    if EXPLICIT:
    #        pass
    #        # <AUTOGEN_INIT>
    #        dataset.alias_key = alias_key
    #        dataset.training_dpath = training_dpath
    #        dataset.data_fpath = data_fpath
    #        dataset.labels_fpath = labels_fpath
    #        dataset.metadata_fpath = metadata_fpath
    #        dataset.data_per_label = data_per_label
    #        dataset.data_shape = data_shape
    #        dataset.output_dims = output_dims
    #        dataset.num_labels = num_labels
    #        dataset.xdata_dpath = xdata_dpath
    #        # </AUTOGEN_INIT>
    #    else:
    #        locals_ = locals()
    #        for key in key_list:
    #            setattr(dataset, key, locals_[key])
    #    # Dictionary for storing different data subsets
    #    dataset.fpath_dict = {
    #        'full' : {
    #            'data': data_fpath,
    #            'labels': labels_fpath,
    #            'metadata': metadata_fpath,
    #        }
    #    }
    #    # Hacky dictionary for custom things
    #    # Probably should be refactored
    #    dataset._lazy_cache = ut.LazyDict()

    def __init__(dataset, cfgstr=None, training_dpath='.', data_shape=None,
                 num_data=None, name=None, ext='.pkl'):
        dataset.name = name
        dataset.cfgstr = cfgstr
        dataset.training_dpath = training_dpath
        assert data_shape is not None, 'must specify'
        dataset._ext = ext
        dataset.data_info = {
            'num_data': num_data,
            'data_shape': data_shape,
            'num_labels': None,
            'unique_labels': None,
            'data_per_label': None,
        }
        # Dictionary for storing different data subsets
        dataset.fpath_dict = {
            'full' : {
                'data': dataset.data_fpath,
                'labels': dataset.labels_fpath,
                'metadata': dataset.metadata_fpath,
            }
        }
        # Hacky dictionary for custom things
        # Probably should be refactored
        dataset._lazy_cache = ut.LazyDict()

    # @classmethod
    # def from_alias_key(cls, alias_key):
    #     if NOCACHE_ALIAS:
    #         raise Exception('Aliasing Disabled')
    #     # shortcut to the cached information so we dont need to
    #     # compute hotspotter matching hashes. There is a change data
    #     # can get out of date while this is enabled.
    #     alias_fpath = get_alias_dict_fpath()
    #     alias_dict = ut.text_dict_read(alias_fpath)
    #     if alias_key in alias_dict:
    #         data_dict = alias_dict[alias_key]
    #         ut.assert_exists(data_dict['training_dpath'])
    #         ut.assert_exists(data_dict['data_fpath'])
    #         ut.assert_exists(data_dict['labels_fpath'])
    #         ut.assert_exists(data_dict['metadata_fpath'])
    #         dataset = cls(**data_dict)
    #         # dataset.build_auxillary_data()
    #         print('[dataset] Returning aliased data alias_key=%r' % (alias_key,))
    #         return dataset
    #     raise Exception('Alias cache miss:\n    alias_key=%r' % (alias_key,))

    def __nice__(dataset):
        return '(' + dataset.dataset_id + ')'

    def ensure_dirs(dataset):
        ut.ensuredir(dataset.full_dpath)
        ut.ensuredir(dataset.split_dpath)

    def print_dir_structure(dataset):
        print(dataset.training_dpath)
        print(dataset.datasets_dpath)  # this is MULTIPLE dataset dir
        print(dataset.xdata_dpath)  # FIXME this is really dataset dir
        print(dataset.data_fpath)
        print(dataset.labels_fpath)
        print(dataset.metadata_fpath)
        print(dataset.info_fpath)
        print(dataset.full_dpath)
        print(dataset.split_dpath)

    def print_dir_tree(dataset):
        fpaths = ut.glob(dataset.xdata_dpath, '*', recursive=True)
        print('\n'.join(sorted(fpaths)))

    @property
    def hashid(dataset):
        if dataset.cfgstr is None:
            return ''
        else:
            return ut.hashstr27(dataset.cfgstr, hashlen=8)

    @property
    def dataset_id(dataset):
        shape_str = 'x'.join(ut.lmap(str, dataset.data_info['data_shape']))
        num_data = dataset.data_info['num_data']
        parts = []
        if dataset.name is not None:
            parts.append(dataset.name)
        if num_data is not None:
            parts.append(str(num_data))
        parts.append(shape_str)
        if dataset.hashid:
            parts.append(dataset.hashid)
        dsid = '_'.join(parts)
        return dsid

    @property
    def datasets_dpath(dataset):
        # FIME confusion between dataset and datasets
        return join(dataset.training_dpath, 'datasets')

    @property
    def xdata_dpath(dataset):
        return join(dataset.datasets_dpath, dataset.dataset_id)

    @property
    def split_dpath(dataset):
        split_dpath = join(dataset.xdata_dpath, 'splits')
        return split_dpath

    @property
    def full_dpath(dataset):
        return join(dataset.xdata_dpath, 'full')

    @property
    def info_fpath(dataset):
        return join(dataset.full_dpath, '%s_info.json' % (dataset.hashid))

    @property
    def data_fpath(dataset):
        return join(dataset.full_dpath, '%s_data%s' % (dataset.hashid, dataset._ext))

    @property
    def labels_fpath(dataset):
        return join(dataset.full_dpath, '%s_labels%s' % (dataset.hashid, dataset._ext))

    @property
    def metadata_fpath(dataset):
        return join(dataset.full_dpath, '%s_metadata%s' % (dataset.hashid, dataset._ext))

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
        """ loads a test/train/valid/full data subset """
        data = dataset.load_subset_data(key)
        labels = dataset.load_subset_labels(key)
        return data, labels

    def print_subset_info(dataset, key='full'):
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

    #def register_self(dataset):
    #    # creates a symlink in the junction dir
    #    register_training_dpath(dataset.training_dpath, dataset.alias_key)

    #def save_alias(dataset, alias_key):
    #    # shortcut to the cached information so we dont need to
    #    # compute hotspotter matching hashes. There is a change data
    #    # can get out of date while this is enabled.
    #    alias_fpath = get_alias_dict_fpath()
    #    alias_dict = ut.text_dict_read(alias_fpath)
    #    data_dict = dataset.asdict()
    #    alias_dict[alias_key] = data_dict
    #    ut.text_dict_write(alias_fpath, alias_dict)

    @ut.memoize
    def load_subset_data(dataset, key='full'):
        data_fpath = dataset.fpath_dict[key]['data']
        data = ut.load_data(data_fpath, verbose=True)
        if len(data.shape) == 3:
            # add channel dimension for implicit grayscale
            data.shape = data.shape + (1,)
        return data

    @ut.memoize
    def load_subset_labels(dataset, key='full'):
        labels_fpath = dataset.fpath_dict[key]['labels']
        labels = (None if labels_fpath is None
                  else ut.load_data(labels_fpath, verbose=True))
        return labels

    @ut.memoize
    def load_subset_metadata(dataset, key='full'):
        metadata_fpath = dataset.fpath_dict[key].get('metadata', None)
        if metadata_fpath is not None:
            flat_metadata = ut.load_data(metadata_fpath, verbose=True)
        else:
            flat_metadata = None
        return flat_metadata

    def clear_cache(dataset, key=None):
        cached_func_list = [
            dataset.load_subset_data,
            dataset.load_subset_labels,
            dataset.load_subset_metadata,
        ]
        if key is None:
            for cached_func in cached_func_list:
                cached_func.cache.clear()
        else:
            for cached_func in cached_func_list:
                del cached_func.cache[key]

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

    def interact(dataset, key='full', **kwargs):
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
            labels, data, metadata, dataset.data_info['data_per_label'],
            **interact_kw)

    def view_directory(dataset):
        ut.view_directory(dataset.xdata_dpath)

    vd = view_directory

    def has_split(dataset, key):
        return key in dataset.fpath_dict

    def get_split_fmtstr(dataset, forward=False):
        # Parse direction
        parse_fmtstr = '{key}_{size:d}_{type_:w}{ext}'
        if forward:
            # hack, need to do actual parsing of the parser here
            def parse_inverse_format(parse_fmtstr):
                # if True:
                # hack impl
                return parse_fmtstr.replace(':w}', '}')
                # else:
                #     # Try and make a better impl
                #     nestings = ut.parse_nestings(parse_fmtstr, only_curl=True)
                #     ut.recombine_nestings(nestings)

            # Generate direction
            fmtstr = parse_inverse_format(parse_fmtstr)
        else:
            fmtstr = parse_fmtstr
        return fmtstr

    def load_splitsets(dataset):
        import parse
        fpath_dict = {}
        fmtstr = dataset.get_split_fmtstr(forward=False)
        for fpath in ut.ls(dataset.split_dpath):
            parsed = parse.parse(fmtstr, basename(fpath))
            if parsed is None:
                print('WARNING: invalid filename %r' % (fpath,))
                continue
            key = parsed['key']
            type_ = parsed['type_']
            splitset = fpath_dict.get(key, {})
            splitset[type_] = fpath
            fpath_dict[key] = splitset
        # check validity of loaded data
        for key, val in fpath_dict.items():
            assert 'data' in val, 'subset missing data'
        dataset.fpath_dict.update(**fpath_dict)

    def load(dataset):
        dataset.ensure_dirs()
        if not exists(dataset.info_fpath):
            raise IOError('dataset info manifest cache miss')
        else:
            dataset.data_info = ut.load_data(dataset.info_fpath)
        if not exists(dataset.data_fpath):
            raise IOError('dataset data cache miss')
        dataset.load_splitsets()
        # Hack
        if not exists(dataset.fpath_dict['full']['metadata']):
            dataset.fpath_dict['full']['metadata'] = None

    def save(dataset, data, labels, metadata=None, data_per_label=1):
        ut.save_data(dataset.data_fpath, data)
        ut.save_data(dataset.labels_fpath, labels)
        if metadata is not None:
            ut.save_data(dataset.metadata_fpath, metadata)
        else:
            dataset.fpath_dict['full']['metadata'] = None
        # cache the data because it is likely going to be used to define a
        # splitset
        dataset.load_subset_data.cache['full'] = data
        dataset.load_subset_labels.cache['full'] = labels
        dataset.load_subset_metadata.cache['full'] = metadata
        # Infer the rest of the required data info
        dataset.data_info['num_labels'] = len(labels)
        dataset.data_info['unique_labels'] = np.unique(labels)
        dataset.data_info['data_per_label'] = data_per_label
        ut.save_data(dataset.info_fpath, dataset.data_info)

    def add_split(dataset, key, idxs):
        print('[dataset] adding split %r' % (key,))
        # Build subset filenames
        ut.ensuredir(dataset.split_dpath)
        ext = dataset._ext
        fmtdict = dict(key=key, ext=ext, size=len(idxs))
        fmtstr = dataset.get_split_fmtstr(forward=True)
        splitset = {
            type_: join(dataset.split_dpath, fmtstr.format(type_=type_, **fmtdict))
            for type_ in ['data', 'labels', 'metadata']
        }
        # Partition data into the subset
        part_dict = {
            'data': dataset.data.take(idxs, axis=0),
            'labels': dataset.labels.take(idxs, axis=0),
        }
        if dataset.metadata is not None:
            taker = ut.partial(ut.take, index_list=idxs)
            part_dict['metadata'] = ut.map_dict_vals(taker, dataset.metadata)
        # Write splitset data to files
        for type_ in part_dict.keys():
            ut.save_data(splitset[type_], part_dict[type_])
        # Register filenames with dataset
        dataset.fpath_dict[key] = splitset

    # def build_auxillary_data(dataset):
    #     # Make test train validatation sets
    #     data_fpath = dataset.data_fpath
    #     labels_fpath = dataset.labels_fpath
    #     metadata_fpath = dataset.metadata_fpath
    #     data_per_label = dataset.data_per_label
    #     split_names = ['train', 'test', 'valid']
    #     fractions = [.7, .2, .1]
    #     named_split_fpath_dict = ondisk_data_split(
    #         data_fpath, labels_fpath, metadata_fpath,
    #         data_per_label, split_names, fractions,
    #     )
    #     for key, val in named_split_fpath_dict.items():
    #         splitset = dataset.fpath_dict.get(key, {})
    #         splitset.update(**val)
    #         dataset.fpath_dict[key] = splitset


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
    """
    Creates a symlink to the training path in the training junction
    """
    junction_dpath = get_juction_dpath()
    training_dname = basename(training_dpath)
    if alias_key is not None:
        # hack for a bit more readable pathname
        prefix = alias_key.split(';')[0].replace(' ', '')
        training_dname = prefix + '_' + training_dname
    training_dlink = join(junction_dpath, training_dname)
    ut.symlink(training_dpath, training_dlink)


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


def stratified_label_shuffle_split(y, labels, fractions, idx=None, rng=None):
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
    rng = ut.ensure_rng(rng)
    #orig_y = y
    unique_labels, groupxs = ut.group_indices(labels)
    grouped_ys = ut.apply_grouping(y, groupxs)
    # Assign each group a probabilistic class
    unique_ys = [ys[rng.randint(0, len(ys))] for ys in grouped_ys]
    # TODO: should weight the following selection based on size of group
    #class_weights = [ut.dict_hist(ys) for ys in grouped_ys]

    unique_idxs = stratified_shuffle_split(unique_ys, fractions, rng)
    index_sets = [np.array(ut.flatten(ut.take(groupxs, idxs))) for idxs in unique_idxs]
    if idx is not None:
        # These indicies subindex into parent set of indicies
        index_sets = [np.take(idx, idxs, axis=0) for idxs in index_sets]
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
