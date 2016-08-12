# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from os.path import join, basename, exists
import six
import numpy as np
import utool as ut
print, rrr, profile = ut.inject2(__name__, '[ibeis_cnn.datset]')


@six.add_metaclass(ut.ReloadingMetaclass)
class DataSet(ut.NiceRepr):
    """
    helper class for managing dataset paths and general metadata

    SeeAlso:
        python -m ibeis_cnn.ingest_data --test-get_ibeis_part_siam_dataset --show

    CommandLine:
        python -m ibeis_cnn.dataset DataSet

    Example:
        >>> from ibeis_cnn.ingest_data import *  # NOQA
        >>> dataset = grab_mnist_category_dataset()
        >>> dataset.print_dir_structure()
        >>> # ----
        >>> from ibeis_cnn.models import MNISTModel
        >>> model = MNISTModel(batch_size=128, data_shape=(24, 24, 1),
        >>>                    output_dims=10, dataset_dpath=dataset.dataset_dpath)
        >>> model.print_structure()
    """
    def __init__(dataset, cfgstr=None, training_dpath='.', data_shape=None,
                 num_data=None, name=None, ext='.pkl'):
        dataset.name = name
        dataset.cfgstr = cfgstr
        dataset.training_dpath = training_dpath
        assert data_shape is not None, 'must specify'
        dataset._ext = ext
        dataset._info = {
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

    def __nice__(dataset):
        return '(' + dataset.dataset_id + ')'

    @property
    def hashid(dataset):
        if dataset.cfgstr is None:
            return ''
        else:
            return ut.hashstr27(dataset.cfgstr, hashlen=8)

    @property
    def dataset_id(dataset):
        shape_str = 'x'.join(ut.lmap(str, dataset._info['data_shape']))
        num_data = dataset._info['num_data']
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
    def dataset_dpath(dataset):
        return join(dataset.training_dpath, dataset.dataset_id)

    @property
    def split_dpath(dataset):
        split_dpath = join(dataset.dataset_dpath, 'splits')
        return split_dpath

    @property
    def full_dpath(dataset):
        return join(dataset.dataset_dpath, 'full')

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
            dataset.ensure_symlinked()
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

    def subset(dataset, key):
        """ loads a test/train/valid/full data subset """
        data = dataset.subset_data(key)
        labels = dataset.subset_labels(key)
        return data, labels

    def print_subset_info(dataset, key='full'):
        data, labels = dataset.subset(key)
        dataset.print_dataset_info(data, labels, key)

    @property
    def data_shape(dataset):
        data_shape = dataset._info['data_shape']
        assert data_shape is not None, 'data_shape is unknown'
        return data_shape

    @property
    def unique_labels(dataset):
        unique_labels = dataset._info['unique_labels']
        assert unique_labels is not None, 'unique_labels is unknown'
        return unique_labels

    @property
    def labels(dataset):
        return dataset.subset_labels()

    @property
    def data(dataset):
        return dataset.subset_data()

    @property
    def metadata(dataset):
        return dataset.subset_metadata()

    def asdict(dataset):
        # save all args passed into constructor as a dict
        key_list = ut.get_func_argspec(dataset.__init__).args[1:]
        data_dict = ut.dict_subset(dataset.__dict__, key_list)
        return data_dict

    @ut.memoize
    def subset_data(dataset, key='full'):
        data_fpath = dataset.fpath_dict[key]['data']
        data = ut.load_data(data_fpath, verbose=True)
        if len(data.shape) == 3:
            # add channel dimension for implicit grayscale
            data.shape = data.shape + (1,)
        return data

    @ut.memoize
    def subset_labels(dataset, key='full'):
        labels_fpath = dataset.fpath_dict[key]['labels']
        labels = (None if labels_fpath is None
                  else ut.load_data(labels_fpath, verbose=True))
        return labels

    @ut.memoize
    def subset_metadata(dataset, key='full'):
        metadata_fpath = dataset.fpath_dict[key].get('metadata', None)
        if metadata_fpath is not None:
            flat_metadata = ut.load_data(metadata_fpath, verbose=True)
        else:
            flat_metadata = None
        return flat_metadata

    def clear_cache(dataset, key=None):
        cached_func_list = [
            dataset.subset_data,
            dataset.subset_labels,
            dataset.subset_metadata,
        ]
        if key is None:
            for cached_func in cached_func_list:
                cached_func.cache.clear()
        else:
            for cached_func in cached_func_list:
                if key in cached_func.cache:
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
        data     = dataset.subset_data(key)
        labels   = dataset.subset_labels(key)
        metadata = dataset.subset_metadata(key)
        return interact_func(
            labels, data, metadata, dataset._info['data_per_label'],
            **interact_kw)

    def view_directory(dataset):
        ut.view_directory(dataset.dataset_dpath)

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
        dataset.ensure_symlinked()
        if not exists(dataset.info_fpath):
            raise IOError('dataset info manifest cache miss')
        else:
            dataset._info = ut.load_data(dataset.info_fpath)
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
        dataset.subset_data.cache['full'] = data
        dataset.subset_labels.cache['full'] = labels
        dataset.subset_metadata.cache['full'] = metadata
        # Infer the rest of the required data info
        dataset._info['num_labels'] = len(labels)
        dataset._info['unique_labels'] = np.unique(labels)
        dataset._info['data_per_label'] = data_per_label
        ut.save_data(dataset.info_fpath, dataset._info)

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

    def ensure_symlinked(dataset):
        """
        Creates a symlink to the training path in the training junction
        """
        junction_dpath = get_juction_dpath()
        dataset_dname = basename(dataset.dataset_dpath)
        dataset_dlink = join(junction_dpath, dataset_dname)
        ut.symlink(dataset.dataset_dpath, dataset_dlink)

    def ensure_dirs(dataset):
        ut.ensuredir(dataset.full_dpath)
        ut.ensuredir(dataset.split_dpath)

    def print_dir_structure(dataset):
        print(dataset.training_dpath)
        print(dataset.dataset_dpath)
        print(dataset.data_fpath)
        print(dataset.labels_fpath)
        print(dataset.metadata_fpath)
        print(dataset.info_fpath)
        print(dataset.full_dpath)
        print(dataset.split_dpath)

    def print_dir_tree(dataset):
        fpaths = ut.glob(dataset.dataset_dpath, '*', recursive=True)
        print('\n'.join(sorted(fpaths)))

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


def stratified_kfold_label_split(y, labels, n_folds=2, idx=None, rng=None):
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

    import sklearn.cross_validation
    xvalkw = dict(n_folds=n_folds, shuffle=True, random_state=43432)
    skf = sklearn.cross_validation.StratifiedKFold(unique_ys, **xvalkw)
    _iter = skf

    folded_index_sets = []

    for label_idx_set in _iter:
        index_sets = [np.array(ut.flatten(ut.take(groupxs, idxs)))
                      for idxs in label_idx_set]
        if idx is not None:
            # These indicies subindex into parent set of indicies
            index_sets = [np.take(idx, idxs, axis=0) for idxs in index_sets]
        folded_index_sets.append(index_sets)

    for train_idx, test_idx in folded_index_sets:
        train_labels = set(ut.take(labels, train_idx))
        test_labels = set(ut.take(labels, test_idx))
        assert len(test_labels.intersection(train_labels)) == 0, 'same labels appeared in both train and test'
        pass
    #import sklearn.model_selection
    #skf = sklearn.model_selection.StratifiedKFold(**xvalkw)
    #_iter = skf.split(X=np.empty(len(target)), y=target)

    #unique_idxs = stratified_shuffle_split(unique_ys, fractions, rng)
    #index_sets = [np.array(ut.flatten(ut.take(groupxs, idxs))) for idxs in unique_idxs]
    #if idx is not None:
    #    # These indicies subindex into parent set of indicies
    #    index_sets = [np.take(idx, idxs, axis=0) for idxs in index_sets]
    return folded_index_sets


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
