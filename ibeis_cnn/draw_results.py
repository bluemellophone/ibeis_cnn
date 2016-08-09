from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import numpy as np
import six
print, rrr, profile = ut.inject2(__name__, '[ibeis_cnn.draw_results]')


def interact_siamsese_data_patches(labels, data, flat_metadata, **kwargs):
    r"""
    Args:
        labels (?):
        data (?):
        flat_metadata (?):

    CommandLine:
        python -m ibeis_cnn.draw_results --test-interact_siamsese_data_patches --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.draw_results import *  # NOQA
        >>> from ibeis_cnn import ingest_data
        >>> data, labels = ingest_data.testdata_patchmatch2()
        >>> flat_metadata = {'fs': np.arange(len(labels))}
        >>> result = interact_siamsese_data_patches(labels, data, flat_metadata)
        >>> ut.show_if_requested()
    """
    (warped_patch1_list, warped_patch2_list) = list(zip(*ut.ichunks(data, 2)))
    data_lists = (warped_patch1_list, warped_patch2_list)
    interact_patches(labels, data_lists, flat_metadata, **kwargs)


def interact_dataset(labels, data, flat_metadata, data_per_label, **kwargs):
    r"""
    Args:
        labels (?):
        data (?):
        flat_metadata (?):

    CommandLine:
        python -m ibeis_cnn.draw_results --test-interact_dataset --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.draw_results import *  # NOQA
        >>> from ibeis_cnn import ingest_data
        >>> data, labels = ingest_data.testdata_patchmatch2()
        >>> flat_metadata = {'fs': np.arange(len(labels))}
        >>> result = interact_siamsese_data_patches(labels, data, flat_metadata)
        >>> ut.show_if_requested()
    """
    data_lists = list(zip(*ut.ichunks(data, data_per_label)))
    interact_patches(labels, data_lists, flat_metadata, **kwargs)


def interact_patches(label_list, data_lists,
                     flat_metadata, sortby=None, figtitle=None,
                     chunck_sizes=None, ibs=None, hack_one_per_aid=True,
                     qreq_=None):
    r"""
    Args:
        label_list (list):
        data_lists (tuple of lists): fixme bad name
        flat_metadata (?):
        sortby (None): (default = None)
        figtitle (None): (default = None)
        chunck_sizes (None): (default = None)

    CommandLine:
        python -m ibeis_cnn.draw_results --exec-interact_patches --show

    SeeAlso:
        python -m ibeis_cnn.ingest_ibeis --exec-extract_annotpair_training_chips --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.draw_results import *  # NOQA
        >>> from ibeis_cnn import ingest_data
        >>> data, labels = ingest_data.testdata_patchmatch2()
        >>> flat_metadata = {'fs': np.arange(len(labels))}
        >>> result = interact_siamsese_data_patches(labels, data, flat_metadata)
        >>> ut.show_if_requested()
        >>> print(result)
    """
    import vtool as vt
    #from ibeis.viz import viz_helpers as vh
    print('Building patch interaction')
    num_datas = list(map(len, data_lists))
    num_data = num_datas[0]
    ut.assert_all_eq(num_datas)
    print('num_datas = %r' % (num_data,))
    if label_list is not None:
        assert len(label_list) == num_data, 'datas must be corresponding'
        print('len(label_list) = %r' % (len(label_list),))

    #chunck_sizes = (6, 8)
    if chunck_sizes is None:
        chunck_sizes = (6, 10)
        #chunck_sizes = (3, 3)

    # Check out the score pdfs
    print('sortby = %r' % (sortby,))
    if sortby is not None:
        if sortby == 'fs':
            index_list = ut.list_argsort(flat_metadata['fs'])[::-1]
        elif sortby == 'rand':
            index_list = ut.random_indexes(len(label_list))
        else:
            raise NotImplementedError('sortby = %r' % (sortby,))
    else:
        unique_labels, groupxs = vt.group_indices(label_list)
        idx_lists = groupxs[::-1]

        if flat_metadata is not None:
            if 'fs' in flat_metadata:
                rng = np.random.RandomState(0)
                # Re order idx lists by score if available
                # weighted random sorting

                fs_lists = [flat_metadata['fs'].take(idxs) for idxs in idx_lists]
                randfs_lists = [fs * rng.rand(*fs.shape) for fs in fs_lists]
                sortx_lists = [rfs.argsort()[::-1] for rfs in randfs_lists]
                sortx_lists = [flat_metadata['fs'].take(idxs).argsort()[::-1] for idxs in idx_lists]
                idx_lists = vt.ziptake(idx_lists, sortx_lists)

                # FILTER TO ONLY SHOW ONE PER AID
                if hack_one_per_aid and 'aid_pairs' in flat_metadata:
                    print('hacking one per aid')
                    aid_pairs = flat_metadata['aid_pairs']
                    dataids = vt.get_undirected_edge_ids(aid_pairs)
                    new_idx_lists = []
                    for idxs in idx_lists:
                        unique_xs = np.unique(dataids.take(idxs), return_index=True)[1]
                        unique_xs.sort()
                        new_idx_lists.append(idxs.take(unique_xs))
                    idx_lists = new_idx_lists

        index_list = ut.flatten(list(ut.interleave(
            [ut.ichunks(idxs, chunck_sizes[1] * chunck_sizes[0] // 2)
             for idxs in idx_lists])))
        #index_list0 = list(ut.interleave(idx_lists))
        #index_list = list(range(len(label_list)))

    assert len(chunck_sizes) == 2

    draw_meta = not ut.get_argflag('--nometa')
    draw_hud = not ut.get_argflag('--nohud')

    interact = make_InteractSiamPatches(ibs, data_lists, label_list,
                                        flat_metadata, chunck_sizes,
                                        index_list, draw_meta=draw_meta,
                                        draw_hud=draw_hud, qreq_=qreq_)

    interact.show_page()
    return interact


def make_InteractSiamPatches(*args, **kwargs):
    import plottool as pt
    from plottool import abstract_interaction
    BASE_CLASS = abstract_interaction.AbstractPagedInteraction

    class InteractSiamPatches(BASE_CLASS):
        def __init__(self, ibs, data_lists, label_list, flat_metadata,
                     chunck_sizes, index_list=None, figtitle=None,
                     draw_meta=True, qreq_=None, **kwargs):
            self.nCols = chunck_sizes[0]
            if index_list is None:
                index_list = list(range(label_list))
            print('len(index_list) = %r' % (len(index_list),))
            print('len(label_list) = %r' % (len(label_list),))
            print('chunck_sizes = %r' % (chunck_sizes,))
            self.multi_chunked_indicies = list(ut.iter_multichunks(index_list, chunck_sizes))
            # print('ut.depth_profile(self.multi_chunked_indicies) = %r' % (ut.depth_profile(self.multi_chunked_indicies),))
            nPages = len(self.multi_chunked_indicies)
            self.data_lists = data_lists
            self.figtitle = figtitle
            self.label_list = label_list
            self.flat_metadata = flat_metadata
            self.draw_meta = draw_meta
            self.qreq_ = qreq_
            self.ibs = ibs
            super(InteractSiamPatches, self).__init__(nPages, **kwargs)

        def show_page(self, pagenum=None, **kwargs):
            self._ensure_running()
            if pagenum is None:
                pagenum = self.current_pagenum
            else:
                self.current_pagenum = pagenum
            self.prepare_page()
            # print('pagenum = %r' % (pagenum,))
            next_pnum = pt.make_pnum_nextgen(1, self.nCols)
            self.multiindicies = self.multi_chunked_indicies[self.current_pagenum]
            self.offset_lists = []
            self.sizes_lists = []
            self.sf_lists = []
            self.ax_list = []
            for indicies in self.multiindicies:
                img, offset_list, sf_list, sizes_list = get_patch_chunk(
                    self.data_lists, self.label_list,
                    self.flat_metadata, indicies, draw_meta=self.draw_meta)
                self.offset_lists.append(offset_list)
                self.sf_lists.append(sf_list)
                self.sizes_lists.append(sizes_list)
                # TODO; use offset_list for interaction
                pt.imshow(img, pnum=next_pnum())
                ax = pt.gca()
                self.ax_list.append(ax)
            if self.figtitle is not None:
                #print(self.figtitle)
                pt.set_figtitle(self.figtitle)
                pass

        def on_click_inside(self, event, ax):
            print('click inside')

            def get_label_index(self, event, ax):
                """ generalize """
                x, y = event.xdata, event.ydata
                def find_offset_index(offset_list, size_list, x, y):
                    x1_pts = offset_list.T[0]
                    x2_pts = offset_list.T[0] + size_list.T[0]
                    y1_pts = offset_list.T[1]
                    y2_pts = offset_list.T[1] + size_list.T[1]

                    in_bounds = np.logical_and.reduce([
                        x >= x1_pts, x < x2_pts,
                        y >= y1_pts, y < y2_pts
                    ])
                    valid_idxs = np.where(in_bounds)[0]
                    #print('valid_idxs = %r' % (valid_idxs,))
                    assert len(valid_idxs) == 1
                    return valid_idxs[0]
                try:
                    plot_index = self.ax_list.index(ax)
                except ValueError:
                    label_index = None
                else:
                    offset_list = np.array(self.offset_lists[plot_index])
                    sf_list = np.array(self.sf_lists[plot_index])
                    orig_size_list = np.array(self.sizes_lists[plot_index])
                    size_list = np.array(orig_size_list) * sf_list
                    num_cols = len(self.data_lists)  # data per label
                    num_rows = (len(offset_list) // num_cols)
                    _subindex = find_offset_index(
                        offset_list, size_list, x, y)
                    row_index = _subindex % num_rows
                    col_index = _subindex // num_rows
                    label_index = self.multiindicies[plot_index][row_index]
                    print('_subindex = %r' % (_subindex,))
                    print('row_index = %r' % (row_index,))
                    print('col_index = %r' % (col_index,))
                    print('label_index = %r' % (label_index,))
                return label_index

            def embed_ipy(self=self, event=event, ax=ax):
                ut.embed()
                len(self.data_lists[0])
                len(self.data_lists[1])

            def dostuff(self=self, event=event, ax=ax):
                label_index = get_label_index(self, event, ax)
                d1 = self.data_lists[0][label_index]
                d2 = self.data_lists[1][label_index]
                import plottool as pt
                pt.imshow(d1, pnum=(2, 1, 1), fnum=2)
                pt.imshow(d2, pnum=(2, 1, 2), fnum=2)
                pt.update()

            options = []

            label_index = get_label_index(self, event, ax)

            if label_index is not None:
                if self.label_list is not None:
                    print('self.label_list[%d] = %r' % (label_index, self.label_list[label_index]))

                if self.flat_metadata is not None:
                    for key, val in self.flat_metadata.items():
                        if len(val) == len(self.label_list):
                            print('self.flat_metadata[%s][%d] = %r' % (
                                key, label_index, val[label_index]) )

                    if 'aid_pairs' in self.flat_metadata:
                        aid1, aid2 = self.flat_metadata['aid_pairs'][label_index]
                        from ibeis.gui import inspect_gui

                        print(ut.repr3(self.ibs.get_annot_info(
                            [aid1], reference_aid=[aid2], case_tags=True,
                            match_tags=True, timedelta=True)))
                        print(ut.repr3(self.ibs.get_annot_info(
                            [aid2], reference_aid=[aid1], case_tags=True,
                            match_tags=True, timedelta=True)))

                        if self.ibs is not None:
                            options += inspect_gui.get_aidpair_context_menu_options(
                                self.ibs, aid1, aid2, None, qreq_=self.qreq_,
                            )
                            #update_callback=update_callback,
                            #backend_callback=backend_callback, aid_list=aid_list)

            if event.button == 3:
                #options = self.context_option_funcs[index]()
                options += [
                    ('Embed', embed_ipy),
                    ('Present', pt.present),
                    ('dostuff', dostuff),
                ]
                self.show_popup_menu(options, event)
    return InteractSiamPatches(*args, **kwargs)


def get_sample_pairimg_from_X(Xb, index_list):
    warped_patch1_list, warped_patch2_list = Xb[::2], Xb[1::2]
    label_list = None
    tup = get_patch_sample_img(warped_patch1_list, warped_patch2_list, label_list, {}, index_list, (len(index_list), 1))
    stacked_img, stacked_offsets, stacked_sfs = tup
    return stacked_img


def get_patch_sample_img(warped_patch1_list, warped_patch2_list, label_list,
                         flat_metadata, index_list, chunck_sizes=(6, 10)):
    #with ut.eoxc
    try:
        multiindices = six.next(ut.iter_multichunks(index_list, chunck_sizes))
        data_lists = [warped_patch1_list, warped_patch2_list]
        tup = get_patch_multichunks(data_lists, label_list, flat_metadata,
                                    multiindices)
        stacked_img, stacked_offsets, stacked_sfs = tup
        return stacked_img, stacked_offsets, stacked_sfs
    except StopIteration:
        if len(index_list) > 0:
            raise
        import vtool as vt
        errorimg = vt.get_no_symbol()
        return errorimg, [], []


def get_patch_multichunks(data_lists, label_list, flat_metadata, multiindicies):
    """

    CommandLine:
        python -m ibeis_cnn.draw_results --test-get_patch_multichunks --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_cnn.draw_results import *  # NOQA
        >>> from ibeis_cnn import ingest_data
        >>> data, label_list = ingest_data.testdata_patchmatch2()
        >>> flat_metadata = {}
        >>> warped_patch1_list, warped_patch2_list = list(zip(*ut.ichunks(data, 2)))
        >>> data_lists = (warped_patch1_list, warped_patch2_list, )
        >>> multiindicies = [np.arange(0, 10), np.arange(10, 20), np.arange(20, 30) ]
        >>> stacked_img, stacked_offsets, stacked_sfs = get_patch_multichunks(data_lists, label_list, flat_metadata, multiindicies)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> pt.imshow(stacked_img)
        >>> ut.show_if_requested()
    """
    multiimg_list = []
    offsets_list = []
    sfs_list = []
    border_color = (100, 10, 10)  # bgr, darkblue
    for indicies in multiindicies:
        img, offset_list, sf_list, stacked_orig_sizes = get_patch_chunk(
            data_lists, label_list, flat_metadata,
            indicies, border_color)
        multiimg_list.append(img)
        offsets_list.append(offset_list)
        sfs_list.append(sf_list)
        # Add horizontal spacing

        solidbar = np.zeros((img.shape[0], int(img.shape[1] * .1), 3), dtype=img.dtype)
        if ut.is_float(solidbar):
            solidbar[:, :, :] = (np.array(border_color) / 255)[None, None]
        else:
            solidbar[:, :, :] = np.array(border_color)[None, None]
        multiimg_list.append(solidbar)
        offsets_list.append([(0, 0)])
        sfs_list.append([(1., 1.)])

    # remove last horizontal bar
    multiimg_list = multiimg_list[:-1]
    offsets_list = offsets_list[:-1]
    sfs_list = sfs_list[:-1]

    import vtool as vt
    stacked_img, stacked_offsets, stacked_sfs = vt.stack_multi_images2(multiimg_list, offsets_list, sfs_list, vert=False)
    return stacked_img, stacked_offsets, stacked_sfs

    # TODO; use offset_list for interaction


def get_patch_chunk(data_lists, label_list,
                    flat_metadata, indicies=None, border_color=(0, 0, 0),
                    draw_meta=True,
                    vert=True, fontScale=2.5):
    """
    indicies = chunked_indicies[0]

    Args:
        data_lists (list) : list of lists. The first dimension is the data_per_label dim.
        warped_patch1_list (list):
        warped_patch2_list (list):
        label_list (list):
        flat_metadata (?):
        indicies (?):

    CommandLine:
        python -m ibeis_cnn.draw_results --test-get_patch_chunk --show
        python -m ibeis_cnn.draw_results --test-get_patch_chunk

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_cnn.draw_results import *  # NOQA
        >>> from ibeis_cnn import ingest_data
        >>> data, label_list = ingest_data.testdata_patchmatch2()
        >>> flat_metadata = {'fs': np.arange(len(label_list))}
        >>> data_lists =  list(zip(*ut.ichunks(data, 2)))
        >>> indicies = np.arange(0, 10)
        >>> img, offset_list, sf_list, stacked_orig_sizes = get_patch_chunk(data_lists, label_list, flat_metadata, indicies)
        >>> result = str(img.shape)
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> pt.imshow(img)
        >>> ut.show_if_requested()
        (1920, 384, 3)
    """
    if indicies is None:
        indicies = list(range(len(label_list)))
    #warped_patch1_list, warped_patch2_list = data_lists
    data_per_label = len(data_lists)
    import utool as ut
    import vtool as vt
    import plottool as pt

    # Ipython embed hates dict comprehensions and locals
    #with ut.embed_on_exception_context:
    if flat_metadata is None:
        flat_metadata_subset = {}
    else:
        flat_metadata_subset = dict([(key, ut.take(vals, indicies))
                                     for key, vals in six.iteritems(flat_metadata)])

    #import utool
    #with utool.embed_on_exception_context:
    patch_list_subsets_ = [
        [vt.ensure_3channel(patch)
         for patch in ut.take(warped_patch_list, indicies)]
        for warped_patch_list in data_lists
    ]

    thickness = 2
    if label_list is not None:
        # draw label border
        label_list_subset = ut.take(label_list, indicies)
        if data_per_label in [1, 2]:
            #truecol  = tuple(pt.color_funcs.to_base255(pt.TRUE_GREEN)[0:3])[::-1]
            truecol  = tuple(pt.color_funcs.to_base255(pt.TRUE_BLUE)[0:3])[::-1]
            falsecol = tuple(pt.color_funcs.to_base255(pt.FALSE_RED)[0:3])[::-1]
            unknncol = tuple(pt.color_funcs.to_base255(pt.PURPLE)[0:3])[::-1]
            colorfn = [falsecol, truecol, unknncol]
        else:
            unique_labels = np.unique(label_list)
            unique_colors = [tuple(pt.color_funcs.to_base255(color)[0:3])[::-1]
                             for color in pt.distinct_colors(len(unique_labels))]
            colorfn = dict(zip(unique_labels, unique_colors))

        num_labels = len(np.unique(label_list))
        if num_labels > 3:
            colorfn = pt.distinct_colors(num_labels)

        patch_list_subsets = [
            [vt.draw_border(patch, color=colorfn[label], thickness=thickness, out=patch)
             for label, patch in zip(label_list_subset, patch_list)]
            for patch_list in patch_list_subsets_
        ]
    else:
        patch_list_subsets = patch_list_subsets_
    del patch_list_subsets_

    # draw black border
    patch_lists = [
        [vt.draw_border(patch, color=border_color, thickness=1, out=patch)
         for patch in patch_list]
        for patch_list in patch_list_subsets
    ]
    patchsize_lists = [
        [vt.get_size(patch) for patch in patch_list]
        for patch_list in patch_lists
    ]
    stacked_orig_sizes = ut.flatten(patchsize_lists)

    # stack into single image
    stack_kw = dict(modifysize=False, return_offset=True, return_sf=True)
    stacktup_list = [
        vt.stack_image_list(patch_list, vert=vert, **stack_kw)
        for patch_list in patch_lists
    ]

    multiimg_list = ut.get_list_column(stacktup_list, 0)
    offsets_list = ut.get_list_column(stacktup_list, 1)
    sfs_list = ut.get_list_column(stacktup_list, 2)

    stacked_patches, offset_list, sf_list = vt.stack_multi_images2(multiimg_list, offsets_list, sfs_list, vert=not vert, modifysize=False)

    if False:
        stacked_patches_, offset_list_, sf_list_ = vt.stack_multi_images(
            multiimg_list[0], multiimg_list[1], offsets_list[0], sfs_list[0], offsets_list[0], sfs_list[1],
            vert=not vert, modifysize=False)
        #stacked_orig_sizes = patchsize1_list + patchsize2_list

    # Draw scores
    patch_texts = None
    if draw_meta is True:
        if 'fs' in flat_metadata_subset:
            scores = flat_metadata_subset['fs']
            patch_texts = ['%.3f' % s for s in scores]
            #right_offsets = offset_list[len(offset_list) // 2:]
        if 'text' in flat_metadata_subset:
            patch_texts = flat_metadata_subset['text']
    elif isinstance(draw_meta, list):
        requested_text = []
        for key in draw_meta:
            if key in flat_metadata_subset:
                col_text = [ut.repr2(v, precision=3) for v in flat_metadata_subset[key]]
                requested_text.append(col_text)
        patch_texts = [' '.join(t) for t in zip(*requested_text)]

    if patch_texts is not None:
        scale_up = 3
        img = vt.resize_image_by_scale(stacked_patches, scale_up)
        offset_list = np.array(offset_list) * scale_up
        sf_list = np.array(sf_list) * scale_up

        left_offsets = offset_list[:len(offset_list) // data_per_label]
        left_sfs = sf_list[:len(offset_list) // data_per_label]

        textcolor_rgb1 = pt.to_base255(pt.BLACK)[:3]
        #textcolor_rgb2 = pt.to_base255(pt.ORANGE)[:3]
        textcolor_rgb2 = pt.to_base255(pt.WHITE)[:3]
        left_patchsizees = patchsize_lists[0]
        for offset, patchsize, sf, text in zip(left_offsets, left_patchsizees, left_sfs, patch_texts):
            #print(offset)
            #print(s)
            import cv2
            scaled_offset = (np.array(offset) + np.array([thickness + 2, -thickness - 2]))
            patch_h = (np.array(patchsize) * sf)[0]
            text_bottom_left = scaled_offset + np.array([0, patch_h - 2])
            org = tuple(text_bottom_left.astype(np.int32).tolist())
            #fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL
            fontFace = cv2.FONT_HERSHEY_PLAIN
            fontkw = dict(bottomLeftOrigin=False, fontScale=fontScale, fontFace=fontFace)
            # Bordered text
            vt.draw_text(img, text, org, thickness=6, textcolor_rgb=textcolor_rgb1, **fontkw)
            vt.draw_text(img, text, org, thickness=2, textcolor_rgb=textcolor_rgb2, **fontkw)
    else:
        img = stacked_patches
    return img, offset_list, sf_list, stacked_orig_sizes


def visualize_score_separability(label_list, warped_patch1_list, warped_patch2_list, flat_metadata):
    #import vtool as vt
    import plottool as pt
    #draw_results.interact_siamsese_data_patches(fp_labels, fp_data, {'fs': fp_scores}, rand=False, figtitle='FP')
    from vtool import score_normalization as scorenorm
    tp_support = np.array(ut.compress(flat_metadata['fs'], label_list)).astype(np.float64)
    tn_support = np.array(ut.compress(flat_metadata['fs'], ut.not_list(label_list))).astype(np.float64)
    scorenorm.test_score_normalization(tp_support, tn_support)
    #(score_domain, p_tp_given_score, p_tn_given_score, p_score_given_tp, p_score_given_tn,
    # p_score, clip_score) = scorenorm.learn_score_normalization(tp_support, tn_support, return_all=True)
    #scorenorm.inspect_pdfs(tn_support, tp_support, score_domain,
    #                       p_tp_given_score, p_tn_given_score, p_score_given_tp, p_score_given_tn, p_score, with_scores=True)
    pt.set_figtitle('HotSpotter Patch Scores')


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis_cnn.draw_results
        python -m ibeis_cnn.draw_results --allexamples
        python -m ibeis_cnn.draw_results --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
