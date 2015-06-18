from __future__ import absolute_import, division, print_function
import utool as ut
import numpy as np
import six
from ibeis_cnn import utils
print, rrr, profile = ut.inject2(__name__, '[ibeis_cnn.draw_results]')


def interact_siamese_data_fpath_patches(data_fpath, labels_fpath, flat_metadata):
    data, labels = utils.load(data_fpath, labels_fpath)
    interact_siamsese_data_patches(labels, data, flat_metadata)


def interact_siamsese_data_patches(labels, data, flat_metadata, **kwargs):
    r"""
    Args:
        labels (?):
        data (?):
        flat_metadata (?):

    CommandLine:
        python -m ibeis_cnn.draw_results --test-interact_siamsese_data_patches

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.draw_results import *  # NOQA
        >>> from ibeis_cnn import ingest_data
        >>> data, labels = ingest_data.testdata_patchmatch2()
        >>> flat_metadata = {'fs': np.arange(len(labels))}
        >>> result = interact_siamsese_data_patches(labels, data, flat_metadata)
        >>> ut.show_if_requested()
    """
    warped_patch1_list, warped_patch2_list = list(zip(*ut.ichunks(data, 2)))
    interact_patches(labels, warped_patch1_list, warped_patch2_list, flat_metadata, **kwargs)


def visualize_score_separability(label_list, warped_patch1_list, warped_patch2_list, flat_metadata):
    #import vtool as vt
    import plottool as pt
    #draw_results.interact_siamsese_data_patches(fp_labels, fp_data, {'fs': fp_scores}, rand=False, figtitle='FP')
    from vtool import score_normalization as scorenorm
    tp_support = np.array(ut.list_compress(flat_metadata['fs'], label_list)).astype(np.float64)
    tn_support = np.array(ut.list_compress(flat_metadata['fs'], ut.not_list(label_list))).astype(np.float64)
    scorenorm.test_score_normalization(tp_support, tn_support)
    #(score_domain, p_tp_given_score, p_tn_given_score, p_score_given_tp, p_score_given_tn,
    # p_score, clip_score) = scorenorm.learn_score_normalization(tp_support, tn_support, return_all=True)
    #scorenorm.inspect_pdfs(tn_support, tp_support, score_domain,
    #                       p_tp_given_score, p_tn_given_score, p_score_given_tp, p_score_given_tn, p_score, with_scores=True)
    pt.set_figtitle('HotSpotter Patch Scores')


def get_sample_pairimg_from_X(Xb, index_list):
    warped_patch1_list, warped_patch2_list = Xb[::2], Xb[1::2]
    label_list = None
    tup = get_patch_sample_img(warped_patch1_list, warped_patch2_list, label_list, {}, index_list, (len(index_list), 1))
    stacked_img, stacked_offsets, stacked_sfs = tup
    return stacked_img


def get_patch_sample_img(warped_patch1_list, warped_patch2_list, label_list, flat_metadata, index_list, chunck_sizes=(6, 10)):
    #with ut.eoxc
    try:
        multiindices = six.next(ut.iter_multichunks(index_list, chunck_sizes))
        tup = get_patch_multichunks(warped_patch1_list, warped_patch2_list, label_list, flat_metadata, multiindices)
        stacked_img, stacked_offsets, stacked_sfs = tup
        return stacked_img, stacked_offsets, stacked_sfs
    except StopIteration:
        if len(index_list) > 0:
            raise
        import vtool as vt
        errorimg = vt.get_no_symbol()
        return errorimg, [], []


def get_patch_multichunks(warped_patch1_list, warped_patch2_list, label_list, flat_metadata, multiindicies):
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
        >>> multiindicies = [np.arange(0, 10), np.arange(10, 20), np.arange(20, 30) ]
        >>> stacked_img, stacked_offsets, stacked_sfs = get_patch_multichunks(warped_patch1_list, warped_patch2_list, label_list, flat_metadata, multiindicies)
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
        img, offset_list, sf_list = get_patch_chunk(warped_patch1_list,
                                                    warped_patch2_list, label_list,
                                                    flat_metadata, indicies, border_color)
        multiimg_list.append(img)
        offsets_list.append(offset_list)
        sfs_list.append(sf_list)
        # Add horizontal spacing

        solidbar = np.zeros((img.shape[0], int(img.shape[1] * .1), 3), dtype=np.uint8)
        solidbar[:, :, :] = np.array(border_color)[None, None]
        multiimg_list.append(solidbar)
        offsets_list.append([(0, 0)])
        sfs_list.append([(1., 1.)])

    # remove last horizontal bar
    multiimg_list = multiimg_list[:-1]
    offsets_list = offsets_list[:-1]
    sfs_list = sfs_list[:-1]

    import plottool as pt
    stacked_img, stacked_offsets, stacked_sfs = pt.stack_multi_images2(multiimg_list, offsets_list, sfs_list, vert=False)
    return stacked_img, stacked_offsets, stacked_sfs

    # TODO; use offset_list for interaction


def get_patch_chunk(warped_patch1_list, warped_patch2_list, label_list,
                    flat_metadata, indicies, border_color=(0, 0, 0)):
    """
    indicies = chunked_indicies[0]

    Args:
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
        >>> warped_patch1_list, warped_patch2_list = list(zip(*ut.ichunks(data, 2)))
        >>> indicies = np.arange(0, 10)
        >>> img, offset_list, sf_list = get_patch_chunk(warped_patch1_list, warped_patch2_list, label_list, flat_metadata, indicies)
        >>> result = str(img.shape)
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> pt.imshow(img)
        >>> ut.show_if_requested()
        (640, 128, 3)

    """
    import utool as ut
    import vtool as vt
    import plottool as pt

    def ensure_colored(patch):
        if len(patch.shape) == 2:
            return np.tile(patch[None, :].T, 3)
        elif len(patch.shape) == 3 and patch.shape[-1] == 1:
            return np.tile(patch, 3)
        else:
            return patch.copy()

    green = tuple(pt.color_funcs.to_base255(pt.TRUE_GREEN)[0:3])[::-1]
    red   = tuple(pt.color_funcs.to_base255(pt.FALSE_RED)[0:3])[::-1]
    purp  = tuple(pt.color_funcs.to_base255(pt.PURPLE)[0:3])[::-1]

    patch1_list_subset = [ensure_colored(patch) for patch in ut.list_take(warped_patch1_list, indicies)]
    patch2_list_subset = [ensure_colored(patch) for patch in ut.list_take(warped_patch2_list, indicies)]

    thickness = 2
    if label_list is not None:
        # draw label border
        label_list_subset = ut.list_take(label_list, indicies)
        colorfn = [red, green, purp]
        patch1_list_subset = [
            vt.draw_border(patch, color=colorfn[label], thickness=thickness, out=patch)
            for label, patch in zip(label_list_subset, patch1_list_subset)
        ]
        patch2_list_subset = [
            vt.draw_border(patch, color=colorfn[label], thickness=thickness, out=patch)
            for label, patch in zip(label_list_subset, patch2_list_subset)
        ]

    # draw black border
    patch1_list = [vt.draw_border(patch, color=border_color, thickness=1, out=patch) for patch in patch1_list_subset]
    patch2_list = [vt.draw_border(patch, color=border_color, thickness=1, out=patch) for patch in patch2_list_subset]
    patchsize1_list = [vt.get_size(patch) for patch in patch1_list]

    # stack into single image
    stack_kw = dict(modifysize=False, return_offset=True, return_sf=True)
    stacked_patch1s, offset_list1, sf_list1 = pt.stack_image_list(patch1_list, vert=True, **stack_kw)
    stacked_patch2s, offset_list2, sf_list2 = pt.stack_image_list(patch2_list, vert=True, **stack_kw)

    stacked_patches, offset_list, sf_list = pt.stack_multi_images(
        stacked_patch1s, stacked_patch2s, offset_list1, sf_list1, offset_list2, sf_list2,
        vert=False, modifysize=False)

    # Draw scores
    # Ipython embed hates dict comprehensions and locals
    flat_metadata_subset = dict([(key, ut.list_take(vals, indicies)) for key, vals in six.iteritems(flat_metadata)])
    patch_texts = None
    if 'fs' in flat_metadata_subset:
        scores = flat_metadata_subset['fs']
        patch_texts = ['%.3f' % s for s in scores]
        #right_offsets = offset_list[len(offset_list) // 2:]
    if 'text' in flat_metadata_subset:
        patch_texts = flat_metadata_subset['text']

    if patch_texts is not None:
        scale_up = 3
        img = vt.resize_image_by_scale(stacked_patches, scale_up)
        offset_list = np.array(offset_list) * scale_up
        sf_list = np.array(sf_list) * scale_up

        left_offsets = offset_list[:len(offset_list) // 2]
        left_sfs = sf_list[:len(offset_list) // 2]

        textcolor_rgb1 = pt.to_base255(pt.BLACK)[:3]
        #textcolor_rgb2 = pt.to_base255(pt.ORANGE)[:3]
        textcolor_rgb2 = pt.to_base255(pt.WHITE)[:3]
        for offset, patchsize, sf, text in zip(left_offsets, patchsize1_list, left_sfs, patch_texts):
            #print(offset)
            #print(s)
            import cv2
            scaled_offset = (np.array(offset) + np.array([thickness + 2, -thickness - 2]))
            patch_h = (np.array(patchsize) * sf)[0]
            text_bottom_left = scaled_offset + np.array([0, patch_h - 2])
            org = tuple(text_bottom_left.astype(np.int32).tolist())
            #fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL
            fontFace = cv2.FONT_HERSHEY_PLAIN
            fontkw = dict(bottomLeftOrigin=False, fontScale=2.5, fontFace=fontFace)
            # Bordered text
            vt.draw_text(img, text, org, thickness=6, textcolor_rgb=textcolor_rgb1, **fontkw)
            vt.draw_text(img, text, org, thickness=2, textcolor_rgb=textcolor_rgb2, **fontkw)
    else:
        img = stacked_patches
    return img, offset_list, sf_list


def interact_patches(label_list, warped_patch1_list, warped_patch2_list, flat_metadata, sortby=None, figtitle=None):
    #from ibeis.viz import viz_helpers as vh
    import plottool as pt

    # Define order of iteration
    #index_list = list(range(len(label_list)))
    #index_list = ut.list_argsort(flat_metadata['fs'])[::1]

    # Check out the score pdfs
    if sortby is not None:
        if sortby == 'fs':
            index_list = ut.list_argsort(flat_metadata['fs'])[::-1]
        elif sortby == 'rand':
            index_list = ut.random_indexes(len(label_list))
        else:
            raise NotImplementedError('sortby = %r' % (sortby,))
    else:
        index_list = list(range(len(label_list)))

    #chunck_sizes = (6, 8)
    chunck_sizes = (6, 10)
    multi_chunked_indicies = list(ut.iter_multichunks(index_list, chunck_sizes))

    #pt.imshow(img)
    #ax = pt.gca()
    #ax.invert_yaxis()
    #pt.iup()
    #cv2.putText(stacked_patches, s, offset)

    #ut.embed()
    fnum = None
    if fnum is None:
        fnum = pt.next_fnum()

    once_ = True

    for multiindicies in ut.InteractiveIter(multi_chunked_indicies, display_item=False):
        assert len(chunck_sizes) == 2
        nCols = chunck_sizes[0]
        next_pnum = pt.make_pnum_nextgen(1, nCols)
        fig = pt.figure(fnum=fnum, pnum=(1, 1, 1), doclf=True)
        #ut.embed()
        for indicies in multiindicies:
            img, offset_list, sf_list = get_patch_chunk(warped_patch1_list,
                                                        warped_patch2_list, label_list,
                                                        flat_metadata, indicies)
            # TODO; use offset_list for interaction
            pt.imshow(img, pnum=next_pnum())
            # FIXME
            #pt.gca().invert_yaxis()
        #pt.update()
        if figtitle is not None:
            print(figtitle)
            pt.set_figtitle(figtitle)
        pt.adjust_subplots(left=0, right=1.0, top=1.0, bottom=0.0, wspace=.1, hspace=0)
        pt.show_figure(fig)
        if once_:
            pt.present()
            once_ = False

    #iter_ = list(zip(range(len(label_list)), label_list, warped_patch1_list, warped_patch2_list))
    #iter_ = list(ut.ichunks(iter_, chunck_size))
    #import vtool as vt
    ##for tup in ut.InteractiveIter(iter_, display_item=False):
    #for tup in ut.InteractiveIter(iter_, display_item=False):
    #    label_list = ut.get_list_column(tup, 1)
    #    patch1_list = ut.get_list_column(tup, 2)
    #    patch2_list = ut.get_list_column(tup, 3)
    #    patch1_list = [patch.copy() for patch in patch1_list]
    #    patch2_list = [patch.copy() for patch in patch2_list]
    #    # draw label border
    #    patch1_list = [vt.draw_border(patch, color=(green if label else red), thickness=3, out=patch)
    #                   for label, patch in zip(label_list, patch1_list)]
    #    patch2_list = [vt.draw_border(patch, color=(green if label else red), thickness=3, out=patch)
    #                   for label, patch in zip(label_list, patch2_list)]
    #    # draw black border
    #    patch1_list = [vt.draw_border(patch, color=(0, 0, 0), thickness=1, out=patch) for patch in patch1_list]
    #    patch2_list = [vt.draw_border(patch, color=(0, 0, 0), thickness=1, out=patch) for patch in patch2_list]
    #    # stack and show
    #    stack_kw = dict(modifysize=False, return_offset=True, return_sf=True)
    #    stacked_patch1s, offset_list1, sf_list1 = pt.stack_image_list(patch1_list, vert=True, **stack_kw)
    #    stacked_patch2s, offset_list2, sf_list2 = pt.stack_image_list(patch2_list, vert=True, **stack_kw)
    #    #stacked_patches = pt.stack_images(stacked_patch1s, stacked_patch2s, vert=False)[0]
    #    stacked_patches, offset_list, sf_list = pt.stack_multi_images(
    #        stacked_patch1s, stacked_patch2s, offset_list1, sf_list1, offset_list2, sf_list2, vert=False, modifysize=False)
    #    pt.figure(fnum=1, pnum=(1, 1, 1), doclf=True)
    #    pt.imshow(stacked_patches)
    #    pt.update()
    #    #pt.imshow(stacked_patch1s, pnum=(1, 2, 1))
    #    #pt.imshow(stacked_patch2s, pnum=(1, 2, 2))
    #    #count, (label, patch1, patch2) = tup
    #    #count, (label, patch1, patch2) = tup
    #    #if aid1_list_ is not None:
    #    #    aid1 = aid1_list_[count]
    #    #    aid2 = aid2_list_[count]
    #    #    print('aid1=%r, aid2=%r, label=%r' % (aid1, aid2, label))
    #    #pt.figure(fnum=1, pnum=(1, 2, 1), doclf=True)
    #    #pt.imshow(patch1, pnum=(1, 2, 1))
    #    #pt.draw_border(pt.gca(), color=vh.get_truth_color(label))
    #    #pt.imshow(patch2, pnum=(1, 2, 2))
    #    #pt.draw_border(pt.gca(), color=vh.get_truth_color(label))
    #    #pt.update()


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
