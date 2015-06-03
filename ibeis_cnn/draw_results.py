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
    warped_patch1_list, warped_patch2_list = list(zip(*ut.ichunks(data, 2)))
    interact_patches(labels, warped_patch1_list, warped_patch2_list, flat_metadata, **kwargs)


def interact_patches(label_list, warped_patch1_list, warped_patch2_list, flat_metadata, rand=True, figtitle=None):
    #from ibeis.viz import viz_helpers as vh
    import plottool as pt
    import vtool as vt

    # Define order of iteration
    #index_list = list(range(len(label_list)))
    #index_list = ut.list_argsort(flat_metadata['fs'])[::1]

    # Check out the score pdfs
    if 'fs' in flat_metadata:
        index_list = ut.list_argsort(flat_metadata['fs'])[::-1]
        from vtool import score_normalization as scorenorm
        tp_support = np.array(ut.list_compress(flat_metadata['fs'], label_list)).astype(np.float64)
        tn_support = np.array(ut.list_compress(flat_metadata['fs'], ut.not_list(label_list))).astype(np.float64)
        scorenorm.test_score_normalization(tp_support, tn_support)
        #(score_domain, p_tp_given_score, p_tn_given_score, p_score_given_tp, p_score_given_tn,
        # p_score, clip_score) = scorenorm.learn_score_normalization(tp_support, tn_support, return_all=True)
        #scorenorm.inspect_pdfs(tn_support, tp_support, score_domain,
        #                       p_tp_given_score, p_tn_given_score, p_score_given_tp, p_score_given_tn, p_score, with_scores=True)
        pt.set_figtitle('HotSpotter Patch Scores')
    else:
        if rand:
            index_list = ut.random_indexes(len(label_list))
        else:
            index_list = list(range(len(label_list)))

    chunck_sizes = (6, 8)
    multi_chunked_indicies = list(ut.iter_multichunks(index_list, chunck_sizes))

    green = tuple(pt.color_funcs.to_base255(pt.TRUE_GREEN)[0:3])
    red   = tuple(pt.color_funcs.to_base255(pt.FALSE_RED)[0:3])[::-1]
    def get_patch_chunk(indicies):
        """
        indicies = chunked_indicies[0]
        """
        import utool as ut
        patch1_list_subset = [np.tile(patch[None, :].T, 3) if len(patch.shape) == 2 else patch.copy() for patch in ut.list_take(warped_patch1_list, indicies)]
        patch2_list_subset = [np.tile(patch[None, :].T, 3) if len(patch.shape) == 2 else patch.copy() for patch in ut.list_take(warped_patch2_list, indicies)]
        label_list_subset = ut.list_take(label_list, indicies)
        # Ipython embed hates dict comprehensions and locals
        flat_metadata_subset = dict([(key, ut.list_take(vals, indicies)) for key, vals in six.iteritems(flat_metadata)])
        # draw label border
        #colorfn = lambda label: (green if label else red)
        colorfn = [red, green]
        thickness = 2
        patch1_list_subset = [
            vt.draw_border(patch, color=colorfn[label], thickness=thickness, out=patch)
            for label, patch in zip(label_list_subset, patch1_list_subset)
        ]
        patch2_list_subset = [
            vt.draw_border(patch, color=colorfn[label], thickness=thickness, out=patch)
            for label, patch in zip(label_list_subset, patch2_list_subset)
        ]
        # draw black border
        patch1_list = [vt.draw_border(patch, color=(0, 0, 0), thickness=1, out=patch) for patch in patch1_list_subset]
        patch2_list = [vt.draw_border(patch, color=(0, 0, 0), thickness=1, out=patch) for patch in patch2_list_subset]
        # stack and show
        stack_kw = dict(modifysize=False, return_offset=True, return_sf=True)
        stacked_patch1s, offset_list1, sf_list1 = pt.stack_image_list(patch1_list, vert=True, **stack_kw)
        stacked_patch2s, offset_list2, sf_list2 = pt.stack_image_list(patch2_list, vert=True, **stack_kw)
        #stacked_patches = pt.stack_images(stacked_patch1s, stacked_patch2s, vert=False)[0]
        stacked_patches, offset_list, sf_list = pt.stack_multi_images(
            stacked_patch1s, stacked_patch2s, offset_list1, sf_list1, offset_list2, sf_list2,
            vert=False, modifysize=False)

        if 'fs' in flat_metadata_subset:
            scores = flat_metadata_subset['fs']
            score_texts = ['%.3f' % s for s in scores]
            left_offsets = offset_list[:len(offset_list) // 2]
            #right_offsets = offset_list[len(offset_list) // 2:]

            scale_up = 3

            img = vt.resize_image_by_scale(stacked_patches, scale_up)
            textcolor_rgb1 = pt.to_base255(pt.BLACK)[:3]
            #textcolor_rgb2 = pt.to_base255(pt.ORANGE)[:3]
            textcolor_rgb2 = pt.to_base255(pt.WHITE)[:3]
            for offset, text in zip(left_offsets, score_texts):
                #print(offset)
                #print(s)
                import cv2
                scaled_offset = tuple(((np.array(offset) + thickness + 2) * scale_up).astype(np.int).tolist())
                #fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL
                fontFace = cv2.FONT_HERSHEY_PLAIN
                fontkw = dict(bottomLeftOrigin=True, fontScale=2.5, fontFace=fontFace)
                # Bordered text
                vt.draw_text(img, text, scaled_offset, thickness=6, textcolor_rgb=textcolor_rgb1, **fontkw)
                vt.draw_text(img, text, scaled_offset, thickness=2, textcolor_rgb=textcolor_rgb2, **fontkw)
        else:
            img = stacked_patches
        return img, offset_list
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
            img, offset_list = get_patch_chunk(indicies)
            # TODO; use offset_list for interaction
            pt.imshow(img, pnum=next_pnum())
            ax = pt.gca()
            ax.invert_yaxis()
        #pt.update()
        if figtitle is not None:
            print(figtitle)
            pt.set_figtitle(figtitle)
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
