from __future__ import absolute_import, division, print_function
import numpy as np
import functools
from ibeis_cnn import draw_results
import utool as ut
print, rrr, profile = ut.inject2(__name__, '[ibeis_cnn.experiments]')


def test_siamese_performance(model, data, labels, dataname=''):
    """
    python -m ibeis_cnn.train --test-train_patchmatch_liberty --show --test
    python -m ibeis_cnn.train --test-train_patchmatch_pz --show --test
    """
    import vtool as vt
    from ibeis_cnn import harness

    # TODO: save in model.trainind_dpath/diagnostics/figures
    epoch_dpath = model.get_epoch_diagnostic_dpath()
    ut.vd(epoch_dpath)

    dataname += ' ' + model.get_model_history_hashid() + '\n'

    history_text = ut.list_str(model.era_history, newlines=True)

    ut.write_to(ut.unixjoin(epoch_dpath, 'era_history.txt'), history_text)

    if True:
        import matplotlib as mpl
        mpl.rcParams['agg.path.chunksize'] = 100000

    #data   = data[::50]
    #labels = labels[::50]
    #from ibeis_cnn import utils
    #data, labels = utils.random_xy_sample(data, labels, 10000, model.data_per_label_input)

    import plottool as pt
    fnum_gen = pt.make_fnum_nextgen()

    fig = model.show_era_history(fnum=fnum_gen(), yscale='linear')
    pt.save_figure(fig=fig, dpath=epoch_dpath, dpi=180)

    fig = model.show_era_history(fnum=fnum_gen(), yscale='log')
    pt.save_figure(fig=fig, dpath=epoch_dpath, dpi=180)

    # hack
    fig = model.show_weights_image(target=0, fnum=fnum_gen())
    pt.save_figure(fig=fig, dpath=epoch_dpath, dpi=180)
    #model.draw_all_conv_layer_weights(fnum=fnum_gen())
    #model.imwrite_weights(1)
    #model.imwrite_weights(2)
    return
    #ut.embed()

    #ut.embed()

    # Compute each type of score
    test_outputs = harness.test_data2(model, data, labels)
    network_output = test_outputs['network_output']
    # hack converting network output to distances for non-descriptor networks
    if len(network_output.shape) == 2 and network_output.shape[1] == 1:
        cnn_scores = network_output.T[0]
    elif len(network_output.shape) == 1:
        cnn_scores = network_output
    elif len(network_output.shape) == 2 and network_output.shape[1] > 1:
        assert model.data_per_label_output == 2
        vecs1 = network_output[0::2]
        vecs2 = network_output[1::2]
        cnn_scores = vt.L2(vecs1, vecs2)
    else:
        assert False
    sift_scores = test_sift_patchmatch_scores(data, labels)

    # Learn encoders
    encoder_kw = {
        #'monotonize': False,
        'monotonize': True,
    }
    cnn_encoder = vt.ScoreNormalizer(**encoder_kw)
    cnn_encoder.fit(cnn_scores, labels)

    sift_encoder = vt.ScoreNormalizer(**encoder_kw)
    sift_encoder.fit(sift_scores, labels)

    #ut.embed()

    # Visualize
    inter_cnn = cnn_encoder.visualize(figtitle=dataname + ' CNN scores. #data=' + str(len(data)), fnum=fnum_gen())
    inter_sift = sift_encoder.visualize(figtitle=dataname + ' SIFT scores. #data=' + str(len(data)), fnum=fnum_gen())

    # Save
    pt.save_figure(fig=inter_cnn.fig, dpath=epoch_dpath)
    pt.save_figure(fig=inter_sift.fig, dpath=epoch_dpath)

    # Save out examples of hard errors
    #cnn_fp_label_indicies, cnn_fn_label_indicies = cnn_encoder.get_error_indicies(cnn_scores, labels)
    #sift_fp_label_indicies, sift_fn_label_indicies = sift_encoder.get_error_indicies(sift_scores, labels)

    cnn_indicies = cnn_encoder.get_confusion_indicies(cnn_scores, labels)
    sift_indicies = sift_encoder.get_confusion_indicies(sift_scores, labels)

    warped_patch1_list, warped_patch2_list = list(zip(*ut.ichunks(data, 2)))
    _sample = functools.partial(draw_results.get_patch_sample_img, warped_patch1_list, warped_patch2_list, labels)

    cnn_fp_img = _sample({'fs': cnn_scores}, cnn_indicies.fp)[0]
    cnn_fn_img = _sample({'fs': cnn_scores}, cnn_indicies.fn)[0]
    cnn_tp_img = _sample({'fs': cnn_scores}, cnn_indicies.tp)[0]
    cnn_tn_img = _sample({'fs': cnn_scores}, cnn_indicies.tn)[0]

    sift_fp_img = _sample({'fs': sift_scores}, sift_indicies.fp)[0]
    sift_fn_img = _sample({'fs': sift_scores}, sift_indicies.fn)[0]
    sift_tp_img = _sample({'fs': sift_scores}, sift_indicies.tp)[0]
    sift_tn_img = _sample({'fs': sift_scores}, sift_indicies.tn)[0]

    #if ut.show_was_requested():
    #def rectify(arr):
    #    return np.flipud(arr)
    SINGLE_FIG = False
    if SINGLE_FIG:
        def dump_img(img_, lbl, fnum):
            fig, ax = pt.imshow(img_, figtitle=dataname + ' ' + lbl, fnum=fnum)
            pt.save_figure(fig=fig, dpath=epoch_dpath, dpi=180)
        dump_img(cnn_fp_img, 'cnn_fp_img', fnum_gen())
        dump_img(cnn_fn_img, 'cnn_fn_img', fnum_gen())
        dump_img(cnn_tp_img, 'cnn_tp_img', fnum_gen())
        dump_img(cnn_tn_img, 'cnn_tn_img', fnum_gen())

        dump_img(sift_fp_img, 'sift_fp_img', fnum_gen())
        dump_img(sift_fn_img, 'sift_fn_img', fnum_gen())
        dump_img(sift_tp_img, 'sift_tp_img', fnum_gen())
        dump_img(sift_tn_img, 'sift_tn_img', fnum_gen())
        #vt.imwrite(dataname + '_' + 'cnn_fp_img.png', (cnn_fp_img))
        #vt.imwrite(dataname + '_' + 'cnn_fn_img.png', (cnn_fn_img))
        #vt.imwrite(dataname + '_' + 'sift_fp_img.png', (sift_fp_img))
        #vt.imwrite(dataname + '_' + 'sift_fn_img.png', (sift_fn_img))
    else:
        fnum = fnum_gen()
        pnum_gen = pt.make_pnum_nextgen(4, 2)
        fig = pt.figure(fnum)
        pt.imshow(cnn_fp_img,  title='CNN FP',  fnum=fnum, pnum=pnum_gen())
        pt.imshow(sift_fp_img, title='SIFT FP', fnum=fnum, pnum=pnum_gen())
        pt.imshow(cnn_fn_img,  title='CNN FN',  fnum=fnum, pnum=pnum_gen())
        pt.imshow(sift_fn_img, title='SIFT FN', fnum=fnum, pnum=pnum_gen())
        pt.imshow(cnn_tp_img,  title='CNN TP',  fnum=fnum, pnum=pnum_gen())
        pt.imshow(sift_tp_img, title='SIFT TP', fnum=fnum, pnum=pnum_gen())
        pt.imshow(cnn_tn_img,  title='CNN TN',  fnum=fnum, pnum=pnum_gen())
        pt.imshow(sift_tn_img, title='SIFT TN', fnum=fnum, pnum=pnum_gen())
        pt.set_figtitle(dataname + ' confusions')
        pt.adjust_subplots(left=0, right=1.0, bottom=0., wspace=.01, hspace=.05)
        pt.save_figure(fig=fig, dpath=epoch_dpath, dpi=180, figsize=(9, 18))

    #ut.vd(epoch_dpath)


def show_hard_cases(model, data, labels, scores):
    from ibeis_cnn import utils
    encoder = model.learn_encoder(labels, scores)
    encoder.visualize()

    #x = encoder.inverse_normalize(np.cast['float32'](encoder.learned_thresh))
    #encoder.normalize_scores(x)
    #encoder.inverse_normalize(np.cast['float32'](encoder.learned_thresh))

    fp_label_indicies, fn_label_indicies = encoder.get_error_indicies(scores, labels)
    fn_data_indicies = utils.expand_data_indicies(fn_label_indicies, model.data_per_label_input)
    fp_data_indicies = utils.expand_data_indicies(fp_label_indicies, model.data_per_label_input)

    fn_data   = data.take(fn_data_indicies, axis=0)
    fn_labels = labels.take(fn_label_indicies, axis=0)
    fn_scores = scores.take(fn_label_indicies, axis=0)

    fp_data   = data.take(fp_data_indicies, axis=0)
    fp_labels = labels.take(fp_label_indicies, axis=0)
    fp_scores = scores.take(fp_label_indicies, axis=0)

    from ibeis_cnn import draw_results
    draw_results.rrr()
    draw_results.interact_siamsese_data_patches(fn_labels, fn_data, {'fs': fn_scores}, figtitle='FN')
    draw_results.interact_siamsese_data_patches(fp_labels, fp_data, {'fs': fp_scores}, figtitle='FP')


def test_sift_patchmatch_scores(data, labels):
    """
    data = X_test
    labels = y_test
    """
    import pyhesaff
    import numpy as np
    if len(data.shape) == 4 and data.shape[-1] == 1:
        data = data.reshape(data.shape[0:3])
    vecs_list = pyhesaff.extract_desc_from_patches(data)
    sqrddist = ((vecs_list[0::2].astype(np.float32) - vecs_list[1::2].astype(np.float32)) ** 2).sum(axis=1)
    sqrddist_ = sqrddist[None, :].T
    VEC_PSEUDO_MAX_DISTANCE_SQRD = 2.0 * (512.0 ** 2.0)
    sift_scores = 1 - (sqrddist_.flatten() / VEC_PSEUDO_MAX_DISTANCE_SQRD)
    return sift_scores
    #test_siamese_thresholds(sqrddist_, labels, figtitle='SIFT descriptor distances')


def test_siamese_thresholds(network_output, y_test, **kwargs):
    """
    Test function to see how good of a threshold we can learn

    network_output = prob_list
    """
    import vtool as vt
    # batch cycling may cause more outputs than test labels.
    # should be able to just crop
    network_output_ = network_output[0:len(y_test)].copy() ** 2
    tp_support = network_output_.T[0][y_test.astype(np.bool)].astype(np.float64)
    tn_support = network_output_.T[0][~(y_test.astype(np.bool))].astype(np.float64)
    if tp_support.mean() < tn_support.mean():
        print('need to invert scores')
        tp_support *= -1
        tn_support *= -1
    bottom = min(tn_support.min(), tp_support.min())
    if bottom < 0:
        print('need to subtract from scores')
        tn_support -= bottom
        tp_support -= bottom

    vt.score_normalization.rrr()
    vt.score_normalization.test_score_normalization(tp_support, tn_support,
                                                    with_scores=False,
                                                    **kwargs)

    #from ibeis.model.hots import score_normalization
    #test_score_normalization
    #learnkw = dict()
    #learntup = score_normalization.learn_score_normalization(
    #    tp_support, tn_support, return_all=False, **learnkw)
    #(score_domain, p_tp_given_score, clip_score) = learntup
    # Plotting
    #import plottool as pt
    #fnum = 1
    #pt.figure(fnum=fnum, pnum=(2, 1, 1), doclf=True, docla=True)
    #score_normalization.plot_support(tn_support, tp_support, fnum=fnum, pnum=(2, 1, 1))
    #score_normalization.plot_postbayes_pdf(
    #    score_domain, 1 - p_tp_given_score, p_tp_given_score, fnum=fnum, pnum=(2, 1, 2))
    #pass
