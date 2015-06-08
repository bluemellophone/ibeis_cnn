from __future__ import absolute_import, division, print_function
import numpy as np
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

    dataname += ' ' + model.get_model_history_hashid() + '\n'

    history_text = ut.list_str(model.era_history, newlines=True)

    ut.write_to(ut.unixjoin(epoch_dpath, 'era_history.txt'), history_text)

    # Compute each type of score
    test_outputs = harness.test_data2(model, data, labels)
    cnn_scores = test_outputs['network_output'].T[0]
    sift_scores = test_sift_patchmatch_scores(data, labels)

    # Learn encoders
    encoder_kw = {
        'monotonize': False,
    }
    cnn_encoder = vt.ScoreNormalizer(**encoder_kw)
    cnn_encoder.fit(cnn_scores, labels)

    sift_encoder = vt.ScoreNormalizer(**encoder_kw)
    sift_encoder.fit(sift_scores, labels)

    # Visualize
    inter_cnn = cnn_encoder.visualize(figtitle=dataname + ' CNN scores. #data=' + str(len(data)), fnum=1)
    inter_sift = sift_encoder.visualize(figtitle=dataname + ' SIFT scores. #data=' + str(len(data)), fnum=2)

    # Save
    import plottool as pt
    pt.save_figure(fig=inter_cnn.fig, dpath=epoch_dpath)
    pt.save_figure(fig=inter_sift.fig, dpath=epoch_dpath)

    # Save out examples of hard errors
    cnn_fp_label_indicies, cnn_fn_label_indicies = cnn_encoder.get_error_indicies(cnn_scores, labels)
    sift_fp_label_indicies, sift_fn_label_indicies = sift_encoder.get_error_indicies(sift_scores, labels)

    warped_patch1_list, warped_patch2_list = list(zip(*ut.ichunks(data, 2)))
    cnn_fp_img = draw_results.get_patch_sample_img(warped_patch1_list,
                                                   warped_patch2_list, labels,
                                                   {'fs': cnn_scores},
                                                   cnn_fp_label_indicies)[0]
    cnn_fn_img = draw_results.get_patch_sample_img(warped_patch1_list,
                                                   warped_patch2_list, labels,
                                                   {'fs': cnn_scores},
                                                   cnn_fn_label_indicies)[0]
    sift_fp_img = draw_results.get_patch_sample_img(warped_patch1_list,
                                                    warped_patch2_list, labels,
                                                    {'fs': sift_scores},
                                                    sift_fp_label_indicies)[0]
    sift_fn_img = draw_results.get_patch_sample_img(warped_patch1_list,
                                                    warped_patch2_list, labels,
                                                    {'fs': sift_scores},
                                                    sift_fn_label_indicies)[0]

    #if ut.show_was_requested():
    #def rectify(arr):
    #    return np.flipud(arr)
    # TODO: higher dpi and figsize
    fig, ax = pt.imshow(cnn_fp_img, figtitle=dataname + ' ' + 'cnn_fp_img', fnum=3)
    pt.save_figure(fig=fig, dpath=epoch_dpath)
    fig, ax = pt.imshow(cnn_fn_img, figtitle=dataname + ' ' + 'cnn_fn_img', fnum=4)
    pt.save_figure(fig=fig, dpath=epoch_dpath)
    fig, ax = pt.imshow(sift_fp_img, figtitle=dataname + ' ' + 'sift_fp_img', fnum=5)
    pt.save_figure(fig=fig, dpath=epoch_dpath)
    fig, ax = pt.imshow(sift_fn_img, figtitle=dataname + ' ' + 'sift_fn_img', fnum=6)
    pt.save_figure(fig=fig, dpath=epoch_dpath)
    #vt.imwrite(dataname + '_' + 'cnn_fp_img.png', (cnn_fp_img))
    #vt.imwrite(dataname + '_' + 'cnn_fn_img.png', (cnn_fn_img))
    #vt.imwrite(dataname + '_' + 'sift_fp_img.png', (sift_fp_img))
    #vt.imwrite(dataname + '_' + 'sift_fn_img.png', (sift_fn_img))

    # hack
    model.draw_all_conv_layer_weights(fnum=7)
    #model.save_model_layer_weights(1)
    #model.save_model_layer_weights(2)


def show_hard_cases(model, data, labels, scores):
    from ibeis_cnn import utils
    encoder = model.learn_encoder(labels, scores)
    encoder.visualize()

    #x = encoder.inverse_normalize(np.cast['float32'](encoder.learned_thresh))
    #encoder.normalize_scores(x)
    #encoder.inverse_normalize(np.cast['float32'](encoder.learned_thresh))

    fp_label_indicies, fn_label_indicies = encoder.get_error_indicies(scores, labels)
    fn_data_indicies = utils.expand_data_indicies(fn_label_indicies, model.data_per_label)
    fp_data_indicies = utils.expand_data_indicies(fp_label_indicies, model.data_per_label)

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
    sqrddist = ((vecs_list[::2].astype(np.float32) - vecs_list[1::2].astype(np.float32)) ** 2).sum(axis=1)
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
