# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import utool as ut
(print, rrr, profile) = ut.inject2(
    __name__, '[ibeis_cnn._plugin_grabmodels]')


#DEFAULT_CNNMODELS_DIR = ut.get_app_resource_dir('ibeis_cnn', 'pretrained')

MODEL_DOMAIN = 'https://lev.cs.rpi.edu/public/models/'
MODEL_URLS = {
    'background_giraffe_masai':       'giraffe_masai_background.npy',
    'background_zebra_plains':        'zebra_plains_background.npy',
    'background_zebra_plains_grevys': 'zebra_plains_grevys_background.npy',
    'background_whale_fluke':         'whale_fluke_background.npy',
    'detect_yolo':                    'detect.yolo.pickle',
    'viewpoint':                      'viewpoint.pickle',
    'caffenet':                       'caffenet.caffe.slice_0_6_None.pickle',
    'caffenet_conv':                  'caffenet.caffe.slice_0_10_None.pickle',
    'caffenet_full':                  'caffenet.caffe.pickle',
    'vggnet':                         'vgg.caffe.slice_0_6_None.pickle',
    'vggnet_conv':                    'vgg.caffe.slice_0_32_None.pickle',
    'vggnet_full':                    'vgg.caffe.pickle',
}


def ensure_model(model, redownload=False):
    try:
        url = MODEL_DOMAIN + MODEL_URLS[model]
        extracted_fpath = ut.grab_file_url(url, appname='ibeis_cnn', redownload=redownload)
    except KeyError as ex:
        ut.printex(ex, 'model is not uploaded', iswarning=True)
        extracted_fpath = ut.unixjoin(ut.get_app_resource_dir('ibeis_cnn'), model)
        ut.assert_exists(extracted_fpath)
    return extracted_fpath


if __name__ == '__main__':
    """

    CommandLine:
        python -m ibeis_cnn._plugin_grabmodels.ensure_models
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
