from __future__ import absolute_import, division, print_function
import utool as ut
(print, print_, printDBG, rrr, profile) = ut.inject(
    __name__, '[ibeis_cnn._plugin_grabmodels]', DEBUG=False)


#DEFAULT_CNNMODELS_DIR = ut.get_app_resource_dir('ibeis_cnn', 'pretrained')

MODEL_URLS = {
    'viewpoint': 'https://www.dropbox.com/s/c0vimzc9pubpjwn/viewpoint.zip',
    'caffenet_full': 'https://www.dropbox.com/s/r9oaif5os45cn2s/caffenet.caffe.pickle',
    'vggnet_full': 'https://www.dropbox.com/s/i7yb2ogmzr3w7v5/vgg.caffe.pickle',
}


def ensure_zipped_model(model, redownload=False):
    url = MODEL_URLS[model]
    extracted_fpath = ut.grab_zipped_url(url, appname='ibeis_cnn', redownload=redownload)
    return extracted_fpath


def ensure_unzipped_model(model, redownload=False):
    url = MODEL_URLS[model]
    extracted_fpath = ut.grab_file_url(url, appname='ibeis_cnn', redownload=redownload)
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
