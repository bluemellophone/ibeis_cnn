from __future__ import absolute_import, division, print_function
import utool as ut
(print, print_, printDBG, rrr, profile) = ut.inject(
    __name__, '[ibeis_cnn._plugin_grabmodels]', DEBUG=False)


#DEFAULT_CNNMODELS_DIR = ut.get_app_resource_dir('ibeis_cnn', 'pretrained')

MODEL_URLS = {
    'viewpoint': 'https://www.dropbox.com/s/6xjtcz8qrdj2cof/viewpoint.pickle?dl=0',
}


def ensure_model(model, redownload=False):
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
