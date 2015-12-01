#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals


def ibeis_cnn_main():
    ignore_prefix = []
    ignore_suffix = []
    import utool as ut
    try:
        print('[ibeis_cnn] Importing ibeis_cnn')
        import ibeis_cnn  # NOQA
        print('[ibeis_cnn] Importing ibeis')
        import ibeis  # NOQA
        print('[ibeis_cnn] Importing ibeis_cnn._plugin')
        import ibeis_cnn._plugin  # NOQA
        print('[ibeis_cnn] Done importing for __main__')
    except ImportError:
        raise
        pass
    # allow for --tf flags
    ut.main_function_tester('ibeis_cnn', ignore_prefix, ignore_suffix)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    print('Checking ibeis_cnn main')
    ibeis_cnn_main()
