"""
References:
    http://deeplearning.net/software/theano/library/config.html

Check Settings:
    python -c 'import theano; print theano.config' | less
"""
import utool as ut
import os
from os.path import join


def init_theanorc():
    theanorc_fpath = join(os.getenv('HOME'), '.theanorc')
    theanorc_text = ut.codeblock(
        '''
        [global]
        floatX = float32
        device = gpu0

        [nvcc]
        fastmath = True
        '''
    )
    if ut.checkpath(theanorc_fpath, verbose=True):
        if not ut.arg_you_sure('overwrite?'):
            return
    ut.write_to(theanorc_fpath, theanorc_text)


if __name__ == '__main__':
    init_theanorc()
