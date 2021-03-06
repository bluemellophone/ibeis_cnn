"""
References:
    http://deeplearning.net/software/theano/library/config.html

Check Settings:
    python -c 'import theano; print theano.config' | less
"""
import utool as ut
import os
from os.path import join

"""
CommandLine:
    cd %CODE_DIR%/ibies_cnn/code
    cd $CODE_DIR/ibies_cnn/code
    code
    cd ibeis_cnn/code
    python train.py

PythonPrereqs:
    pip uninstall theano
    pip uninstall lasagne
    pip uninstall pylearn2

    co
    git clone https://github.com/Theano/Theano.git
    git clone git://github.com/lisa-lab/pylearn2.git
    git clone https://github.com/Erotemic/Lasagne.git
    cd ~/code/Theano
    python setup.py develop
    cd ~/code/pylearn2
    python setup.py develop
    cd ~/code/Lasagne/
    python setup.py develop

    python -c "import pylearn2; print(pylearn2.__file__)"
    python -c "import theano; print(theano.__version__)"
    python -c "import lasagne; print(lasagne.__version__)"


git checkout 8758ac1434175159e5c1f30123041799c2b6098a
OLD:
    git clone https://github.com/Lasagne/Lasagne.git
    pip install theano
    pip install git+https://github.com/Lasagne/Lasagne.git
    pip install git+git://github.com/lisa-lab/pylearn2.git
    #pip install lasagne
    #pip install pylearn2
"""


def init_theanorc():
    theanorc_fpath = join(os.getenv('HOME'), '.theanorc')
    theanorc_text = ut.codeblock(
        '''
        [global]
        floatX = float32
        device = gpu0
        openmp = True

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
