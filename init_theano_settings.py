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

Purge from system and environ:
    cd

    python -c "import utool as ut; ut.total_purge_developed_repo('~/code/pylearn2')"
    python -c "import utool as ut; ut.total_purge_developed_repo('~/code/Theano')"
    python -c "import utool as ut; ut.total_purge_developed_repo('~/code/Lasange')"

    # Remove pylearn2 scripts
    sudo rm /home/joncrall/venv/bin/pylearn2-*
    sudo rm /usr/local/bin/pylearn2-*
    locate pylearn2 | grep -v /home/joncrall/code/pylearn2 | grep -v /home/jason/code/pylearn2

    pip uninstall theano
    pip uninstall lasagne
    pip uninstall pylearn2

    sudo -H pip uninstall theano
    sudo -H pip uninstall lasagne
    sudo -H pip uninstall pylearn2

    sudo pip uninstall theano
    sudo pip uninstall lasagne
    sudo pip uninstall pylearn2

    # If they do try chowning to current user
    sudo chown -R $USER:$USER ~/code/pylearn2
    sudo chown -R $USER:$USER ~/code/Theano
    sudo chown -R $USER:$USER ~/code/Lasagne

    export GLOBAL_SITE_PKGS=$(python -c "import utool as ut; print(ut.get_global_dist_packages_dir())")
    export LOCAL_SITE_PKGS=$(python -c "import utool as ut; print(ut.get_local_dist_packages_dir())")
    export VENV_SITE_PKGS=$(python -c "import utool as ut; print(ut.get_site_packages_dir())")

    # Test that they dont exist
    python -c "import pylearn2; print(pylearn2.__file__)"
    python -c "import theano; print(theano.__version__)"
    python -c "import lasagne; print(lasagne.__version__)"


PythonPrereqs:
    co
    git clone git://github.com/lisa-lab/pylearn2.git
    git clone https://github.com/Theano/Theano.git
    git clone https://github.com/Erotemic/Lasagne.git
    cd ~/code/pylearn2 && git pull && python setup.py develop
    cd ~/code/Theano   && git pull && python setup.py develop
    cd ~/code/Lasagne  && git pull && python setup.py develop

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


Ensure CuDNN is installed
    http://lasagne.readthedocs.io/en/latest/user/installation.html#cudnn

    # Test if Theano Works with CUDNN
    python -c "from theano.sandbox.cuda.dnn import dnn_available as d; print(d() or d.msg)"

    # Need to register with nvidia
    https://developer.nvidia.com/rdp/cudnn-download

    # Check cuda version
    nvcc --version

    # Check if cuda is globally installed
    ls -al /usr/local/cuda

    # Check if CUDNN is globally installed
    ls -al /usr/local/cuda/include/cudnn.h
    ls -al /usr/local/cuda/lib64/cudnn*

    # Download approprate version
    cd ~/Downloads
    # doesnt work if you dont sign in
    # wget https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v5.1/rc/7.5/cudnn-7.5-linux-x64-v5.1-rc-tgz

    # Unpack appropriate version
    cd ~/Downloads
    7z x cudnn-7.5-linux-x64-v5.1-rc.tgz && 7z x -ocudnn5.1 cudnn-7.5-linux-x64-v5.1-rc.tar
    7z x cudnn-7.5-linux-x64-v5.0-ga.tgz && 7z x -ocudnn5.0 cudnn-7.5-linux-x64-v5.0-ga.tar
    7z x cudnn-7.0-linux-x64-v4.0-prod.tgz && 7z x -ocudnn4.0  cudnn-7.0-linux-x64-v4.0-prod.tar
    tree ~/Downloads/cudnn5.1/
    tree ~/Downloads/cudnn4/

    # DEFINE WHERE CUDA LIVES
    export CUDADIR=/usr/local/cuda
    export TARGET_CUDNN_VERSION=5.1
    MAIN_CUDNN_VERSION="$(echo $TARGET_CUDNN_VERSION | head -c 1)"

    # Check CUDNN Install
    ls -al $CUDADIR/include/cudnn.h
    ls -al $CUDADIR/lib64/libcudnn*

    #Look at other cuda install permissions
    ls -al $CUDADIR/include/cublas.h
    ls -al $CUDADIR/lib64/libcublas*

    # REMOVE / UNINSTALL OLD CUDNN
    sudo rm -rf $CUDADIR/include/cudnn.h
    sudo rm -rf $CUDADIR/lib64/libcudnn*

    # Extract into folder called cuda, need to move it to wherever cuda is installed
    # cudnn consists of one header and 4 libraries
    sudo cp -rv ~/Downloads/cudnn$TARGET_CUDNN_VERSION/cuda/include/cudnn.h $CUDADIR/include/cudnn.h
    sudo cp -rv ~/Downloads/cudnn$TARGET_CUDNN_VERSION/cuda/lib64/libcudnn.so.$TARGET_CUDNN_VERSION* $CUDADIR/lib64/
    sudo cp -rv ~/Downloads/cudnn$TARGET_CUDNN_VERSION/cuda/lib64/libcudnn_static.a $CUDADIR/lib64/

    # Manually make symlinks (ones nvidia ships are broken)
    sudo ln -s $CUDADIR/lib64/libcudnn.so.$TARGET_CUDNN_VERSION* $CUDADIR/lib64/libcudnn.so.$MAIN_CUDNN_VERSION
    sudo ln -s $CUDADIR/lib64/libcudnn.so.$MAIN_CUDNN_VERSION $CUDADIR/lib64/libcudnn.so

    # Set permissions to reflect cuda install
    sudo chmod 755 /usr/local/cuda/lib64/libcudnn.so.$TARGET_CUDNN_VERSION*

    # Check CUDNN Install
    ls -al $CUDADIR/include/cudnn.h
    ls -al $CUDADIR/lib64/libcudnn*

    # Test if Theano Works with CUDNN
    python -c "from theano.sandbox.cuda.dnn import dnn_available as d; print(d() or d.msg)"
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
