#!/usr/bin/env python2.7
from __future__ import absolute_import, division, print_function
import six
from utool import util_setup
from setuptools import setup


CHMOD_PATTERNS = [
    'run_tests.sh',
]

PROJECT_DIRS = ['ibeis_cnn', ]

CLUTTER_PATTERNS = [
    '\'',
    '*.dump.txt',
    '*.prof',
    '*.prof.txt',
    '*.lprof',
    '*.ln.pkg',
    'test_times.txt',
]

CLUTTER_DIRS = [
    'logs/',
    'dist/',
    'testsuite',
    '__pycache__/',
]

"""
Need special theano
References:
    http://lasagne.readthedocs.org/en/latest/user/installation.html
    pip install -r https://raw.githubusercontent.com/Lasagne/Lasagne/v0.1/requirements.txt
"""

INSTALL_REQUIRES = [
    'scikit-learn >= 0.16.1',
    'theano',
    'lasagne',
    # 'h5py',  # Install this instead 'sudo apt-get install libhdf5-dev' due to Numpy versioning issues
    #'pylearn2',
    #'git+git://github.com/lisa-lab/pylearn2.git'

    #'utool >= 1.0.0.dev1',
    #'vtool >= 1.0.0.dev1',
    ##'pyhesaff >= 1.0.0.dev1',
    #'pyrf >= 1.0.0.dev1',
    #'guitool >= 1.0.0.dev1',
    #'plottool >= 1.0.0.dev1',
    #'matplotlib >= 1.3.1',
    #'scipy >= 0.13.2',
    #'numpy >= 1.8.0',
    #'Pillow >= 2.4.0',
    #'psutil',
    #'requests >= 0.8.2',
    #'setproctitle >= 1.1.8',
    ##'decorator',
    #'lockfile >= 0.10.2',
    #'apipkg',
    #'objgraph',
    #'pycallgraph',
    #'gevent',
    #'PyQt 4/5 >= 4.9.1', # cannot include because pyqt4 is not in pip
]

#INSTALL_OPTIONAL = [
#    'tornado',
#    'flask',
#    'autopep8',
#    'pyfiglet',
#    'theano',
#    'pylearn2'
#    'lasenge'
#]

if six.PY2:
    INSTALL_REQUIRES.append('requests >= 0.8.2')


if __name__ == '__main__':
    print('[setup] Entering IBEIS setup')
    kwargs = util_setup.setuptools_setup(
        setup_fpath=__file__,
        name='ibeis_cnn',
        #author='Hendrik Weideman, Jason Parham, and Jon Crall',
        #author_email='erotemic@gmail.com',
        packages=util_setup.find_packages(),
        version=util_setup.parse_package_for_version('ibeis_cnn'),
        license=util_setup.read_license('LICENSE'),
        long_description=util_setup.parse_readme('README.md'),
        ext_modules=util_setup.find_ext_modules(),
        cmdclass=util_setup.get_cmdclass(),
        project_dirs=PROJECT_DIRS,
        chmod_patterns=CHMOD_PATTERNS,
        clutter_patterns=CLUTTER_PATTERNS,
        clutter_dirs=CLUTTER_DIRS,
        install_requires=INSTALL_REQUIRES
        #cython_files=CYTHON_FILES,
    )
    import utool as ut
    print('kwargs = %s' % (ut.dict_str(kwargs),))
    setup(**kwargs)
