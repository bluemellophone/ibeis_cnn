# flake8: noqa
from ibeis_cnn import __THEANO__ as theano
try:
    from lasagne import *  # NOQA
except ImportError as ex:
    print('Lasagne failed to import')
    print('theano.__version__ = %r' % (theano.__version__,))
    print('theano.__file__ = %r' % (theano.__file__,))
    raise
from lasagne import layers
from lasagne import init
from lasagne import nonlinearities
import utool as ut
ut.noinject(__name__, '[lasagne]]')
