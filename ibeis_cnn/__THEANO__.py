# flake8: noqa
import sys
import os
import utool as ut
ut.noinject(__name__, '[theano]]')

DEVICE = ut.get_argval('--device', type_=str, default=None)


def parse_theano_flags():
    """
    export THEANO_FLAGS="device=cpu,print_active_device=True,enable_initial_driver_test=True"
    export THEANO_FLAGS="device=gpu2,print_active_device=True,enable_initial_driver_test=False"
    set THEANO_FLAGS="device=cpu,print_active_device=True,enable_initial_driver_test=True"
    """
    theano_flags_str = os.environ.get('THEANO_FLAGS', '')
    theano_flags_itemstrs = theano_flags_str.split(',')
    theano_flags = ut.odict([itemstr.split('=') for itemstr in theano_flags_itemstrs if len(itemstr) > 0])
    return theano_flags

def write_theano_flags(theano_flags):
    #print('theano_flags = %r' % (theano_flags,))
    theano_flags_itemstrs = [key + '=' + str(val) for key, val in theano_flags.items()]
    theano_flags_str = ','.join(theano_flags_itemstrs)
    os.environ['THEANO_FLAGS'] = theano_flags_str

if DEVICE is not None:
    # http://deeplearning.net/software/theano/library/config.html
    print('Change device to %r' % (DEVICE,))
    theano_flags = parse_theano_flags()
    theano_flags['cnmem'] = False
    theano_flags['device'] = DEVICE
    #theano_flags['print_active_device'] = False
    write_theano_flags(theano_flags)
    #python -c 'import theano; print theano.config'

# assert 'theano' not in sys.modules, 'Theano should not be imported yet'
if 'theano' in sys.modules:
    print('IBEIS_CNN cannot apply settings to theano because it was already imported')

from theano import *  # NOQA
#from theano import tensor
