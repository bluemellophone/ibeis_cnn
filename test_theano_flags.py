import utool as ut


def parse_theano_flags():
    import os
    theano_flags_str = os.environ.get('THEANO_FLAGS', '')
    theano_flags_itemstrs = theano_flags_str.split(',')
    theano_flags = ut.odict(
        [itemstr.split('=')
         for itemstr in theano_flags_itemstrs
         if len(itemstr) > 0])
    return theano_flags


def write_theano_flags(theano_flags):
    import os
    theano_flags_itemstrs = [
        key + '=' + str(val)
        for key, val in theano_flags.items()]
    theano_flags_str = ','.join(theano_flags_itemstrs)
    os.environ['THEANO_FLAGS'] = theano_flags_str

theano_flags = parse_theano_flags()
theano_flags['cnmem'] = 0
#theano_flags['device'] = DEVICE
theano_flags['print_active_device'] = False
theano_flags['enable_initial_driver_test'] = False
write_theano_flags(theano_flags)

with ut.Timer():
    import theano

print(ut.get_memstats_str())
