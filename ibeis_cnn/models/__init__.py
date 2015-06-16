### __init__.py ###
# flake8: noqa
from __future__ import absolute_import, division, print_function
from ibeis_cnn.models import abstract_models
from ibeis_cnn.models import dummy
from ibeis_cnn.models import mnist
from ibeis_cnn.models import siam
import utool
print, print_, printDBG, rrr, profile = utool.inject(
    __name__, '[ibeis_cnn.models]')


def reassign_submodule_attributes(verbose=True):
    """
    why reloading all the modules doesnt do this I don't know
    """
    import sys
    if verbose and '--quiet' not in sys.argv:
        print('dev reimport')
    # Self import
    import ibeis_cnn.models
    # Implicit reassignment.
    seen_ = set([])
    for tup in IMPORT_TUPLES:
        if len(tup) > 2 and tup[2]:
            continue  # dont import package names
        submodname, fromimports = tup[0:2]
        submod = getattr(ibeis_cnn.models, submodname)
        for attr in dir(submod):
            if attr.startswith('_'):
                continue
            if attr in seen_:
                # This just holds off bad behavior
                # but it does mimic normal util_import behavior
                # which is good
                continue
            seen_.add(attr)
            setattr(ibeis_cnn.models, attr, getattr(submod, attr))


def reload_subs(verbose=True):
    """ Reloads ibeis_cnn.models and submodules """
    rrr(verbose=verbose)
    def fbrrr(*args, **kwargs):
        """ fallback reload """
        pass
    getattr(abstract_models, 'rrr', fbrrr)(verbose=verbose)
    getattr(dummy, 'rrr', fbrrr)(verbose=verbose)
    getattr(mnist, 'rrr', fbrrr)(verbose=verbose)
    getattr(siam, 'rrr', fbrrr)(verbose=verbose)
    rrr(verbose=verbose)
    try:
        # hackish way of propogating up the new reloaded submodule attributes
        reassign_submodule_attributes(verbose=verbose)
    except Exception as ex:
        print(ex)
rrrr = reload_subs

IMPORT_TUPLES = [
    ('abstract_models', None),
    ('dummy', None),
    ('mnist', None),
    ('siam', None),
]
"""
python -c "import ibeis_cnn.models" --dump-ibeis_cnn.models-init
python -c "import ibeis_cnn.models" --update-ibeis_cnn.models-init
"""
__DYNAMIC__ = True
if __DYNAMIC__:
    # TODO: import all utool external prereqs. Then the imports will not import
    # anything that has already in a toplevel namespace
    # COMMENTED OUT FOR FROZEN __INIT__
    # Dynamically import listed util libraries and their members.
    from utool._internal import util_importer
    # FIXME: this might actually work with rrrr, but things arent being
    # reimported because they are already in the modules list
    import_execstr = util_importer.dynamic_import(__name__, IMPORT_TUPLES)
    exec(import_execstr)
    DOELSE = False
else:
    # Do the nonexec import (can force it to happen no matter what if alwyas set
    # to True)
    DOELSE = True

if DOELSE:
    # <AUTOGEN_INIT>

    from ibeis_cnn.models import abstract_models
    from ibeis_cnn.models import dummy
    from ibeis_cnn.models import mnist
    from ibeis_cnn.models import siam
    from ibeis_cnn.models.abstract_models import (AbstractCategoricalModel, 
                                                  AbstractSiameseModel, 
                                                  BaseModel, Conv2DLayer, 
                                                  MaxPool2DLayer, 
                                                  PretrainedNetwork, 
                                                  evaluate_layer_list, 
                                                  testdata_model_with_history,) 
    from ibeis_cnn.models.dummy import (DummyModel,) 
    from ibeis_cnn.models.mnist import (MNISTModel,) 
    from ibeis_cnn.models.siam import (SiameseCenterSurroundModel, SiameseL2,) 
    import utool
    print, print_, printDBG, rrr, profile = utool.inject(
        __name__, '[ibeis_cnn.models]')
    
    
    def reassign_submodule_attributes(verbose=True):
        """
        why reloading all the modules doesnt do this I don't know
        """
        import sys
        if verbose and '--quiet' not in sys.argv:
            print('dev reimport')
        # Self import
        import ibeis_cnn.models
        # Implicit reassignment.
        seen_ = set([])
        for tup in IMPORT_TUPLES:
            if len(tup) > 2 and tup[2]:
                continue  # dont import package names
            submodname, fromimports = tup[0:2]
            submod = getattr(ibeis_cnn.models, submodname)
            for attr in dir(submod):
                if attr.startswith('_'):
                    continue
                if attr in seen_:
                    # This just holds off bad behavior
                    # but it does mimic normal util_import behavior
                    # which is good
                    continue
                seen_.add(attr)
                setattr(ibeis_cnn.models, attr, getattr(submod, attr))
    
    
    def reload_subs(verbose=True):
        """ Reloads ibeis_cnn.models and submodules """
        rrr(verbose=verbose)
        def fbrrr(*args, **kwargs):
            """ fallback reload """
            pass
        getattr(abstract_models, 'rrr', fbrrr)(verbose=verbose)
        getattr(dummy, 'rrr', fbrrr)(verbose=verbose)
        getattr(mnist, 'rrr', fbrrr)(verbose=verbose)
        getattr(siam, 'rrr', fbrrr)(verbose=verbose)
        rrr(verbose=verbose)
        try:
            # hackish way of propogating up the new reloaded submodule attributes
            reassign_submodule_attributes(verbose=verbose)
        except Exception as ex:
            print(ex)
    rrrr = reload_subs
    # </AUTOGEN_INIT>
"""
Regen Command:
    cd /home/joncrall/code/ibeis_cnn/ibeis_cnn/models
    makeinit.py --star
"""