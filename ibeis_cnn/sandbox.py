# flake8: noqa
import cPickle as pickle
from ibeis_cnn import abstract_models
from ibeis_cnn.abstract_models import *


def check_external_training_paths():
    checkpoints_dir = '/home/joncrall/.config/ibeis_cnn/training_junction/train_patchmetric((315)iofvvdflcllgjkyu)/checkpoints'
    #checkpoints_dir = '/media/raid/work/PZ_MTEST/_ibsdb/_ibeis_cache/nets/train_patchmetric((315)iofvvdflcllgjkyu)/checkpoints/hist_eras2_epochs23_aayzxkezpzjgwupd'
    from os.path import *
    model_fpaths = ut.glob(checkpoints_dir, '*.pkl', recursive=True)
    tmp_model_list = []
    for fpath in model_fpaths:
        pass
        tmp_model = abstract_models.BaseModel()
        tmp_model.load_model_state(fpath=fpath)
        tmp_model_list.append(tmp_model)
        #hashid = tmp_model.get_model_history_hashid()
        #dpath = dirname(fpath)
        #new_dpath = join(dirname(dpath), hashid)
        #ut.move(dpath, new_dpath)

    for tmp_model in ut.InteractiveIter(tmp_model_list):
        print(fpath)
        print(sum([len(era['epoch_list']) for era in tmp_model.era_history]))
        tmp_model.show_era_history(fnum=1)


def load_tmp_model():
    from ibeis_cnn import abstract_models
    fpath = '/home/joncrall/.config/ibeis_cnn/training_junction/liberty/model_state.pkl'
    fpath = '/home/joncrall/.config/ibeis_cnn/training_junction/liberty/model_state_dozer.pkl'

    fpath = '/home/joncrall/.config/ibeis_cnn/training_junction/train_patchmetric((15152)nunivgoaibmjsdbs)/model_state.pkl'

    tmp_model = abstract_models.BaseModel()
    tmp_model.load_model_state(fpath=fpath)
    tmp_model.rrr()
    tmp_model.show_era_history(fnum=1)
    pt.iup()


def grab_model_from_dozer():
    remote = 'dozer'
    user = 'joncrall'
    #remote_path = '/home/joncrall/.config/ibeis_cnn/training/liberty/model_state.pkl'
    #local_path = '/home/joncrall/.config/ibeis_cnn/training/liberty/model_state_dozer.pkl'

    remote_path = '/home/joncrall/.config/ibeis_cnn/training/liberty/checkpoints/hist_eras3_epochs30_zqwhqylxyihnknxc/model_state_arch_tiloohclkatusmmp'
    local_path = '/home/joncrall/.config/ibeis_cnn/training/liberty/checkpoints/hist_eras3_epochs30_zqwhqylxyihnknxc/model_state_arch_tiloohclkatusmmp.pkl'

    from os.path import dirname
    ut.ensuredir(dirname(local_path))

    ut.scp_pull(remote_path, local_path, remote, user)

    ut.vd(dirname(local_path))


def convert_old_weights():
    ## DO CONVERSION
    #if False:
    #    old_weights_fpath = ut.truepath('~/Dropbox/ibeis_cnn_weights_liberty.pickle')
    #    if ut.checkpath(old_weights_fpath, verbose=True):
    #        self = model
    #        self.load_old_weights_kw(old_weights_fpath)
    #    self.save_model_state()
    #    #self.save_state()
    pass

