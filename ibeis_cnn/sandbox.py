# flake8: noqa
from six.moves import cPickle as pickle
from ibeis_cnn import abstract_models
from ibeis_cnn.abstract_models import *


def check_external_training_paths():
    """
    Notes:
        http://cs231n.github.io/neural-networks-3/#distr
    """
    checkpoints_dir = '.'
    checkpoints_dir = '/home/joncrall/.config/ibeis_cnn/training_junction/train_patchmetric((315)iofvvdflcllgjkyu)/checkpoints'
    #checkpoints_dir = '/media/raid/work/PZ_MTEST/_ibsdb/_ibeis_cache/nets/train_patchmetric((315)iofvvdflcllgjkyu)/checkpoints/hist_eras2_epochs23_aayzxkezpzjgwupd'
    checkpoints_dir = '/home/joncrall/.config/ibeis_cnn/training_junction/train_patchmetric((1576)fxzkszaajypyzqne)/checkpoints'

    checkpoints_dir = '/home/joncrall/.config/ibeis_cnn/training_junction/liberty/checkpoints'
    checkpoints_dir = '/home/joncrall/.config/ibeis_cnn/training_junction/NNP_Master3_patchmatch-_24285_xatwrytpdbfttoax-/checkpoints'

    from six.moves import cPickle as pickle
    from ibeis_cnn import abstract_models
    from ibeis_cnn.abstract_models import *
    from os.path import *
    model_fpaths = ut.glob(checkpoints_dir, '*.pkl', recursive=True)
    tmp_model_list = []
    for fpath in model_fpaths:
        tmp_model = abstract_models.BaseModel()
        tmp_model.load_model_state(fpath=fpath)
        tmp_model_list.append(tmp_model)
        #hashid = tmp_model.get_history_hashid()
        #dpath = dirname(fpath)
        #new_dpath = join(dirname(dpath), hashid)
        #ut.move(dpath, new_dpath)

    for model in tmp_model_list:
        model.rrr(verbose=False)

    vallist_ = [tmp_model.get_total_epochs() for tmp_model in tmp_model_list]
    tmp_model_list = ut.sortedby(tmp_model_list, vallist_)


    for tmp_model in tmp_model_list:
        #tmp_model.checkpoint_save_model_info()
        print(tmp_model.best_results['train_loss'])
        print(tmp_model.best_results['valid_loss'])
        print('----')

    for tmp_model in ut.InteractiveIter(tmp_model_list):
        print(fpath)
        print(sum([len(era['epoch_list']) for era in tmp_model.era_history]))
        tmp_model.show_era_loss(fnum=1)


def load_tmp_model():
    from ibeis_cnn import abstract_models
    fpath = '/home/joncrall/.config/ibeis_cnn/training_junction/liberty/model_state.pkl'
    fpath = '/home/joncrall/.config/ibeis_cnn/training_junction/liberty/model_state_dozer.pkl'
    fpath = '/home/joncrall/.config/ibeis_cnn/training_junction/train_patchmetric((15152)nunivgoaibmjsdbs)/model_state.pkl'

    tmp_model = abstract_models.BaseModel()
    tmp_model.load_model_state(fpath=fpath)
    tmp_model.rrr()
    tmp_model.show_era_loss(fnum=1)
    pt.iup()


def grab_model_from_dozer():
    remote = 'dozer'
    user = 'joncrall'
    #remote_path = '/home/joncrall/.config/ibeis_cnn/training/liberty/model_state.pkl'
    #local_path = '/home/joncrall/.config/ibeis_cnn/training/liberty/model_state_dozer.pkl'

    remote_path = '/home/joncrall/.config/ibeis_cnn/training/liberty/checkpoints/hist_eras3_epochs30_zqwhqylxyihnknxc/model_state_arch_tiloohclkatusmmp'
    local_path = '/home/joncrall/.config/ibeis_cnn/training/liberty/checkpoints/hist_eras3_epochs30_zqwhqylxyihnknxc/model_state_arch_tiloohclkatusmmp.pkl'

    remote_path = '/home/joncrall/.config/ibeis_cnn/training/liberty/checkpoints'
    local_path = '/home/joncrall/.config/ibeis_cnn/training/liberty'

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


def theano_gradient_funtimes():
    import theano
    import numpy as np
    import theano.tensor as T
    import lasagne
    import ibeis_cnn.theano_ext as theano_ext

    TEST = True

    x_data = np.linspace(-10, 10, 100).astype(np.float32)[:, None, None, None]
    y_data = (x_data ** 2).flatten()[:, None]

    X = T.tensor4('x')
    y = T.matrix('y')

    #x_data_batch =
    #y_data_batch =
    inputs_to_value = {X: x_data[0:16], y: y_data[0:16]}

    l_in = lasagne.layers.InputLayer((16, 1, 1, 1))
    l_out = lasagne.layers.DenseLayer(l_in, num_units=1, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.Orthogonal())

    network_output = lasagne.layers.get_output(l_out, X)

    # TEST NETWORK OUTPUT

    if TEST:
        result = theano_ext.eval_symbol(network_output, inputs_to_value)
        print('network_output = %r' % (result,))

    loss_function = lasagne.objectives.squared_error
    #def loss_function(network_output, labels):
    #    return (network_output - labels) ** 2

    losses = loss_function(network_output, y)
    if TEST:
        result = theano_ext.eval_symbol(losses, inputs_to_value)
        print('losses = %r' % (result,))

    loss = lasagne.objectives.aggregate(losses, mode='mean')

    if TEST:
        result = theano_ext.eval_symbol(loss, inputs_to_value)
        print('loss = %r' % (result,))

    L2 = lasagne.regularization.regularize_network_params(l_out, lasagne.regularization.l2)
    weight_decay = .0001
    loss_regularized = loss + weight_decay * L2
    loss_regularized.name = 'loss_regularized'

    parameters = lasagne.layers.get_all_params(l_out)

    gradients_regularized = theano.grad(loss_regularized, parameters, add_names=True)

    if TEST:
        if False:
            s = T.sum(1 / (1 + T.exp(-X)))
            s.name = 's'
            gs = T.grad(s, X, add_names=True)
            theano.pp(gs)
            inputs_to_value = {X: x_data[0:16], y: y_data[0:16]}
            result = theano_ext.eval_symbol(gs, inputs_to_value)
            print('%s = %r' % (gs.name, result,))
            inputs_to_value = {X: x_data[16:32], y: y_data[16:32]}
            result = theano_ext.eval_symbol(gs, inputs_to_value)
            print('%s = %r' % (gs.name, result,))

        for grad in gradients_regularized:
            result = theano_ext.eval_symbol(grad, inputs_to_value)
            print('%s = %r' % (grad.name, result,))

        grad_on_losses = theano.grad(losses, parameters, add_names=True)

    learning_rate_theano = .0001
    momentum = .9
    updates = lasagne.updates.nesterov_momentum(gradients_regularized, parameters, learning_rate_theano, momentum)

    X_batch = T.tensor4('x_batch')
    y_batch = T.fvector('y_batch')

    func = theano.function(
        inputs=[
            theano.Param(X_batch),
            theano.Param(y_batch)
        ],
        outputs=[network_output, losses],
        #updates=updates,
        givens={
            X: X_batch,
            y: y_batch,
        },
    )

    y_predict_batch, loss_batch = func(inputs_to_value[X], inputs_to_value[y])


    if ut.inIPython():
        import IPython
        IPython.get_ipython().magic('pylab qt4')

    import plottool as pt
    pt.plot(x_data, y_predict)
    pt.iup()
    pass
