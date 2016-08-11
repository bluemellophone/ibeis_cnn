# -*- coding: utf-8 -*-
"""

Directory structure of training

The network directory is the root of the structure and is typically in
_ibeis_cache/nets for ibeis databases. Otherwise it it custom defined (like in
.cache/ibeis_cnn/training for mnist tests)

----------------
|-- netdir <training_dpath>
----------------

Datasets contain ingested data packed into a single file for quick loading.
Data can be presplit into testing /  learning / validation sets.  Metadata is
always a dictionary where keys specify columns and each item corresponds a row
of data. Non-corresponding metadata is currently not supported, but should
probably be located in a manifest.json file.

# TODO: what is the same data has tasks that use different labels?
# need to incorporate that structure.

----------------
|   |-- datasets
|   |   |-- dataset_{dataset_id} *
|   |   |   |-- full
|   |   |   |   |-- {dataset_id}_data.pkl
|   |   |   |   |-- {dataset_id}_labels.pkl
|   |   |   |   |-- {dataset_id}_labels_{task1}.pkl?
|   |   |   |   |-- {dataset_id}_labels_{task2}.pkl?
|   |   |   |   |-- {dataset_id}_metadata.pkl
|   |   |   |-- splits
|   |   |   |   |-- {split_id}_{num} *
|   |   |   |   |   |-- {dataset_id}_{split_id}_data.pkl
|   |   |   |   |   |-- {dataset_id}_{split_id}_labels.pkl
|   |   |   |   |   |-- {dataset_id}_{split_id}_metadata.pkl
----------------

The model directory must keep track of several things:
    * The network architecture (which may depend on the dataset being used)
        - input / output shape
        - network layers
    * The state of learning
        - epoch/era number
        - learning rate
        - regularization rate
    * diagnostic information
        - graphs of loss / error rates
        - images of convolutional weights
        - other visualizations


----------------
|   |-- models
|   |   |-- arch_{archid} *
|   |   |   |-- best_results
|   |   |   |   |-- model_state.pkl
|   |   |   |-- checkpoints
|   |   |   |   |-- {history_id} *
|   |   |   |   |    |-- model_history.pkl
|   |   |   |   |    |-- model_state.pkl
|   |   |   |-- progress
|   |   |   |   |-- <latest>
|   |   |   |-- diagnostics
|   |   |   |   |-- {history_id} *
|   |   |   |   |   |-- <files>

----------------

"""
from __future__ import absolute_import, division, print_function
import functools
import six
import numpy as np
import utool as ut
from os.path import join, exists, dirname, basename
from six.moves import cPickle as pickle  # NOQA
import warnings
from ibeis_cnn import net_strs
from ibeis_cnn import draw_net
from ibeis_cnn import utils
#ut.noinject('ibeis_cnn.abstract_models')
print, rrr, profile = ut.inject2(__name__, '[ibeis_cnn.abstract_models]')


def imwrite_wrapper(show_func):
    r"""
    helper to convert show funcs into imwrite funcs
    Automatically creates filenames if not specified

    DEPRICATE
    """
    def imwrite_func(model, dpath=None, fname=None, dpi=180, asdiagnostic=True,
                     ascheckpoint=None, verbose=1, **kwargs):
        #import plottool as pt
        # Resolve path to save the image
        if dpath is None:
            if ascheckpoint is True:
                history_hashid = model.get_history_hashid()
                dpath = model._get_model_dpath(checkpoint_tag=history_hashid)
            elif asdiagnostic is True:
                dpath = model.get_epoch_diagnostic_dpath()
            else:
                dpath = model.model_dpath
        # Make the image
        fig = show_func(model, **kwargs)
        # Save the image
        fpath = join(dpath, fname)
        fig.savefig(fpath, dpi=dpi)
        #output_fpath = pt.save_figure(fig=fig, dpath=dpath, dpi=dpi, verbose=verbose)
        return fpath
    return imwrite_func


class LearnPropertyInjector(type):
    def __init__(cls, name, bases, dct):
        super(LearnPropertyInjector, cls).__init__(name, bases, dct)
        cls._keys = [
            'momentum',
            'weight_decay',
            'learning_rate',
        ]

        def _make_prop(key):
            def fget(self):
                return self.getitem(key)
            ut.set_funcname(fget, key)

            def fset(self, value):
                self.setitem(key, value)
            ut.set_funcname(fset, key)
            prop = property(fget=fget, fset=fset)
            return prop
        # Inject properties
        for key in cls._keys:
            prop = _make_prop(key)
            setattr(cls, key, prop)


@ut.reloadable_class
@six.add_metaclass(LearnPropertyInjector)
class LearnState(ut.DictLike):
    """ Keeps track of shared variables that can be changed during theano execution """
    def __init__(self, learning_rate, momentum, weight_decay):
        # import ibeis_cnn.__THEANO__ as theano
        self._shared_state = {key: None for key in self._keys}
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

    @property
    def shared(self):
        return self._shared_state

    def keys(self):
        return self._keys

    def getitem(self, key):
        _shared = self._shared_state[key]
        value = _shared.get_value()
        # if value is not None:
        #     value = value.tolist()
        return value

    def setitem(self, key, value):
        import ibeis_cnn.__THEANO__ as theano
        print('[model] setting %s to %.9r' % (key, value))
        _shared = self._shared_state[key]
        if value is None and _shared is not None:
            raise ValueError('Cannot set an initialized shared variable to None.')
        elif _shared is None and value is not None:
            self._shared_state[key] = theano.shared(
                np.cast['float32'](value), name=key)
        elif _shared is not None:
            _shared.set_value(np.cast['float32'](value))


@ut.reloadable_class
class _ModelFitting(object):
    """
    CommandLine:
        python -m ibeis_cnn _ModelFitting.fit:0
    """
    def _init_fit_vars(model, kwargs):
        # era=(group of epochs)
        model.current_era = None
        model.era_history = []
        # Training state
        model.requested_headers = ['learn_loss', 'valid_loss', 'learnval_rat']
        model.preproc_kw   = None
        # Stores current result
        model.best_results = {
            'epoch'      : None,
            'learn_loss' : np.inf,
            'valid_loss' : np.inf,
            'weights'    : None
        }
        model.learn_state = LearnState(
            learning_rate=kwargs.pop('learning_rate', .005),
            momentum=kwargs.pop('momentum', .9),
            weight_decay=kwargs.pop('weight_decay', None),
        )
        # Theano shared state
        model.train_config = {
            'era_schedule': 100,
            'max_epochs': None,
            'learning_rate_adjust': .8,
            'checkpoint_freq': 200,
            'monitor': ut.get_argflag('--monitor'),
        }

    def fit(model, X_train, y_train, X_valid=None, y_valid=None, valid_idx=None):
        r"""
        Trains the network with backprop.

        CommandLine:
            python -m ibeis_cnn _ModelFitting.fit:0
            python -m ibeis_cnn _ModelFitting.fit:0 --vd

        Example0:
            >>> from ibeis_cnn import ingest_data
            >>> from ibeis_cnn.models import MNISTModel
            >>> dataset = ingest_data.grab_mnist_category_dataset()
            >>> dataset = ingest_data.grab_mnist_category_dataset_old()
            >>> model = MNISTModel(batch_size=128, data_shape=dataset.data_shape,
            >>>                    output_dims=dataset.output_dims,
            >>>                    #arch_tag=dataset.alias_key,
            >>>                    learning_rate=.01,
            >>>                    training_dpath=dataset.training_dpath)
            >>> model.encoder = None
            >>> model.initialize_architecture()
            >>> model.train_config['monitor'] = True
            >>> model.learn_state['weight_decay'] = None
            >>> model.print_layer_info()
            >>> #model.reinit_weights()
            >>> X_train, y_train = dataset.load_subset('train')
            >>> valid_idx = None
            >>> model.fit(X_train, y_train)
        """
        from ibeis_cnn import utils
        print('\n[train] --- TRAINING LOOP ---')

        X_learn, y_learn, X_valid, y_valid = model._prefit(
            X_train, y_train, X_valid, y_valid, valid_idx)

        if model.train_config['monitor']:
            model._init_monitor()

        # create theano symbolic expressions that define the network
        theano_backprop = model.build_backprop_func()
        theano_forward = model.build_forward_func()

        epoch = model.best_results['epoch']

        if epoch is None:
            epoch = 0
            print('Initializng training at epoch=%r' % (epoch,))
        else:
            print('Resuming training at epoch=%r' % (epoch,))

        # number of non-best iterations after, that triggers a best save
        # This prevents strings of best-saves one after another
        save_after_best_wait_epochs = 2
        save_after_best_countdown = None

        # Begin training the neural network
        print('learn_state = %s' % ut.repr3(model.learn_state.asdict(), precision=2))
        printcol_info = utils.get_printcolinfo(model.requested_headers)

        model.start_new_era(X_learn, y_learn, X_valid, y_valid)
        utils.print_header_columns(printcol_info)

        tt = ut.Timer(verbose=False)
        while True:
            try:
                # ---------------------------------------
                # Execute backwards and forward passes
                tt.tic()
                learn_info = model._epoch_learn_step(theano_backprop, X_learn, y_learn)
                if learn_info.get('diverged'):
                    break
                valid_info = model._epoch_validate_step(theano_forward, X_valid, y_valid)

                # ---------------------------------------
                # Summarize the epoch
                epoch_info = {
                    'epoch': epoch,
                }
                epoch_info.update(**learn_info)
                epoch_info.update(**valid_info)
                epoch_info['duration'] = tt.toc()
                epoch_info['learnval_rat'] = (
                    epoch_info['learn_loss'] / epoch_info['valid_loss'])

                # ---------------------------------------
                # Record this epoch in history
                model.record_epoch(epoch_info)

                # ---------------------------------------
                # Check how we are learning
                if epoch_info['valid_loss'] < model.best_results['valid_loss']:
                    model.best_results['weights'] = model.get_all_param_values()
                    model.best_results['epoch'] = epoch_info['epoch']
                    for key in model.requested_headers:
                        model.best_results[key] = epoch_info[key]
                    save_after_best_countdown = save_after_best_wait_epochs
                    #epoch_marker = epoch

                # Check frequencies and countdowns
                checkpoint_flag = utils.checkfreq(model.train_config['checkpoint_freq'], epoch)
                if save_after_best_countdown is not None:
                    if save_after_best_countdown == 0:
                        ## Callbacks on best found
                        save_after_best_countdown = None
                        checkpoint_flag = True
                    else:
                        save_after_best_countdown -= 1

                # ---------------------------------------
                # Output Diagnostics

                # Print the epoch
                utils.print_epoch_info(model, printcol_info, epoch_info)

                # Output any diagnostics
                if checkpoint_flag:
                    model.checkpoint_save_model_info()
                    model.save_model_info()
                    model.checkpoint_save_model_state()
                    model.save_model_state()

                if model.train_config['monitor']:
                    model._dump_epoch_monitor()

                # Update learning rate at the start of each new era
                if utils.checkfreq(model.train_config['era_schedule'], epoch):
                    #epoch_marker = epoch
                    frac = model.train_config['learning_rate_adjust']
                    learn_state = model.learn_state
                    learn_state.learning_rate = (learn_state.learning_rate * frac)
                    model.start_new_era(X_learn, y_learn, X_valid, y_valid)
                    utils.print_header_columns(printcol_info)

                # Break on max epochs
                if model.train_config['max_epochs'] is not None:
                    if epoch >= model.train_config['max_epochs']:
                        print('\n[train] maximum number of epochs reached\n')
                        break
                # Increment the epoch
                epoch += 1

            except KeyboardInterrupt:
                print('\n[train] Caught CRTL+C')
                resolution = ''
                while not (resolution.isdigit()):
                    print('\n[train] What do you want to do?')
                    print('[train]     0 - Continue')
                    print('[train]     1 - Shock weights')
                    print('[train]     2 - Save best weights')
                    print('[train]     3 - View training directory')
                    print('[train]     4 - Embed into IPython')
                    print('[train]     5 - Draw current weights')
                    print('[train]     6 - Show training state')
                    print('[train]  ELSE - Stop network training')
                    resolution = str(input('[train] Resolution: '))
                resolution = int(resolution)
                # We have a resolution
                if resolution == 0:
                    print('resuming training...')
                elif resolution == 1:
                    # Shock the weights of the network
                    utils.shock_network(model.output_layer)
                    learn_state = model.learn_state
                    learn_state.learning_rate = learn_state.learning_rate * 2
                    #epoch_marker = epoch
                    utils.print_header_columns(printcol_info)
                elif resolution == 2:
                    # Save the weights of the network
                    model.checkpoint_save_model_info()
                    model.save_model_info()
                    model.checkpoint_save_model_state()
                    model.save_model_state()
                elif resolution == 3:
                    ut.view_directory(model.model_dpath)
                elif resolution == 4:
                    ut.embed()
                elif resolution == 5:
                    output_fpath_list = model.imwrite_weights(index=0)
                    for fpath in output_fpath_list:
                        ut.startfile(fpath)
                elif resolution == 6:
                    model.print_state_str()
                else:
                    # Terminate the network training
                    raise
        # Save the best network
        model.checkpoint_save_model_state()
        model.save_model_state()
        #from ibeis_cnn import harness
        #harness.train(model, X_train, y_train, X_valid, y_valid, dataset, config)

    #@property
    #def best_weights(model):
    #    return model.best_results['weights']
    #@best_weights.setter
    #def best_weights(model, weights):
    #    model.best_results['weights'] = weights

    def start_new_era(model, X_learn, y_learn, X_valid, y_valid):
        """
        Used to denote a change in hyperparameters during training.
        """
        # TODO: fix the training data hashid stuff
        y_hashid = ut.hashstr_arr(y_learn, 'y', alphabet=ut.ALPHABET_27)

        learn_hashid =  str(model.arch_id) + '_' + y_hashid
        # learn_hashid =  str(model.arch_tag) + '_' + y_hashid
        if model.current_era is not None and len(model.current_era['epoch_list']) == 0:
            print('Not starting new era (old one hasnt begun yet')
        else:
            new_era = {
                'learn_hashid': learn_hashid,
                'arch_hashid': model.get_architecture_hashid(),
                # 'arch_tag': model.arch_tag,
                'arch_id': model.arch_id,
                'num_learn': len(y_learn),
                'num_valid': len(y_valid),
                'valid_loss_list': [],
                'learn_loss_list': [],
                'epoch_list': [],
                'learn_state': [model.learn_state.asdict()],
            }
            num_eras = len(model.era_history)
            print('starting new era %d' % (num_eras,))
            #if model.current_era is not None:
            model.current_era = new_era
            model.era_history.append(model.current_era)

    def record_epoch(model, epoch_info):
        """
        Records an epoch in an era.
        """
        # each key/val in an epoch_info dict corresponds to a key/val_list in
        # an era dict.
        for key in epoch_info:
            key_ = key + '_list'
            if key_ not in model.current_era:
                model.current_era[key_] = []
            model.current_era[key_].append(epoch_info[key])

    def _init_monitor(model):
        # FIXME; put into better place
        ut.ensuredir(model.progress_dpath)
        def prog_metric_path(x):
            path_fmt = join(model.progress_dpath, x)
            return ut.get_nonconflicting_path(path_fmt)
        def prog_metric_dir(x):
            return ut.ensuredir(prog_metric_path(x))
        history_progress_dir = prog_metric_dir(
            str(model.arch_id) + '_%02d_history')
        weights_progress_dir = prog_metric_dir(
            str(model.arch_id) + '_%02d_weights')
        history_text_fpath = prog_metric_path(
            str(model.arch_id) + '_%02d_era_history.txt')
        # history_progress_dir = prog_metric_dir(
        #     str(model.arch_tag) + '_%02d_history')
        # weights_progress_dir = prog_metric_dir(
        #     str(model.arch_tag) + '_%02d_weights')
        # history_text_fpath = prog_metric_path(
        #     str(model.arch_tag) + '_%02d_era_history.txt')
        if ut.get_argflag('--vd'):
            ut.vd(model.progress_dpath)

        # Write initial states of the weights
        fpath = model.imwrite_weights(dpath=weights_progress_dir,
                                      fname='weights_' + model.get_history_hashid() + '.png',
                                      fnum=2, verbose=0)
        overwrite_latest_image(fpath, 'weights')
        model._fit_progress_info = {
            'history_progress_dir':  history_progress_dir,
            'history_text_fpath': history_text_fpath,
            'weights_progress_dir': weights_progress_dir,
        }

    def _dump_epoch_monitor(model):
        history_dir = model._fit_progress_info['history_progress_dir']
        weights_dir = model._fit_progress_info['weights_progress_dir']
        text_fpath = model._fit_progress_info['history_text_fpath']

        # Save loss graphs
        fpath = model.imwrite_era_history(dpath=history_dir,
                                          fname='history_' + model.get_history_hashid() + '.png',
                                          fnum=1, verbose=0)
        overwrite_latest_image(fpath, 'history')
        # Save weights images
        fpath = model.imwrite_weights(dpath=weights_dir,
                                      fname='weights_' + model.get_history_hashid() + '.png',
                                      fnum=2, verbose=0)
        overwrite_latest_image(fpath, 'weights')
        # Save text info
        history_text = ut.list_str(model.era_history, newlines=True)
        ut.write_to(text_fpath, history_text, verbose=False)

    def _prefit(model, X_train, y_train, X_valid, y_valid, valid_idx):
        # Center the data by subtracting the mean
        model.check_data_shape(X_train)

        if X_valid is not None:
            assert valid_idx is None, 'Cant specify both valid_idx and X_valid'
            # When X_valid is given assume X_train is actually X_learn
            X_learn = X_train
            y_learn = y_train
        else:
            if valid_idx is None:
                # Split training set into a learning / validation set
                from ibeis_cnn.dataset import stratified_shuffle_split
                train_idx, valid_idx = stratified_shuffle_split(y_train, fractions=[.7, .3],
                                                                rng=432321)
                #import sklearn.cross_validation
                #xvalkw = dict(n_folds=2, shuffle=True, random_state=43432)
                #skf = sklearn.cross_validation.StratifiedKFold(y_train, **xvalkw)
                #train_idx, valid_idx = list(skf)[0]
            elif valid_idx is None and X_valid is None:
                train_idx = ut.index_complement(valid_idx, len(X_train))
            else:
                assert False, 'impossible state'
            # Set to learn network weights
            X_learn = X_train.take(train_idx, axis=0)
            y_learn = y_train.take(train_idx, axis=0)
            # Set to crossvalidate hyperparamters
            X_valid = X_train.take(valid_idx, axis=0)
            y_valid = y_train.take(valid_idx, axis=0)

        model.ensure_training_state(X_learn, y_learn)

        print('Learn y histogram: ' + ut.repr2(ut.dict_hist(y_learn)))
        print('Valid y histogram: ' + ut.repr2(ut.dict_hist(y_valid)))

        # print('\n[train] --- MODEL INFO ---')
        # model.print_architecture_str()
        # model.print_layer_info()

        return X_learn, y_learn, X_valid, y_valid

    def _epoch_learn_step(model, theano_backprop, X_learn, y_learn):
        """
        Backwards propogate -- Run learning set through the backwards pass
        """
        from ibeis_cnn import batch_processing as batch

        learn_outputs = batch.process_batch(
            model, X_learn, y_learn, theano_backprop, randomize_batch_order=True,
            augment_on=True, buffered=True)

        # compute the loss over all learning batches
        learn_info = {}
        learn_info['learn_loss'] = learn_outputs['loss'].mean()

        if 'loss_regularized' in learn_outputs:
            # Regularization information
            learn_info['learn_loss_regularized'] = (
                learn_outputs['loss_regularized'].mean())
            #if 'valid_acc' in model.requested_headers:
            #    learn_info['test_acc']  = learn_outputs['accuracy']
            regularization_amount = (
                learn_outputs['loss_regularized'] - learn_outputs['loss'])
            regularization_ratio = (
                regularization_amount / learn_outputs['loss'])
            regularization_percent = (
                regularization_amount / learn_outputs['loss_regularized'])

            learn_info['regularization_percent'] = regularization_percent
            learn_info['regularization_ratio'] = regularization_ratio

        param_update_mags = {}
        for key, val in learn_outputs.items():
            if key.startswith('param_update_magnitude_'):
                key_ = key.replace('param_update_magnitude_', '')
                param_update_mags[key_] = (val.mean(), val.std())
        if param_update_mags:
            learn_info['param_update_mags'] = param_update_mags

        # If the training loss is nan, the training has diverged
        if np.isnan(learn_info['learn_loss']):
            print('\n[train] train loss is Nan. training diverged\n')
            #import utool
            #utool.embed()
            print('learn_outputs = %r' % (learn_outputs,))
            """
            from ibeis_cnn import draw_net
            draw_net.imwrite_theano_symbolic_graph(theano_backprop)
            """
            import utool
            utool.embed()
            # imwrite_theano_symbolic_graph(thean_expr):
            learn_info['diverged'] = True
        return learn_info

    def _epoch_validate_step(model, theano_forward, X_valid, y_valid):
        """
        Forwards propogate -- Run validation set through the forwards pass
        """
        from ibeis_cnn import batch_processing as batch
        valid_outputs = batch.process_batch(
            model, X_valid, y_valid, theano_forward, augment_on=False,
            randomize_batch_order=False)
        valid_info = {}
        valid_info['valid_loss'] = valid_outputs['loss_determ'].mean()
        valid_info['valid_loss_std'] = valid_outputs['loss_determ'].std()
        if 'valid_acc' in model.requested_headers:
            valid_info['valid_acc'] = valid_outputs['accuracy'].mean()
        return valid_info


@ut.reloadable_class
class _ModelLegacy(object):
    """
    contains old functions for backwards compatibility
    that may be eventually be depricated
    """

    def load_old_weights_kw(model, old_weights_fpath):
        print('[model] loading old model state from: %s' % (
            old_weights_fpath,))
        with open(old_weights_fpath, 'rb') as file_:
            oldkw = pickle.load(file_)
        # Model architecture and weight params
        data_shape  = oldkw['model_shape'][1:]
        input_shape = (None, data_shape[2], data_shape[0], data_shape[1])
        output_dims  = oldkw['output_dims']

        if model.output_dims is None:
            model.output_dims = output_dims

        # Perform checks
        assert input_shape[1:] == model.input_shape[1:], (
            'architecture disagreement')
        assert output_dims == model.output_dims, (
            'architecture disagreement')

        model.preproc_kw = {
            'center_mean' : oldkw['center_mean'],
            'center_std'  : oldkw['center_std'],
        }
        # Set class attributes
        model.best_results = {
            'epoch'          : oldkw['best_epoch'],
            'test_accuracy'  : oldkw['best_test_accuracy'],
            'learn_loss'     : oldkw['best_learn_loss'],
            'valid_accuracy' : oldkw['best_valid_accuracy'],
            'valid_loss'     : oldkw['best_valid_loss'],
            'weights'   : oldkw['best_weights'],
        }

        # Need to build architecture first
        model.initialize_architecture()

        model.encoder = oldkw.get('encoder', None)

        # Set architecture weights
        weights_list = model.best_results['weights']
        model.set_all_param_values(weights_list)

    def load_old_weights_kw2(model, old_weights_fpath):
        print('[model] loading old model state from: %s' % (old_weights_fpath,))
        with open(old_weights_fpath, 'rb') as file_:
            oldkw = pickle.load(file_)

        # Model architecture and weight params
        output_dims = model.best_results['weights'][-1][0]

        if model.output_dims is None:
            model.output_dims = output_dims

        # Set class attributes
        model.preproc_kw = {
            'center_mean' : oldkw['data_whiten_mean'],
            'center_std'  : oldkw['data_whiten_std'],
        }
        model.best_results = {
            'epoch'          : oldkw['best_epoch'],
            'test_accuracy'  : oldkw['best_valid_accuracy'],
            'learn_loss'     : oldkw['best_learn_loss'],
            'valid_accuracy' : oldkw['best_valid_accuracy'],
            'valid_loss'     : oldkw['best_valid_loss'],
            'weights':  oldkw['best_fit_weights']
        }

        # Need to build architecture first
        model.initialize_architecture()
        model.encoder = oldkw.get('data_label_encoder', None)
        model.batch_size = oldkw['train_batch_size']

        # Set architecture weights
        model.set_all_param_values(model.best_results['weights'])


@ut.reloadable_class
class _ModelVisualization(object):
    """
    """

    def show_architecture_image(model, **kwargs):
        layers = model.get_all_layers()
        draw_net.show_architecture_nx_graph(layers, **kwargs)

    def show_era_history(model, fnum=None):
        r"""
        Args:
            fnum (int):  figure number(default = None)

        Returns:
            mpl.Figure: fig

        CommandLine:
            python -m ibeis_cnn --tf BaseModel.show_era_history --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis_cnn.models.abstract_models import *  # NOQA
            >>> model = testdata_model_with_history()
            >>> fnum = None
            >>> model.show_era_history(fnum)
            >>> ut.show_if_requested()
        """
        import plottool as pt
        fnum = pt.ensure_fnum(fnum)
        fig = pt.figure(fnum=fnum, pnum=(1, 1, 1), doclf=True, docla=True)
        next_pnum = pt.make_pnum_nextgen(nRows=2, nCols=2)
        model.show_era_loss(fnum=fnum, pnum=next_pnum(), yscale='log')
        model.show_era_loss(fnum=fnum, pnum=next_pnum(), yscale='linear')
        model.show_era_lossratio(fnum=fnum, pnum=next_pnum())
        model.show_weight_updates(fnum=fnum, pnum=next_pnum())

        pt.set_figtitle('Era History: ' + model.get_history_hashid())
        return fig

    def show_era_lossratio(model, fnum=None, pnum=(1, 1, 1)):
        import plottool as pt

        fnum = pt.ensure_fnum(fnum)

        fig = pt.figure(fnum=fnum, pnum=pnum)
        colors = pt.distinct_colors(len(model.era_history))
        num_eras = len(model.era_history)
        for index, era in enumerate(model.era_history):
            epochs = era['epoch_list']
            learn_loss = np.array(era['learn_loss_list'])
            valid_loss = np.array(era['valid_loss_list'])
            #print('era = %s' % (ut.dict_str(era),))
            #valid_loss_std = np.array(era['valid_loss_std_list'])  # NOQA
            learnval_ratio = learn_loss / valid_loss

            era_color = colors[index]
            #yscale = 'linear'
            #yscale = 'log'
            if index == len(model.era_history) - 1:
                pt.plot(epochs, learnval_ratio, '-o', color=era_color,
                        label='train/valid')
            else:
                pt.plot(epochs, learnval_ratio, '-o', color=era_color)

        #append_phantom_legend_label
        pt.set_xlabel('epoch')
        pt.set_ylabel('train/valid ratio')

        if num_eras > 0:
            pt.legend()
        pt.dark_background()
        return fig

    def show_weight_updates(model, fnum=None, pnum=(1, 1, 1)):
        import plottool as pt
        fnum = pt.ensure_fnum(fnum)
        fig = pt.figure(fnum=fnum, pnum=pnum)
        num_eras = len(model.era_history)
        has_mag_updates = False
        for index, era in enumerate(model.era_history):
            epochs = era['epoch_list']
            if 'param_update_mags_list' not in era:
                continue
            update_mags_list = era['param_update_mags_list']
            # Transpose keys and positions
            param_keys = list(set(ut.flatten([
                dict_.keys() for dict_ in update_mags_list
            ])))
            param_val_list = [
                [list_[param] for list_ in update_mags_list]
                for param in param_keys
            ]
            colors = pt.distinct_colors(len(param_val_list))
            for key, val, color in zip(param_keys, param_val_list, colors):
                update_mag_mean = ut.get_list_column(val, 0)
                update_mag_std = ut.get_list_column(val, 1)  # NOQA
                #pt.plot(epochs, update_mag_mean, marker='-x', color=color)
                if index == len(model.era_history) - 1:
                    # Dont show the first update
                    pt.interval_line_plot(epochs[1:], update_mag_mean[1:],
                                          update_mag_std[1:], marker='x',
                                          linestyle='-', color=color,
                                          label=key)
                else:
                    pt.interval_line_plot(epochs[1:], update_mag_mean[1:],
                                          update_mag_std[1:], marker='x',
                                          linestyle='-', color=color)
                #, label=valid_label, yscale=yscale)
            has_mag_updates = True
            pass
        if has_mag_updates:
            if num_eras > 0:
                pt.legend()

        pt.dark_background()
        return fig

    def show_regularization_stuff(model, fnum=None, pnum=(1, 1, 1)):
        import plottool as pt
        fnum = pt.ensure_fnum(fnum)
        fig = pt.figure(fnum=fnum, pnum=pnum)
        num_eras = len(model.era_history)
        for index, era in enumerate(model.era_history):
            epochs = era['epoch_list']
            if 'update_mags_list' not in era:
                continue
            update_mags_list = era['update_mags_list']
            # Transpose keys and positions
            param_keys = list(set(ut.flatten([
                dict_.keys()
                for dict_ in update_mags_list
            ])))
            param_val_list = [
                [list_[param] for list_ in update_mags_list]
                for param in param_keys
            ]
            colors = pt.distinct_colors(len(param_val_list))
            for key, val, color in zip(param_keys, param_val_list, colors):
                update_mag_mean = ut.get_list_column(val, 0)
                update_mag_std = ut.get_list_column(val, 1)
                if index == len(model.era_history) - 1:
                    pt.interval_line_plot(epochs, update_mag_mean,
                                          update_mag_std, marker='x',
                                          linestyle='-', color=color,
                                          label=key)
                else:
                    pt.interval_line_plot(epochs, update_mag_mean,
                                          update_mag_std, marker='x',
                                          linestyle='-', color=color)
            pass
        if num_eras > 0:
            pt.legend()

        pt.dark_background()
        return fig

    def show_era_loss(model, fnum=None, pnum=(1, 1, 1), yscale='log'):
        import plottool as pt

        fnum = pt.ensure_fnum(fnum)

        fig = pt.figure(fnum=fnum, pnum=pnum)
        num_eras = len(model.era_history)
        colors = pt.distinct_colors(num_eras)
        for index, era in enumerate(model.era_history):
            epochs = era['epoch_list']
            learn_loss = era['learn_loss_list']
            valid_loss = era['valid_loss_list']
            if 'num_valid' in era:
                valid_label = 'valid_loss ' + str(era['num_valid'])
            if 'num_learn' in era:
                learn_label = 'learn_loss ' + str(era['num_learn'])

            era_color = colors[index]
            if index == len(model.era_history) - 1:
                pt.plot(epochs, valid_loss, '-x', color=era_color,
                        label=valid_label, yscale=yscale)
                pt.plot(epochs, learn_loss, '-o', color=era_color,
                        label=learn_label, yscale=yscale)
            else:
                pt.plot(epochs, valid_loss, '-x', color=era_color,
                        yscale=yscale)
                pt.plot(epochs, learn_loss, '-o', color=era_color,
                        yscale=yscale)

        # append_phantom_legend_label
        pt.set_xlabel('epoch')
        pt.set_ylabel('loss')
        if num_eras > 0:
            pt.legend()
        pt.dark_background()
        return fig

    def show_weights_image(model, index=0, *args, **kwargs):
        import plottool as pt
        network_layers = model.get_all_layers()
        cnn_layers = [layer_ for layer_ in network_layers
                      if hasattr(layer_, 'W')]
        layer = cnn_layers[index]
        all_weights = layer.W.get_value()
        layername = net_strs.make_layer_str(layer)
        fig = draw_net.show_convolutional_weights(all_weights, **kwargs)
        history_hashid = model.get_history_hashid()
        figtitle = layername + '\n' + history_hashid
        pt.set_figtitle(figtitle, subtitle='shape=%r, sum=%.4f, l2=%.4f' %
                        (all_weights.shape, all_weights.sum(),
                         (all_weights ** 2).sum()))
        return fig

    # --- IMAGE WRITE

    imwrite_era_history = imwrite_wrapper(show_era_history)

    imwrite_weights = imwrite_wrapper(show_weights_image)

    #imwrite_architecture = imwrite_wrapper(show_architecture_image)

    def imwrite_architecture(model, fpath='arch_image.png'):
        # FIXME
        layers = model.get_all_layers()
        draw_net.imwrite_architecture(layers, fpath)
        ut.startfile(fpath)


@ut.reloadable_class
class _ModelStrings(object):
    """
    """

    def get_state_str(model, other_override_reprs={}):
        era_history_str = ut.list_str(
            [ut.dict_str(era, truncate=True, sorted_=True)
             for era in model.era_history], strvals=True)

        override_reprs = {
            'best_results': ut.dict_str(model.best_results),
            #'best_weights': ut.truncate_str(str(model.best_weights)),
            'preproc_kw': ('None' if model.preproc_kw is None else
                           ut.dict_str(model.preproc_kw, truncate=True)),
            'learn_state': ut.dict_str(model.learn_state),
            'era_history': era_history_str,
        }
        override_reprs.update(other_override_reprs)
        keys = list(set(model.__dict__.keys()) - set(override_reprs.keys()))
        for key in keys:
            if ut.is_func_or_method(model.__dict__[key]):
                # rrr support
                continue
            override_reprs[key] = repr(model.__dict__[key])

        state_str = ut.dict_str(override_reprs, sorted_=True, strvals=True)
        return state_str

    def get_architecture_str(model, sep='_', with_noise=True):
        r"""
        with_noise is a boolean that specifies if layers that doesnt
        affect the flow of information in the determenistic setting are to be
        included. IE get rid of dropout.

        CommandLine:
            python -m ibeis_cnn.models.abstract_models --test-_ModelStrings.get_architecture_str:0

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis_cnn.models.abstract_models import *  # NOQA
            >>> from ibeis_cnn.models import MNISTModel
            >>> model = MNISTModel(batch_size=128, data_shape=(24, 24, 1),
            >>>                    output_dims=10, training_dpath='.')
            >>> model.initialize_architecture()
            >>> result = model.get_architecture_str(sep=ut.NEWLINE, with_noise=False)
            >>> print(result)
            InputLayer(name=I0,shape=(128, 1, 24, 24))
            Conv2DDNNLayer(name=C1,num_filters=32,stride=(1, 1),nonlinearity=rectify)
            MaxPool2DDNNLayer(name=P1,stride=(2, 2))
            Conv2DDNNLayer(name=C2,num_filters=32,stride=(1, 1),nonlinearity=rectify)
            MaxPool2DDNNLayer(name=P2,stride=(2, 2))
            DenseLayer(name=F3,num_units=256,nonlinearity=rectify)
            DenseLayer(name=O4,num_units=10,nonlinearity=softmax)
        """
        if model.output_layer is None:
            return ''
        layer_list = model.get_all_layers(with_noise=with_noise)
        layer_str_list = [net_strs.make_layer_str(layer)
                          for layer in layer_list]
        architecture_str = sep.join(layer_str_list)
        return architecture_str

    def get_layer_info_str(model):
        """
        CommandLine:
            python -m ibeis_cnn.models.abstract_models --test-_ModelStrings.get_layer_info_str:0

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis_cnn.models.abstract_models import *  # NOQA
            >>> from ibeis_cnn.models import MNISTModel
            >>> model = MNISTModel(batch_size=128, data_shape=(24, 24, 1),
            >>>                    output_dims=10, training_dpath='.')
            >>> model.initialize_architecture()
            >>> result = model.get_layer_info_str()
            >>> print(result)
            Network Structure:
             index  Name  Layer               Outputs      Bytes OutShape           Params
             0      I0    InputLayer              576    294,912 (128, 1, 24, 24)   []
             1      C1    Conv2DDNNLayer       12,800  6,556,928 (128, 32, 20, 20)  [C1.W(32,1,5,5, {t,r}), C1.b(32, {t})]
             2      P1    MaxPool2DDNNLayer     3,200  1,638,400 (128, 32, 10, 10)  []
             3      C2    Conv2DDNNLayer        1,152    692,352 (128, 32, 6, 6)    [C2.W(32,32,5,5, {t,r}), C2.b(32, {t})]
             4      P2    MaxPool2DDNNLayer       288    147,456 (128, 32, 3, 3)    []
             5      D2    DropoutLayer            288    147,456 (128, 32, 3, 3)    []
             6      F3    DenseLayer              256    427,008 (128, 256)         [F3.W(288,256, {t,r}), F3.b(256, {t})]
             7      D3    DropoutLayer            256    131,072 (128, 256)         []
             8      O4    DenseLayer               10     15,400 (128, 10)          [O4.W(256,10, {t,r}), O4.b(10, {t})]
            ...this model has 103,018 learnable parameters
            ...this model will use 10,050,984 bytes = 9.59 MB
        """
        return net_strs.get_layer_info_str(model.get_all_layers())

    @property
    def total_epochs(model):
        return sum([len(era['epoch_list']) for era in model.era_history])

    @property
    def total_eras(model):
        return len(model.era_history)

    # --- PRINTING

    def print_state_str(model, **kwargs):
        print(model.get_state_str(**kwargs))

    def print_layer_info(model):
        net_strs.print_layer_info(model.get_all_layers())

    def print_architecture_str(model, sep='\n  '):
        architecture_str = model.get_architecture_str(sep=sep)
        if architecture_str is None or architecture_str == '':
            architecture_str = 'UNDEFINED'
        print('\nArchitecture:' + sep + architecture_str)

    def print_dense_architecture_str(model):
        print('\n---- Arch Str')
        model.print_architecture_str(sep='\n')
        print('\n---- Layer Info')
        model.print_layer_info()
        print('\n---- HashID')
        print('hashid=%r' % (model.get_architecture_hashid()),)
        print('----')


@ut.reloadable_class
class _ModelIDs(object):

    def _init_id_vars(model, kwargs):
        pass
        # FIXME: figure out how arch tag fits in here
        #model.name = kwargs.pop('name', None)
        model.arch_tag = kwargs.pop('arch_tag', None)
        model.name = kwargs.pop('name', None)
        #if model.name is None:
        #    model.name = ut.get_classname(model.__class__, local=True)

    def __nice__(self):
        return '(' + self.get_arch_nice() + ' ' + self.get_history_nice() + ')'

    @property
    def hash_id(model):
        arch_hashid = model.get_architecture_hashid()
        history_hashid = model.get_history_hashid()
        hashid = ut.hashstr27(history_hashid + arch_hashid)
        return hashid

    @property
    def arch_id(model):
        """
        CommandLine:
            python -m ibeis_cnn.models.abstract_models --test-_ModelIDs.arch_id:0

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis_cnn.models.abstract_models import *  # NOQA
            >>> from ibeis_cnn.models import MNISTModel
            >>> model = MNISTModel(batch_size=128, data_shape=(24, 24, 1),
            >>>                    output_dims=10, training_dpath='.')
            >>> model.initialize_architecture()
            >>> result = str(model.arch_id)
            >>> print(result)
        """
        arch_id = 'arch_' + model.get_arch_nice() + '_' + model.get_architecture_hashid()
        return arch_id

    @property
    def hist_id(model):
        r"""
        CommandLine:
            python -m ibeis_cnn.models.abstract_models --test-_ModelIDs.hist_id:0

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis_cnn.models.abstract_models import *  # NOQA
            >>> model = testdata_model_with_history()
            >>> result = str(history_hashid)
            >>> print(result)
            eras003_epochs0012_eeeejtddhhhkoaim
        """
        hashid = model.get_history_hashid()
        nice = model.get_history_nice()
        history_id = nice + hashid
        return history_id

    def get_architecture_hashid(model):
        """
        Returns a hash identifying the architecture of the determenistic net.
        This does not involve any dropout or noise layers, nor does the
        initialization of the weights matter.
        """
        arch_str = model.get_architecture_str(with_noise=False)
        arch_hashid = ut.hashstr27(arch_str, hashlen=8)
        return arch_hashid

    def get_history_hashid(model):
        r"""
        Builds a hashid that uniquely identifies the architecture and the
        training procedure this model has gone through to produce the current
        architecture weights.
        """
        era_hash_list = [ut.hashstr27(ut.repr2(era))
                         for era in model.era_history]
        era_hash_str = ''.join(era_hash_list)
        history_hashid = ut.hashstr27(era_hash_str, hashlen=8)
        return history_hashid

    def get_arch_nice(model):
        """
        Makes a string that shows the number of input units, output units,
        hidden units, parameters, and model depth.

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis_cnn.models.abstract_models import *  # NOQA
            >>> from ibeis_cnn.models import MNISTModel
            >>> model = MNISTModel(batch_size=128, data_shape=(24, 24, 1),
            >>>                    output_dims=10, training_dpath='.')
            >>> model.initialize_architecture()
            >>> result = str(model.get_arch_nice())
            >>> print(result)
        """
        if model.arch_tag is not None:
            return model.arch_tag
        elif model.output_layer is None:
            return 'NOARCH'
        else:
            weighted_layers = model.get_all_layers(with_noise=False, with_weightless=False)
            info_list = [net_strs.get_layer_info(layer) for layer in weighted_layers]
            # The number of units depends if you look at things via input or output
            # does a convolutional layer have its outputs or the outputs of the pooling layer?
            #num_units1 = sum([info['num_outputs'] for info in info_list])
            nhidden = sum([info['num_inputs'] for info in info_list[1:]])
            #num_units2 += net_strs.get_layer_info(model.output_layer)['num_outputs']
            depth = len(weighted_layers)
            nparam = sum([info['num_params'] for info in info_list])
            nin = np.prod(model.data_shape)
            nout = model.output_dims
            fmtdict = dict(
                nin=nin,
                nout=nout,
                depth=depth,
                nhidden=nhidden,
                nparam=nparam,
                #logmcomplex=int(np.round(np.log10(nhidden + nparam)))
                mcomplex=int(np.round((nhidden + nparam) / 1000))
            )
            #nice = 'i{nin}_o{nout}_d{depth}_h{nhidden}_p{nparam}'.format(**fmtdict)
            # Use a complexity measure
            nice = 'i{nin}_o{nout}_d{depth}_c{mcomplex}'.format(**fmtdict)
            # Use a logspace complexity measure
            #nice = 'i{nin}_o{nout}_d{depth}_c{logmcomplex}'.format(**fmtdict)
            return nice

    def get_history_nice(model):
        nice = 'era%03d_epoch%04d' % (model.total_eras, model.total_epochs)
        return nice


@ut.reloadable_class
class _ModelIO(object):

    def _init_io_vars(model, kwargs):
        model.training_dpath = kwargs.pop('training_dpath', '.')

    def print_structure(model):
        print(model.model_dpath)
        print(model.arch_dpath)
        print(model.best_dpath)
        print(model.progress_dpath)
        print(model.checkpoint_dpath)
        print(model.diagnostic_dpath)

    @property
    def model_dpath(model):
        return join(model.training_dpath, 'models')

    @property
    def arch_dpath(model):
        return join(model.model_dpath, model.arch_id)

    @property
    def best_dpath(model):
        return join(model.arch_dpath, 'best')

    @property
    def checkpoint_dpath(model):
        return join(model.arch_dpath, 'checkpoints')

    @property
    def progress_dpath(model):
        return join(model.arch_dpath, 'progress')

    @property
    def diagnostic_dpath(model):
        return join(model.arch_dpath, 'diagnostics')

    def get_epoch_diagnostic_dpath(model, epoch=None):
        import utool as ut
        history_hashid = model.get_history_hashid()
        diagnostic_dpath = model.diagnostic_dpath
        ut.ensuredir(diagnostic_dpath)
        epoch_dpath = ut.unixjoin(diagnostic_dpath, history_hashid)
        ut.ensuredir(epoch_dpath)
        return epoch_dpath

    def list_saved_checkpoints(model):
        dpath = model._get_model_dpath(None, True)
        checkpoint_dirs = sorted(ut.glob(dpath, '*', fullpath=False))
        return checkpoint_dirs

    def _get_model_dpath(model, dpath, checkpoint_tag):
        dpath = model.arch_dpath if dpath is None else dpath
        if checkpoint_tag is not None:
            # checkpoint dir requested
            dpath = join(dpath, 'checkpoints')
            if checkpoint_tag is not True:
                # specific checkpoint requested
                dpath = join(dpath, checkpoint_tag)
        return dpath

    def _get_model_file_fpath(model, default_fname, fpath, dpath, fname,
                              checkpoint_tag):
        if fpath is None:
            fname = default_fname if fname is None else fname
            dpath = model._get_model_dpath(dpath, checkpoint_tag)
            fpath = join(dpath, fname)
        else:
            assert checkpoint_tag is None, 'fpath overrides all other settings'
            assert dpath is None, 'fpath overrides all other settings'
            assert fname is None, 'fpath overrides all other settings'
        return fpath

    def resolve_fuzzy_checkpoint_pattern(model, checkpoint_pattern,
                                         extern_dpath=None):
        r"""
        tries to find a matching checkpoint so you dont have to type a full
        hash
        """
        dpath = model._get_model_dpath(extern_dpath, checkpoint_pattern)
        if exists(dpath):
            checkpoint_tag = checkpoint_pattern
        else:
            checkpoint_dpath = dirname(dpath)
            checkpoint_globpat = '*' + checkpoint_pattern + '*'
            matching_dpaths = ut.glob(checkpoint_dpath, checkpoint_globpat)
            if len(matching_dpaths) == 0:
                raise RuntimeError(
                    'Could not resolve checkpoint_pattern=%r. No Matches' %
                    (checkpoint_pattern,))
            elif len(matching_dpaths) > 1:
                raise RuntimeError(
                    ('Could not resolve checkpoint_pattern=%r. '
                        'matching_dpaths=%r. Too many matches') %
                    (checkpoint_pattern, matching_dpaths))
            else:
                checkpoint_tag = basename(matching_dpaths[0])
                print('Resolved checkpoint pattern to checkpoint_tag=%r' %
                        (checkpoint_tag,))
        return checkpoint_tag

    def has_saved_state(model, checkpoint_tag=None):
        """
        Check if there are any saved model states matching the checkpoing tag.
        """
        fpath = model.get_model_state_fpath(checkpoint_tag=checkpoint_tag)
        if checkpoint_tag is not None:
            ut.assertpath(fpath)
        return ut.checkpath(fpath)

    def get_model_state_fpath(model, fpath=None, dpath=None, fname=None,
                              checkpoint_tag=None):
        default_fname = 'model_state_arch_%s.pkl' % (
            model.get_architecture_hashid())
        model_state_fpath = model._get_model_file_fpath(default_fname, fpath,
                                                        dpath, fname,
                                                        checkpoint_tag)
        return model_state_fpath

    def get_model_info_fpath(model, fpath=None, dpath=None, fname=None,
                             checkpoint_tag=None):
        default_fname = 'model_info_arch_%s.pkl' % (
            model.get_architecture_hashid())
        model_state_fpath = model._get_model_file_fpath(default_fname, fpath,
                                                        dpath, fname,
                                                        checkpoint_tag)
        return model_state_fpath

    def checkpoint_save_model_state(model):
        history_hashid = model.get_history_hashid()
        fpath = model.get_model_state_fpath(checkpoint_tag=history_hashid)
        ut.ensuredir(dirname(fpath))
        model.save_model_state(fpath=fpath)

    def checkpoint_save_model_info(model):
        history_hashid = model.get_history_hashid()
        fpath = model.get_model_info_fpath(checkpoint_tag=history_hashid)
        ut.ensuredir(dirname(fpath))
        model.save_model_info(fpath=fpath)

    def save_model_state(model, **kwargs):
        """ saves current model state """
        current_weights = model.get_all_param_values()
        model_state = {
            'best_results': model.best_results,
            'preproc_kw':   model.preproc_kw,
            'current_weights': current_weights,

            'input_shape':  model.input_shape,
            'data_shape': model.data_shape,
            'batch_size': model.data_shape,
            'output_dims':  model.output_dims,

            'era_history':  model.era_history,
            # 'arch_tag': model.arch_tag,
        }
        model_state_fpath = model.get_model_state_fpath(**kwargs)
        print('saving model state to: %s' % (model_state_fpath,))
        with open(model_state_fpath, 'wb') as file_:
            pickle.dump(model_state, file_, protocol=2)  # Use protocol 2 to support python2 and 3
        print('finished saving')

    def save_model_info(model, **kwargs):
        """ save model information (history and results but no weights) """
        model_info = {
            'best_results': model.best_results,
            'input_shape':  model.input_shape,
            'output_dims':  model.output_dims,
            'era_history':  model.era_history,
        }
        model_info_fpath = model.get_model_state_fpath(**kwargs)
        print('saving model info to: %s' % (model_info_fpath,))
        with open(model_info_fpath, 'wb') as file_:
            pickle.dump(model_info, file_, protocol=2)  # Use protocol 2 to support python2 and 3

        print('finished saving')

    def load_model_state(model, **kwargs):
        """
        kwargs = {}
        TODO: resolve load_model_state and load_extern_weights into a single
            function that is less magic in what it does and more
            straightforward
        """
        model_state_fpath = model.get_model_state_fpath(**kwargs)
        print('[model] loading model state from: %s' % (model_state_fpath,))
        with open(model_state_fpath, 'rb') as file_:
            model_state = pickle.load(file_)
        if model.__class__.__name__ != 'BaseModel':
            assert model_state['input_shape'][1:] == model.input_shape[1:], (
                'architecture disagreement')
            assert model_state['output_dims'] == model.output_dims, (
                'architecture disagreement')
            model.preproc_kw   = model_state['preproc_kw']
        else:
            # HACK TO LOAD ABSTRACT MODEL FOR DIAGNOSITIC REASONS
            print("WARNING LOADING ABSTRACT MODEL")
        model.best_results = model_state['best_results']
        model.input_shape  = model_state['input_shape']
        model.output_dims  = model_state['output_dims']
        model.era_history  = model_state.get('era_history', [None])
        if model.__class__.__name__ != 'BaseModel':
            # hack for abstract model
            # model.output_layer is not None
            model.set_all_param_values(model.best_results['weights'])

    def load_extern_weights(model, **kwargs):
        """ load weights from another model """
        model_state_fpath = model.get_model_state_fpath(**kwargs)
        print('[model] loading extern weights from: %s' % (model_state_fpath,))
        with open(model_state_fpath, 'rb') as file_:
            model_state = pickle.load(file_)
        if False or utils.VERBOSE_CNN:
            print('External Model State:')
            print(ut.dict_str(model_state, truncate=True))
        # check compatibility with this architecture
        assert model_state['input_shape'][1:] == model.input_shape[1:], (
            'architecture disagreement')
        assert model_state['output_dims'] == model.output_dims, (
            'architecture disagreement')
        # Just set the weights, no other training state variables
        model.set_all_param_values(model_state['best_weights'])
        # also need to make sure the same preprocessing is used
        # TODO make this a layer?
        model.preproc_kw = model_state['preproc_kw']
        model.era_history = model_state['era_history']


class _ModelUtility(object):

    def set_all_param_values(model, weights_list):
        import ibeis_cnn.__LASAGNE__ as lasagne
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', '.*topo.*')
            lasagne.layers.set_all_param_values(
                model.output_layer, weights_list)

    def get_all_param_values(model):
        import ibeis_cnn.__LASAGNE__ as lasagne
        weights_list = lasagne.layers.get_all_param_values(model.output_layer)
        return weights_list

    def get_all_params(model, **tags):
        import ibeis_cnn.__LASAGNE__ as lasagne
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', '.*topo.*')
            parameters = lasagne.layers.get_all_params(
                model.output_layer, **tags)
            return parameters

    def get_all_layers(model, with_noise=True, with_weightless=True):
        import ibeis_cnn.__LASAGNE__ as lasagne
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', '.*topo.*')
            warnings.filterwarnings('ignore', '.*layer.get_all_layers.*')
            assert model.output_layer is not None, 'need to initialize'
            layer_list_ = lasagne.layers.get_all_layers(model.output_layer)
        layer_list = layer_list_
        if not with_noise:
            # Remove dropout / gaussian noise layers
            layer_list = [
                layer for layer in layer_list_
                if layer.__class__.__name__ not in lasagne.layers.noise.__all__
            ]
        if not with_weightless:
            # Remove layers without weights
            layer_list = [layer for layer in layer_list if hasattr(layer, 'W')]
        return layer_list

    def get_output_layer(model):
        if model.output_layer is not None:
            return model.output_layer
        else:
            return None

    def check_data_shape(model, X_train):
        """ Check to make sure data agrees with model input """
        input_layer = model.get_all_layers()[0]
        expected_item_shape = ut.take(input_layer.shape[1:], [1, 2, 0])
        expected_item_shape = tuple(expected_item_shape)
        given_item_shape = X_train.shape[1:]
        if given_item_shape != expected_item_shape:
            raise ValueError(
                'inconsistent item shape: ' +
                ('expected_item_shape = %r, ' % (expected_item_shape,)) +
                ('given_item_shape = %r' % (given_item_shape,))
            )

    def make_random_testdata(model, num=1000, seed=0, cv2_format=False):
        print('made random testdata')
        rng = np.random.RandomState(seed)
        num_labels = num
        num_data   = num * model.data_per_label_input
        X_unshared = rng.rand(num_data, *model.data_shape)
        y_unshared = rng.rand(num_labels) * (model.output_dims + 1)
        X_unshared = X_unshared.astype(np.float32)
        y_unshared = y_unshared.astype(np.int32)
        if ut.VERBOSE:
            print('made random testdata')
            print('size(X_unshared) = %r' % (
                ut.get_object_size_str(X_unshared),))
            print('size(y_unshared) = %r' % (
                ut.get_object_size_str(y_unshared),))
        if cv2_format:
            pass
        return X_unshared, y_unshared


class _ModelBackend(object):
    """
    Functions that build and compile theano exepressions
    """

    def _init_compile_vars(model, kwargs):
        model._theano_exprs = ut.ddict(lambda: None)
        model._theano_backprop = None
        model._theano_forward = None
        model._theano_predict = None
        model._mode = None

    @property
    def theano_mode(model):
        # http://deeplearning.net/software/theano/library/compile/function.html#theano.compile.function.function
        # http://deeplearning.net/software/theano/tutorial/modes.html
        # theano.compile.MonitorMode
        # theano.compile.FAST_COMPILE
        # theano.compile.FAST_RUN
        # theano.compile.Mode(linker=None, optimizer='default')
        return model._mode
        #mode = None
        #mode = theano.compile.FAST_COMPILE
        #return mode

    @theano_mode.setter
    def theano_mode(model, mode):
        import ibeis_cnn.__THEANO__ as theano
        if mode is None:
            pass
        elif mode == 'FAST_COMPILE':
            mode = theano.compile.FAST_COMPILE
        elif mode == 'FAST_RUN':
            mode = theano.compile.FAST_RUN
        else:
            raise ValueError('Unknown mode=%r' % (mode,))
        return mode

    def build(model):
        print('[model] --- BUILDING SYMBOLIC THEANO FUNCTIONS ---')
        model.build_backprop_func()
        model.build_forward_func()
        model.build_predict_func()
        print('[model] --- FINISHED BUILD ---')

    def build_predict_func(model):
        if model._theano_predict is None:
            import ibeis_cnn.__THEANO__ as theano
            from ibeis_cnn.__THEANO__ import tensor as T  # NOQA
            print('[model.build] request_predict')

            netout_exprs = model._get_network_output()
            network_output_determ = netout_exprs['network_output_determ']
            unlabeled_outputs = model._get_unlabeled_outputs()

            X, X_batch = model._get_batch_input_exprs()
            y, y_batch = model._get_batch_output_exprs()
            theano_predict = theano.function(
                inputs=[theano.In(X_batch)],
                outputs=[network_output_determ] + unlabeled_outputs,
                givens={X: X_batch},
                updates=None,
                mode=model.theano_mode,
                name=':theano_predict:explicit'
            )
            model._theano_predict = theano_predict
        return model._theano_predict

    def build_forward_func(model):
        if model._theano_forward is None:
            import ibeis_cnn.__THEANO__ as theano
            from ibeis_cnn.__THEANO__ import tensor as T  # NOQA

            X, X_batch = model._get_batch_input_exprs()
            y, y_batch = model._get_batch_output_exprs()
            labeled_outputs = model._get_labeled_outputs()
            unlabeled_outputs = model._get_unlabeled_outputs()

            loss_exprs = model._get_loss_exprs()
            loss_determ = loss_exprs['loss_determ']

            print('[model.build] request_forward')
            theano_forward = theano.function(
                inputs=[theano.In(X_batch), theano.In(y_batch)],
                outputs=[loss_determ] + labeled_outputs + unlabeled_outputs,
                givens={X: X_batch, y: y_batch},
                updates=None,
                mode=model.theano_mode,
                name=':theano_forward:explicit'
            )
            model._theano_forward = theano_forward
        return model._theano_forward

    def build_backprop_func(model):
        if model._theano_backprop is None:
            print('[model.build] request_backprop')
            import ibeis_cnn.__THEANO__ as theano
            from ibeis_cnn.__THEANO__ import tensor as T  # NOQA

            X, X_batch = model._get_batch_input_exprs()
            y, y_batch = model._get_batch_output_exprs()
            labeled_outputs = model._get_labeled_outputs()

            # Build backprop losses
            loss_exprs = model._get_loss_exprs()
            loss             = loss_exprs['loss']
            loss_regularized = loss_exprs['loss_regularized']

            if loss_regularized is not None:
                backprop_loss_ = loss_regularized
            else:
                backprop_loss_ = loss

            backprop_losses = []
            if loss_regularized is not None:
                backprop_losses.append(loss_regularized)
            backprop_losses.append(loss)

            # Updates network parameters based on the training loss
            parameters = model.get_all_params(trainable=True)

            updates = model._make_updates(parameters, backprop_loss_)
            monitor_outputs = model._make_monitor_outputs(parameters, updates)

            theano_backprop = theano.function(
                inputs=[theano.In(X_batch), theano.In(y_batch)],
                outputs=(backprop_losses + labeled_outputs + monitor_outputs),
                givens={X: X_batch, y: y_batch},
                updates=updates,
                mode=model.theano_mode,
                name=':theano_backprop:explicit',
            )
            model._theano_backprop = theano_backprop
        return model._theano_backprop

    def _get_batch_input_exprs(model):
        from ibeis_cnn.__THEANO__ import tensor as T  # NOQA
        if model._theano_exprs['batch_input'] is None:
            #if input_type is None:
            input_type = T.tensor4
            X = input_type('x')
            X_batch = input_type('x_batch')
            model._theano_exprs['batch_input'] = X, X_batch
        return model._theano_exprs['batch_input']

    def _get_batch_output_exprs(model):
        from ibeis_cnn.__THEANO__ import tensor as T  # NOQA
        if model._theano_exprs['batch_output'] is None:
            #if output_type is None:
            output_type = T.ivector
            y = output_type('y')
            y_batch = output_type('y_batch')
            model._theano_exprs['batch_output'] = y, y_batch
        return model._theano_exprs['batch_output']

    def _get_loss_exprs(model):
        r"""
        Requires that a custom loss function is defined in the inherited class
        """
        if model._theano_exprs['loss'] is None:
            import ibeis_cnn.__LASAGNE__ as lasagne
            with warnings.catch_warnings():
                #warnings.filterwarnings('ignore', '.*topo.*')
                #warnings.filterwarnings('ignore', '.*get_all_non_bias_params.*')
                #warnings.filterwarnings('ignore', '.*layer.get_output.*')

                X, X_batch = model._get_batch_input_exprs()
                y, y_batch = model._get_batch_output_exprs()

                netout_exprs = model._get_network_output()
                network_output_learn = netout_exprs['network_output_learn']
                network_output_determ = netout_exprs['network_output_determ']

                print('Building symbolic loss function')
                losses = model.loss_function(network_output_learn, y_batch)
                loss = lasagne.objectives.aggregate(losses, mode='mean')
                loss.name = 'loss'

                print('Building symbolic loss function (determenistic)')
                losses_determ = model.loss_function(network_output_determ, y_batch)
                loss_determ = lasagne.objectives.aggregate(losses_determ,
                                                           mode='mean')
                loss_determ.name = 'loss_determ'

                # Regularize
                # TODO: L2 should be one of many available options for
                # regularization
                L2 = lasagne.regularization.regularize_network_params(
                    model.output_layer, lasagne.regularization.l2)
                weight_decay = model.learn_state['weight_decay']
                if weight_decay is not None or weight_decay == 0:
                    regularization_term = weight_decay * L2
                    regularization_term.name = 'regularization_term'
                    #L2 = lasagne.regularization.l2(model.output_layer)
                    loss_regularized = loss + regularization_term
                    loss_regularized.name = 'loss_regularized'
                else:
                    loss_regularized = None

                loss_exprs = {
                    'loss': loss,
                    'loss_determ': loss_determ,
                    'loss_regularized': loss_regularized,
                }
            model._theano_exprs['loss'] = loss_exprs
        return model._theano_exprs['loss']

    def _make_updates(model, parameters, backprop_loss_):
        import ibeis_cnn.__LASAGNE__ as lasagne
        import ibeis_cnn.__THEANO__ as theano
        from ibeis_cnn.__THEANO__ import tensor as T  # NOQA

        grads = theano.grad(backprop_loss_, parameters, add_names=True)

        shared_learning_rate = model.learn_state.shared['learning_rate']
        momentum = model.learn_state['momentum']

        updates = lasagne.updates.nesterov_momentum(
            loss_or_grads=grads,
            params=parameters,
            learning_rate=shared_learning_rate,
            momentum=momentum
            # add_names=True  # TODO; commit to lasange
        )

        # workaround for pylearn2 bug documented in
        # https://github.com/Lasagne/Lasagne/issues/728
        for param, update in updates.items():
            if param.broadcastable != update.broadcastable:
                updates[param] = T.patternbroadcast(update, param.broadcastable)
        return updates

    def _get_network_output(model):
        if model._theano_exprs['netout'] is None:
            import ibeis_cnn.__LASAGNE__ as lasagne
            X, X_batch = model._get_batch_input_exprs()

            network_output_learn = lasagne.layers.get_output(
                model.output_layer, X_batch)
            network_output_learn.name = 'network_output_learn'

            network_output_determ = lasagne.layers.get_output(
                model.output_layer, X_batch, deterministic=True)
            network_output_determ.name = 'network_output_determ'

            netout_exprs = {
                'network_output_learn': network_output_learn,
                'network_output_determ': network_output_determ,
            }
            model._theano_exprs['netout'] = netout_exprs
        return model._theano_exprs['netout']

    def _get_unlabeled_outputs(model):
        if model._theano_exprs['unlabeled_out'] is None:
            netout_exprs = model._get_network_output()
            network_output_determ = netout_exprs['network_output_determ']
            model._theano_exprs['unlabeled_out'] = model.build_unlabeled_output_expressions(
                network_output_determ)
        return model._theano_exprs['unlabeled_out']

    def _get_labeled_outputs(model):
        if model._theano_exprs['labeled_out'] is None:
            netout_exprs = model._get_network_output()
            network_output_determ = netout_exprs['network_output_determ']
            y, y_batch = model._get_batch_output_exprs()
            model._theano_exprs['labeled_out'] = model.build_labeled_output_expressions(
                network_output_determ, y_batch)
        return model._theano_exprs['labeled_out']

    def _make_monitor_outputs(model, parameters, updates):
        """
        Builds parameters to monitor the magnitude of updates durning learning
        """
        from ibeis_cnn.__THEANO__ import tensor as T  # NOQA
        # Build outputs to babysit training
        monitor_outputs = []
        if model.train_config['monitor']:
            for param in parameters:
                # The vector each param was udpated with
                # (one vector per channel)
                param_update_vec = updates[param] - param
                param_update_vec.name = 'param_update_vector_' + param.name
                flat_shape = (param_update_vec.shape[0],
                              T.prod(param_update_vec.shape[1:]))
                flat_param_update_vec = param_update_vec.reshape(flat_shape)
                param_update_mag = (flat_param_update_vec ** 2).sum(-1)
                param_update_mag.name = 'param_update_magnitude_' + param.name
                monitor_outputs.append(param_update_mag)
        return monitor_outputs

    def build_unlabeled_output_expressions(model, network_output):
        """
        override in inherited subclass to enable custom symbolic expressions
        based on the network output alone
        """
        raise NotImplementedError('need override')
        return []

    def build_labeled_output_expressions(model, network_output, y_batch):
        """
        override in inherited subclass to enable custom symbolic expressions
        based on the network output and the labels
        """
        raise NotImplementedError('need override')
        return []


def report_error(msg):
    if False:
        raise ValueError(msg)
    else:
        print('WARNING:' + msg)


@ut.reloadable_class
class BaseModel(_ModelLegacy, _ModelVisualization, _ModelIO, _ModelStrings,
                _ModelIDs, _ModelBackend, _ModelFitting, _ModelUtility,
                ut.NiceRepr):
    """
    Abstract model providing functionality for all other models to derive from
    """
    def __init__(model, **kwargs):
        """
        Guess on Shapes:
            input_shape (tuple): in Theano format (b, c, h, w)
            data_shape (tuple):  in  Numpy format (b, h, w, c)
        """
        kwargs = kwargs.copy()
        if kwargs.pop('verbose_compile', True):
            import logging
            compile_logger = logging.getLogger('theano.compile')
            compile_logger.setLevel(-10)
        model._init_io_vars(kwargs)
        model._init_id_vars(kwargs)
        model._init_shape_vars(kwargs)
        model._init_compile_vars(kwargs)
        model._init_fit_vars(kwargs)
        model.output_layer = None
        assert len(kwargs) == 0, (
            'Model was given unused keywords=%r' % (list(kwargs.keys())))

    def _init_shape_vars(model, kwargs):
        input_shape = kwargs.pop('input_shape', None)
        batch_size = kwargs.pop('batch_size', None)
        data_shape = kwargs.pop('data_shape', None)
        output_dims = kwargs.pop('output_dims', None)

        if input_shape is None and data_shape is None:
            report_error(
                'Must specify either input_shape or data_shape')
        elif input_shape is None:
            input_shape = (batch_size, data_shape[2], data_shape[0],
                           data_shape[1])
        elif data_shape is None and batch_size is None:
            data_shape = (input_shape[2], input_shape[3], input_shape[1])
            batch_size = input_shape[0]
        else:
            report_error(
                'Dont specify batch_size or data_shape with input_shape')

        model.output_dims = output_dims
        model.input_shape = input_shape
        model.data_shape = data_shape
        model.batch_size = batch_size
        # bad name, says that this network will take
        # 2*N images in a batch and N labels that map to
        # two images a piece
        model.data_per_label_input  = 1  # state of network input
        model.data_per_label_output = 1  # state of network output

    # --- OTHER
    @property
    def input_batchsize(model):
        return model.input_shape[0]

    @property
    def input_channels(model):
        return model.input_shape[1]

    @property
    def input_height(model):
        return model.input_shape[2]

    @property
    def input_width(model):
        return model.input_shape[3]

    # --- INITIALIZATION

    def initialize_architecture(model):
        raise NotImplementedError('reimplement')

    def ensure_training_state(model, X_learn, y_learn):
        # TODO: move to dataset. This is independant of the model.
        if model.preproc_kw is None:
            # TODO: move this to data preprocessing, not model preprocessing
            model.preproc_kw = {}
            print('computing center mean.')
            model.preproc_kw['center_mean'] = np.mean(
                X_learn.astype(np.float32), axis=0)
            print('computing center std. (hacks to 255 or 1.0)')
            if ut.is_int(X_learn):
                ut.assert_inbounds(X_learn, 0, 255, eq=True,
                                   verbose=ut.VERBOSE)
                model.preproc_kw['center_std'] = 255.0
            else:
                ut.assert_inbounds(X_learn, 0.0, 1.0, eq=True,
                                   verbose=ut.VERBOSE)
                model.preproc_kw['center_std'] = 1.0
        if getattr(model, 'encoder', None) is None:
            if hasattr(model, 'initialize_encoder'):
                model.initialize_encoder(y_learn)

    def reinit_weights(model, W=None):
        """
        initailizes weights after the architecture has been defined.
        """
        import ibeis_cnn.__LASAGNE__ as lasagne
        if W is None:
            W = 'orthogonal'
        if isinstance(W, six.string_types):
            if W == 'orthogonal':
                W = lasagne.init.Orthogonal()
        print('Reinitializing all weights to %r' % (W,))
        weights_list = model.get_all_params(regularizable=True, trainable=True)
        #print(weights_list)
        for weights in weights_list:
            #print(weights)
            shape = weights.get_value().shape
            new_values = W.sample(shape)
            weights.set_value(new_values)

    # ---- UTILITY

    def fit_dataset(model, X_train, y_train, X_valid, y_valid, dataset, config):
        from ibeis_cnn import harness
        harness.train(model, X_train, y_train, X_valid, y_valid, dataset, config)

    def predict2(model, X_test):
        """ FIXME: turn into a real predict function """
        from ibeis_cnn import batch_processing as batch
        if ut.VERBOSE:
            print('\n[train] --- MODEL INFO ---')
            model.print_architecture_str()
            model.print_layer_info()
        # create theano symbolic expressions that define the network
        theano_predict = model.build_predict_func()
        # Begin testing with the neural network
        print('\n[test] predict with batch size %0.1f' % (
            model.batch_size))
        test_outputs = batch.process_batch(model, X_test, None, theano_predict,
                                           fix_output=True)
        return test_outputs


class AbstractCategoricalModel(BaseModel):
    """ base model for catagory classifiers """

    def __init__(model, **kwargs):
        super(AbstractCategoricalModel, model).__init__(**kwargs)
        model.encoder = None
        # categorical models have a concept of accuracy
        #model.requested_headers += ['valid_acc', 'test_acc']
        model.requested_headers += ['valid_acc']

    def initialize_encoder(model, labels):
        print('[model] encoding labels')
        from sklearn import preprocessing
        model.encoder = preprocessing.LabelEncoder()
        model.encoder.fit(labels)
        model.output_dims = len(list(np.unique(labels)))
        print('[model] model.output_dims = %r' % (model.output_dims,))

    def loss_function(model, network_output, truth):
        # https://en.wikipedia.org/wiki/Loss_functions_for_classification
        from ibeis_cnn.__THEANO__ import tensor as T  # NOQA
        # categorical cross-entropy between predictions and targets
        # L_i = -\sum_{j} t_{i,j} \log{p_{i, j}}
        return T.nnet.categorical_crossentropy(network_output, truth)

    def build_unlabeled_output_expressions(model, network_output):
        from ibeis_cnn.__THEANO__ import tensor as T  # NOQA
        # Network outputs define category probabilities
        probabilities = network_output
        predictions = T.argmax(probabilities, axis=1)
        predictions.name = 'predictions'
        confidences = probabilities.max(axis=1)
        confidences.name = 'confidences'
        unlabeled_outputs = [predictions, confidences]
        return unlabeled_outputs

    def build_labeled_output_expressions(model, network_output, y_batch):
        from ibeis_cnn.__THEANO__ import tensor as T  # NOQA
        probabilities = network_output
        predictions = T.argmax(probabilities, axis=1)
        predictions.name = 'tmp_predictions'
        accuracy = T.mean(T.eq(predictions, y_batch))
        accuracy.name = 'accuracy'
        labeled_outputs = [accuracy]
        return labeled_outputs


class PretrainedNetwork(object):
    """
    Intialize weights from a specified (Caffe) pretrained network layers

    Args:
        layer (int) : int

    CommandLine:
        python -m ibeis_cnn --tf PretrainedNetwork:0
        python -m ibeis_cnn --tf PretrainedNetwork:1

    Example0:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.models import *  # NOQA
        >>> self = PretrainedNetwork('caffenet', show_network=True)
        >>> print('done')

    Example1:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.models import *  # NOQA
        >>> self = PretrainedNetwork('vggnet', show_network=True)
        >>> print('done')
    """
    def __init__(self, modelkey=None, show_network=False):
        from ibeis_cnn._plugin_grabmodels import ensure_model
        self.modelkey = modelkey
        weights_path = ensure_model(modelkey)
        try:
            self.pretrained_weights = ut.load_cPkl(weights_path)
        except Exception:
            raise IOError('The specified model was not found: %r' %
                          (weights_path, ))
        if show_network:
            net_strs.print_pretrained_weights(
                self.pretrained_weights, weights_path)

    def get_num_layers(self):
        return len(self.pretrained_weights)

    def get_layer_num_filters(self, layer_index):
        assert layer_index <= len(self.pretrained_weights), (
            'Trying to specify a layer that does not exist')
        shape = self.pretrained_weights[layer_index].shape
        fanout, fanin, height, width = shape
        return fanout

    def get_layer_filter_size(self, layer_index):
        assert layer_index <= len(self.pretrained_weights), (
            'Trying to specify a layer that does not exist')
        shape = self.pretrained_weights[layer_index].shape
        fanout, fanin, height, width = shape
        return (height, width)

    def get_conv2d_layer(self, layer_index, name=None, **kwargs):
        """ Assumes requested layer is convolutional

        Returns:
            lasange.layers.Layer: Layer
        """
        if name is None:
            name = '%s_layer%r' % (self.modelkey, layer_index)
        W = self.get_pretrained_layer(layer_index)
        try:
            b = self.get_pretrained_layer(layer_index + 1)
            assert W.shape[0] == b.shape[0]
        except:
            b = None
        print(W.shape)
        print(b.shape)
        num_filters = self.get_layer_num_filters(layer_index)
        filter_size = self.get_layer_filter_size(layer_index)

        from ibeis_cnn import custom_layers
        Conv2DLayer = custom_layers.Conv2DLayer
        #MaxPool2DLayer = custom_layers.MaxPool2DLayer

        Layer = functools.partial(
            Conv2DLayer, num_filters=num_filters,
            filter_size=filter_size, W=W, b=b, name=name, **kwargs)
        return Layer

    def get_pretrained_layer(self, layer_index, rand=False):
        import ibeis_cnn.__LASAGNE__ as lasagne
        assert layer_index <= len(self.pretrained_weights), (
            'Trying to specify a layer that does not exist')
        pretrained_layer = self.pretrained_weights[layer_index]

        class _PretrainedLayerInitializer(lasagne.init.Initializer):
            def __init__(self, pretrained_layer):
                self.pretrained_layer = pretrained_layer

            def sample(self, shape):
                if len(shape) == 1:
                    assert shape[0] <= self.pretrained_layer.shape[0]
                    pretrained_weights = self.pretrained_layer[:shape[0]]
                else:
                    is_conv = len(shape) == 4
                    assert len(shape) == len(self.pretrained_layer.shape), (
                        'Layer shape mismatch. Expected %r got %r' % (
                            self.pretrained_layer.shape, shape))
                    fanout, fanin = shape[:2]
                    fanout_, fanin_ = self.pretrained_layer.shape[:2]
                    assert fanout <= fanout_, ('Cannot increase weight fan-out dimension')
                    assert fanin <= fanin_,  ('Cannot increase weight fan-in dimension')
                    if is_conv:
                        height, width = shape[2:]
                        height_, width_ = self.pretrained_layer.shape[2:]
                        assert height == height_, ('Layer height must equal Weight height')
                        assert width == width_,  ('Layer width must equal Weight width')
                    if is_conv:
                        pretrained_weights = self.pretrained_layer[:fanout, :fanin, :, :]
                    else:
                        pretrained_weights = self.pretrained_layer[:fanout, :fanin]
                pretrained_sample = lasagne.utils.floatX(pretrained_weights)
                return pretrained_sample

        weights_initializer = _PretrainedLayerInitializer(pretrained_layer)
        if rand:
            np.random.shuffle(weights_initializer)
        return weights_initializer


def evaluate_layer_list(network_layers_def, verbose=None):
    r"""
    compiles a sequence of partial functions into a network
    """
    if verbose is None:
        verbose = utils.VERBOSE_CNN
    total = len(network_layers_def)
    network_layers = []
    if verbose:
        print('Evaluting List of %d Layers' % (total,))
    layer_fn_iter = iter(network_layers_def)
    #if True:
    #with warnings.catch_warnings():
    #    warnings.filterwarnings(
    #        'ignore', '.*The uniform initializer no longer uses Glorot.*')
    try:
        with ut.Indenter(' ' * 4, enabled=verbose):
            next_args = tuple()
            for count, layer_fn in enumerate(layer_fn_iter, start=1):
                if verbose:
                    print('Evaluating layer %d/%d (%s) ' %
                          (count, total, ut.get_funcname(layer_fn), ))
                with ut.Timer(verbose=False) as tt:
                    layer = layer_fn(*next_args)
                next_args = (layer,)
                network_layers.append(layer)
                if verbose:
                    print('  * took %.4fs' % (tt.toc(),))
                    print('  * layer = %r' % (layer,))
                    if hasattr(layer, 'input_shape'):
                        print('  * layer.input_shape = %r' % (
                            layer.input_shape,))
                    if hasattr(layer, 'shape'):
                        print('  * layer.shape = %r' % (
                            layer.shape,))
                    print('  * layer.output_shape = %r' % (
                        layer.output_shape,))
    except Exception as ex:
        keys = ['layer_fn', 'layer_fn.func', 'layer_fn.args',
                'layer_fn.keywords', 'layer', 'count']
        ut.printex(ex,
                   ('Error buildling layers.\n'
                    'layer.name=%r') % (layer),
                   keys=keys)
        raise
    return network_layers


def overwrite_latest_image(fpath, new_name):
    """
    copies the new image to a path to be overwritten so new updates are
    shown
    """
    from os.path import split, join, splitext, dirname
    import shutil
    dpath, fname = split(fpath)
    ext = splitext(fpath)[1]
    shutil.copy(fpath, join(dpath, 'latest ' + new_name + ext))
    shutil.copy(fpath, join(dirname(dpath), 'latest ' + new_name + ext))


def testdata_model_with_history():
    model = BaseModel()
    # make a dummy history
    X_train, y_train = [1, 2, 3], [0, 0, 1]
    rng = np.random.RandomState(0)
    def dummy_epoch_dict(num):
        epoch_info = {
            'epoch': num,
            'loss': 1 / np.exp(num / 10) + rng.rand() / 100,
            'learn_loss': 1 / np.exp(num / 10) + rng.rand() / 100,
            'learn_loss_regularized': (1 / np.exp(num / 10) +
                                       np.exp(rng.rand() * num) +
                                       rng.rand() / 100),
            'valid_loss': 1 / np.exp(num / 10) - rng.rand() / 100,
            'param_update_mags': {
                'C0': (rng.normal() ** 2, rng.rand()),
                'F1': (rng.normal() ** 2, rng.rand()),
            }
        }
        return epoch_info
    count = 0
    for era_length in [4, 4, 4]:
        model.start_new_era(X_train, y_train, X_train, y_train)
        for count in range(count, count + era_length):
            model.record_epoch(dummy_epoch_dict(count))
    #model.record_epoch({'epoch': 1, 'valid_loss': .8, 'learn_loss': .9})
    #model.record_epoch({'epoch': 2, 'valid_loss': .5, 'learn_loss': .7})
    #model.record_epoch({'epoch': 3, 'valid_loss': .3, 'learn_loss': .6})
    #model.record_epoch({'epoch': 4, 'valid_loss': .2, 'learn_loss': .3})
    #model.record_epoch({'epoch': 5, 'valid_loss': .1, 'learn_loss': .2})
    return model


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis_cnn.abstract_models
        python -m ibeis_cnn.abstract_models --allexamples
        python -m ibeis_cnn.abstract_models --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
