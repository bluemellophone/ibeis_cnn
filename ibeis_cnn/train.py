#!/usr/bin/env python
"""
train.py
constructs the Theano optimization and trains a learning model,
optionally by initializing the network with pre-trained weights.
"""
from __future__ import absolute_import, division, print_function
from ibeis_cnn import utils
from ibeis_cnn import models

import time
import theano
import numpy as np
import cPickle as pickle
from lasagne import layers
from sklearn import preprocessing

import utool as ut
from os.path import join, abspath


def train(data_file, labels_file, model, weights_file, pretrained_weights_file=None,
          pretrained_kwargs=False, **kwargs):
    """
    Driver function

    Args:
        data_file (?):
        labels_file (?):
        model (?):
        weights_file (?):
        pretrained_weights_file (None):
        pretrained_kwargs (bool):

    CommandLine:
        python -m ibeis_cnn.train --test-train

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.train import *  # NOQA
        >>> # build test data
        >>> data_file = '?'
        >>> labels_file = '?'
        >>> model = '?'
        >>> weights_file = '?'
        >>> pretrained_weights_file = None
        >>> pretrained_kwargs = False
        >>> kwargs = {}
        >>> # execute function
        >>> result = train(data_file, labels_file, model, weights_file, pretrained_weights_file, pretrained_kwargs)
        >>> # verify results
        >>> print(result)
    """

    ######################################################################################

    # Load the data
    print('\n[data] loading data...')
    data, labels = utils.load(data_file, labels_file)
    train_(data, labels, model, weights_file, pretrained_weights_file=None, pretrained_kwargs=False, **kwargs)


def train_(data, labels, model, weights_file, pretrained_weights_file=None, pretrained_kwargs=False, **kwargs):
    r"""
    Args:
        data (?):
        labels (?):
        model (?):
        weights_file (?):
        pretrained_weights_file (None):
        pretrained_kwargs (bool):

    CommandLine:
        python -m ibeis_cnn.train --test-train_

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.train import *  # NOQA
        >>> # build test data
        >>> data = '?'
        >>> labels = '?'
        >>> model = '?'
        >>> weights_file = '?'
        >>> pretrained_weights_file = None
        >>> pretrained_kwargs = False
        >>> kwargs = {}
        >>> # execute function
        >>> result = train_(data, labels, model, weights_file, pretrained_weights_file, pretrained_kwargs)
        >>> # verify results
        >>> print(result)
    """
    # Training parameters defaults
    utils._update(kwargs, 'center',         True)
    utils._update(kwargs, 'encode',         True)
    utils._update(kwargs, 'learning_rate',  0.0003)
    utils._update(kwargs, 'momentum',       0.9)
    utils._update(kwargs, 'batch_size',     128)
    utils._update(kwargs, 'patience',       10)
    utils._update(kwargs, 'test',           5)  # Test every X epochs
    utils._update(kwargs, 'max_epochs',     kwargs.get('patience') * 10)
    # utils._update(kwargs, 'regularization', None)
    utils._update(kwargs, 'regularization', 0.001)
    utils._update(kwargs, 'output_dims',    None)

    # Automatically figure out how many classes
    if kwargs.get('output_dims') is None:
        kwargs['output_dims'] = len(list(np.unique(labels)))

    # print('[load] adding channels...')
    # data = utils.add_channels(data)
    print('[train]     data.shape = %r' % (data.shape,))
    print('[train]     data.dtype = %r' % (data.dtype,))
    print('[train]     labels.shape = %r' % (labels.shape,))
    print('[train]     labels.dtype = %r' % (labels.dtype,))

    import utool as ut
    import six

    labelhist = {key: len(val) for key, val in six.iteritems(ut.group_items(labels, labels))}
    print('label stats = \n' + ut.dict_str(labelhist))
    print('train kwargs = \n' + (ut.dict_str(kwargs)))

    # Encoding labels
    kwargs['encoder'] = None
    if kwargs.get('encode', False):
        kwargs['encoder'] = preprocessing.LabelEncoder()
        kwargs['encoder'].fit(labels)

    # utils.show_image_from_data(data)

    # Split the dataset into training and validation
    print('[train] creating train, validation datasaets...')
    dataset = utils.train_test_split(data, labels, eval_size=0.2)
    X_train, y_train, X_valid, y_valid = dataset
    dataset = utils.train_test_split(X_train, y_train, eval_size=0.1)
    X_train, y_train, X_test, y_test = dataset

    # Build and print the model
    print('\n[model] building model...')
    input_cases, input_channels, input_height, input_width = data.shape
    output_layer = model.build_model(
        kwargs.get('batch_size'), input_width, input_height,
        input_channels, kwargs.get('output_dims'))
    utils.print_layer_info(output_layer)

    # Create the Theano primitives
    print('[model] creating Theano primitives...')
    learning_rate_theano = theano.shared(utils.float32(kwargs.get('learning_rate')))
    all_iters = utils.create_iter_funcs(learning_rate_theano, output_layer, **kwargs)
    train_iter, valid_iter, test_iter = all_iters

    # Load the pretrained model if specified
    if pretrained_weights_file is not None:
        print('[model] loading pretrained weights from %s' % (pretrained_weights_file))
        with open(pretrained_weights_file, 'rb') as pfile:
            kwargs_ = pickle.load(pfile)
            pretrained_weights = kwargs_.get('best_weights', None)
            layers.set_all_param_values(output_layer, pretrained_weights)
            if pretrained_kwargs:
                kwargs = kwargs_

    # Center the data by subtracting the mean (AFTER KWARGS UPDATE)
    if kwargs.get('center'):
        print('[train] applying data centering...')
        utils._update(kwargs, 'center_mean', np.mean(X_train, axis=0))
        # utils._update(kwargs, 'center_std', np.std(X_train, axis=0))
        utils._update(kwargs, 'center_std', 255.0)
    else:
        utils._update(kwargs, 'center_mean', 0.0)
        utils._update(kwargs, 'center_std', 255.0)

    # Begin training the neural network
    vals = (utils.get_current_time(), kwargs.get('learning_rate'), )
    print('\n[train] starting training at %s with learning rate %.9f' % vals)
    utils.print_header_columns()

    utils._update(kwargs, 'best_weights',        None)
    utils._update(kwargs, 'best_valid_accuracy', 0.0)
    utils._update(kwargs, 'best_test_accuracy',  0.0)
    utils._update(kwargs, 'best_train_loss',     np.inf)
    utils._update(kwargs, 'best_valid_loss',     np.inf)
    utils._update(kwargs, 'best_valid_accuracy', 0.0)
    utils._update(kwargs, 'best_test_accuracy',  0.0)
    try:
        epoch = 0
        epoch_marker = epoch
        while True:
            try:
                # Start timer
                t0 = time.time()

                # compute the loss over all training and validation batches
                augment_fn = getattr(model, 'augment', None)
                avg_train_loss = utils.forward_train(X_train, y_train, train_iter, rand=True,
                                                     augment=augment_fn, **kwargs)
                avg_valid_data = utils.forward_valid(X_valid, y_valid, valid_iter, **kwargs)
                avg_valid_loss, avg_valid_accuracy = avg_valid_data

                # If the training loss is nan, the training has diverged
                if np.isnan(avg_train_loss):
                    print('\n[train] training diverged\n')
                    break

                # Is this model the best we've ever seen?
                best_found = avg_valid_loss < kwargs.get('best_valid_loss')
                if best_found:
                    kwargs['best_epoch'] = epoch
                    epoch_marker = epoch
                    kwargs['best_weights'] = layers.get_all_param_values(output_layer)

                # compute the loss over all testing batches
                request_test = kwargs.get('test') is not None and epoch % kwargs.get('test') == 0
                if best_found or request_test:
                    avg_test_accuracy = utils.forward_test(X_test, y_test, test_iter, show=True, **kwargs)
                else:
                    avg_test_accuracy = None

                # Running tab for what the best model
                if avg_train_loss < kwargs.get('best_train_loss'):
                    kwargs['best_train_loss'] = avg_train_loss
                if avg_valid_loss < kwargs.get('best_valid_loss'):
                    kwargs['best_valid_loss'] = avg_valid_loss
                if avg_valid_accuracy > kwargs.get('best_valid_accuracy'):
                    kwargs['best_valid_accuracy'] = avg_valid_accuracy
                if avg_test_accuracy > kwargs.get('best_test_accuracy'):
                    kwargs['best_test_accuracy'] = avg_test_accuracy

                # Learning rate schedule update
                if epoch >= epoch_marker + kwargs.get('patience'):
                    epoch_marker = epoch
                    utils.set_learning_rate(learning_rate_theano, model.learning_rate_update)

                # End timer
                t1 = time.time()

                # Increment the epoch
                epoch += 1
                utils.print_epoch_info(avg_train_loss, avg_valid_loss,
                                       avg_valid_accuracy, avg_test_accuracy,
                                       epoch, t1 - t0, **kwargs)

                # Break on max epochs
                if epoch >= kwargs.get('max_epochs'):
                    print('\n[train] maximum number of epochs reached\n')
                    break
            except KeyboardInterrupt:
                # We have caught the Keyboard Interrupt, figure out what resolution mode
                print('\n[train] Caught CRTL+C')
                resolution = ''
                while not (resolution.isdigit() and int(resolution) in [1, 2, 3]):
                    print('\n[train] What do you want to do?')
                    print('[train]     1 - Shock weights')
                    print('[train]     2 - Save best weights')
                    print('[train]     3 - Stop network training')
                    resolution = raw_input('[train] Resolution: ')
                resolution = int(resolution)
                # We have a resolution
                if resolution == 1:
                    # Shock the weights of the network
                    utils.shock_network(output_layer)
                    epoch_marker = epoch
                    utils.set_learning_rate(learning_rate_theano, model.learning_rate_shock)
                elif resolution == 2:
                    # Save the weights of the network
                    utils.save_model(kwargs, weights_file)
                else:
                    # Terminate the network training
                    raise KeyboardInterrupt
    except KeyboardInterrupt:
        print('\n\n[train] ...stopping network training\n')

    # Save the best network
    utils.save_model(kwargs, weights_file)


#@ibeis.register_plugin()
def get_identification_decision_training_data(ibs):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object

    CommandLine:
        python -m ibeis_cnn.train --test-get_identification_decision_training_data

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_cnn.train import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> # execute function
        >>> result = get_identification_decision_training_data(ibs)
        >>> # verify results
        >>> print(result)
    """
    print('get_identification_decision_training_data')
    # Grab marked hard cases
    #am_rowids = ibs._get_all_annotmatch_rowids()
    # The verified set
    #verified_aid1_list = ibs.get_annotmatch_aid1(am_rowids)
    #verified_aid2_list = ibs.get_annotmatch_aid1(am_rowids)
    # The nonverified set

    def get_test_aid_pairs():
        aid_list = ibs.get_valid_aids()
        import utool as ut
        aid_list = ut.list_compress(aid_list, ibs.get_annot_has_groundtruth(aid_list))
        qres_list = ibs.query_chips(aid_list, aid_list)

        num = 3
        aid1_list = np.array(ut.flatten([[qres.qaid] * num for qres in qres_list]))
        aid2_list = np.array(ut.flatten([qres.get_top_aids()[0:num] for qres in qres_list]))

        nid1_list = ibs.get_annot_nids(aid1_list)
        nid2_list = ibs.get_annot_nids(aid2_list)

        nid1_list = np.array(nid1_list)
        nid2_list = np.array(nid2_list)
        truth_list = nid1_list == nid2_list
        return aid1_list, aid2_list, truth_list

    aid1_list, aid2_list, truth_list = get_test_aid_pairs()

    def get_aidpair_training_data(aid1_list, aid2_list):
        #ibs.get_annot_pair_truth(aid1_list, aid2_list)
        chip1_list = ibs.get_annot_chips(aid1_list)
        chip2_list = ibs.get_annot_chips(aid2_list)
        import vtool as vt
        sizes1 = np.array([vt.get_size(chip1) for chip1 in chip1_list])
        sizes2 = np.array([vt.get_size(chip2) for chip2 in chip2_list])
        ar1_list = sizes1.T[0] / sizes1.T[1]
        ar2_list = sizes2.T[0] / sizes2.T[1]
        ave_ar = np.hstack((ar1_list, ar2_list)).mean()
        target_height = 64
        target_size = (np.round(ave_ar * target_height), target_height)
        target_size = (32, 32 * 2)
        #np.round(ave_ar * target_height), target_height)
        #dsize =
        thumb1_list = [vt.padded_resize(chip1, target_size)
                        for chip1 in chip1_list]
        thumb2_list = [vt.padded_resize(chip2, target_size)
                        for chip2 in chip2_list]

        # Stacking these might not be the exact correct thing to do.
        img_list = [
            np.hstack((thumb1, thumb2)) for thumb1, thumb2, in
            zip(thumb1_list, thumb2_list)
        ]
        def convert_imagelist_to_data(img_list):
            """
            Args:
                img_list (list of ndarrays): in the format [h, w, c]

            Returns:
                data: in the format [b, c, h, w]
            """
            #[img.shape for img in img_list]
            # format to [b, c, h, w]
            theano_style_imgs = [np.transpose(img, (2, 0, 1))[None, :] for img in img_list]
            data = np.vstack(theano_style_imgs)
            #data = np.vstack([img[None, :] for img in img_list])
            return data

        data = convert_imagelist_to_data(img_list)
        return data

    data = get_aidpair_training_data(aid1_list, aid2_list)
    model                   = models.IdentificationModel()
    #config                  = {}
    #def train_from_files():
    #root                    = abspath(join('..', 'data'))
    root = ut.unixjoin(ibs.get_cachedir(), 'nets')
    ut.ensuredir(root)
    #train_data_file         = join(root, 'numpy', 'id', 'X.npy')
    #train_labels_file       = join(root, 'numpy', 'id', 'y.npy')
    weights_file            = join(root, 'ibeis_cnn_weights.pickle')
    pretrained_weights_file = join(root,  'pretrained_weights.pickle')  # NOQA

    #labels_ = [1 if truth is True else (0 if truth is False else 2) for truth in truth_list]
    labels = np.array(truth_list)
    train_(data, labels, model, weights_file, batch_size=8)
    #X = k


def train_pz():
    r"""
    CommandLine:
        python -m ibeis_cnn.train --test-train_pz

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_cnn.train import *  # NOQA
        >>> # build test data
        >>> # execute function
        >>> result = train_pz()
        >>> # verify results
        >>> print(result)
    """
    project_name            = 'viewpoint'
    model                   = models.PZ_GIRM_Model()
    # project_name            = 'plains'
    # model                   = models.PZ_Model()
    config                  = {}
    #def train_from_files():
    root                    = abspath(join('..', 'data'))
    train_data_file         = join(root, 'numpy', project_name, 'X.npy')
    train_labels_file       = join(root, 'numpy', project_name, 'y.npy')
    weights_file            = join(root, 'nets', 'ibeis_cnn_weights.pickle')
    pretrained_weights_file = join(root, 'nets', 'pretrained_weights.pickle')  # NOQA

    train(train_data_file, train_labels_file, model, weights_file, **config)
    #train(train_data_file, train_labels_file, weights_file, pretrained_weights_file)


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis_cnn.train
        python -m ibeis_cnn.train --allexamples
        python -m ibeis_cnn.train --allexamples --noface --nosrc

    CommandLine:
        cd %CODE_DIR%/ibies_cnn/code
        cd $CODE_DIR/ibies_cnn/code
        code
        cd ibeis_cnn/code
        python train.py

    PythonPrereqs:
        pip install theano
        pip install git+https://github.com/Lasagne/Lasagne.git
        pip install git+git://github.com/lisa-lab/pylearn2.git
        #pip install lasagne
        #pip install pylearn2
        git clone git://github.com/lisa-lab/pylearn2.git
        git clone https://github.com/Lasagne/Lasagne.git
        cd pylearn2
        python setup.py develop
        cd ..
        cd Lesagne
        git checkout 8758ac1434175159e5c1f30123041799c2b6098a
        python setup.py develop
        """
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
    #if __name__ == '__main__':
    #    """
    #    train_pz()
    pass

if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis_cnn.train
        python -m ibeis_cnn.train --allexamples
        python -m ibeis_cnn.train --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
