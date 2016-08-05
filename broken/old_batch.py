#def create_sliced_iter_funcs_train2(model, X_unshared, y_unshared):
#    """
#    WIP: NEW IMPLEMENTATION WITH PRELOADING GPU DATA

#    build the Theano functions (symbolic expressions) that will be used in the
#    optimization refer to this link for info on tensor types:

#    References:
#        http://deeplearning.net/software/theano/library/tensor/basic.html
#        http://deeplearning.net/software/theano/tutorial/aliasing.html#borrowing-when-creating-shared-variables
#        http://deeplearning.net/tutorial/lenet.html
#        # TODO: Deal with batching to the GPU by setting the value of the shared variables.

#    CommandLine:
#        python -m ibeis_cnn.batch_processing --test-create_sliced_iter_funcs_train2

#    Example:
#        >>> # DISABLE_DOCTEST
#        >>> from ibeis_cnn.batch_processing import *  # NOQA
#        >>> from ibeis_cnn import draw_net
#        >>> from ibeis_cnn import models
#        >>> model = models.DummyModel(autoinit=True)
#        >>> X_unshared, y_unshared = model.make_random_testdata()
#        >>> train_iter = model._build_theano_funcs(model)
#        >>> print(train_iter)
#        >>> loss_train, newtork_output, prediction, accuracy = train_iter(0)
#        >>> print('loss = %r' % (loss,))
#        >>> print('net_out = %r' % (outvec,))
#        >>> print('newtork_output = %r' % (newtork_output,))
#        >>> print('accuracy = %r' % (accuracy,))
#        >>> #draw_net.draw_theano_symbolic_expression(train_iter)
#        >>> assert outvec.shape == (model.batch_size, model.output_dims)
#    """
#    # Attempt to load data on to the GPU
#    # Labels to go into the GPU as float32 and then cast to int32 once inside
#    X_unshared = np.asarray(X_unshared, dtype=theano.config.floatX)
#    y_unshared = np.asarray(y_unshared, dtype=theano.config.floatX)

#    X_shared = theano.shared(X_unshared, borrow=True)
#    y_shared = T.cast(theano.shared(y_unshared, borrow=True), 'int32')

#    # Build expressions which sample a batch
#    batch_size = model.batch_size

#    # Initialize symbolic input variables
#    index = T.lscalar(name='index')
#    X_batch = T.tensor4(name='X_batch')
#    y_batch = T.ivector(name='y_batch')

#    WHITEN = False
#    if WHITEN:
#        # We might be able to perform some data augmentation here symbolicly
#        data_mean = X_unshared.mean()
#        data_std = X_unshared.std()
#        givens = {
#            X_batch: (X_shared[index * batch_size: (index + 1) * batch_size] - data_mean) / data_std,
#            y_batch: y_shared[index * batch_size: (index + 1) * batch_size],
#        }
#    else:
#        givens = {
#            X_batch: X_shared[index * batch_size: (index + 1) * batch_size],
#            y_batch: y_shared[index * batch_size: (index + 1) * batch_size],
#        }

#    output_layer = model.get_output_layer()

#    # Build expression to evalute network output without dropout
#    #newtork_output = output_layer.get_output(X_batch, deterministic=True)
#    newtork_output = layers.get_output(output_layer, X_batch, deterministic=True)
#    newtork_output.name = 'network_output'

#    # Build expression to evaluate loss
#    objective = objectives.Objective(output_layer, loss_function=model.loss_function)
#    loss_train = objective.get_loss(X_batch, target=y_batch)  # + 0.0001 * lasagne.regularization.l2(output_layer)
#    loss_train.name = 'loss_train'

#    # Build expression to evaluate updates
#    with warnings.catch_warnings():
#        warnings.filterwarnings('ignore', '.*topo.*')
#        all_params = lasagne.layers.get_all_params(output_layer, trainable=True)
#    updates = lasagne.updates.nesterov_momentum(loss_train, all_params, model.learning_rate, model.momentum)

#    # Get performance indicator outputs:

#    # Build expression to convert network output into a prediction
#    prediction = model.make_prediction_expr(newtork_output)

#    # Build expression to compute accuracy
#    accuracy = model.make_accuracy_expr(prediction, y_batch)

#    theano_backprop = theano.function(
#        inputs=[index],
#        outputs=[loss_train, newtork_output, prediction, accuracy],
#        updates=updates,
#        givens=givens
#    )
#    theano_backprop.name += ':theano_backprob:indexed'

#    #other_outputs = [probabilities, predictions, confidences]

#    #theano_backprop = theano.function(
#    #    inputs=[theano.Param(X_batch), theano.Param(y_batch)],
#    #    outputs=[loss] + other_outputs,
#    #    updates=updates,
#    #    givens={
#    #        X: X_batch,
#    #        y: y_batch,
#    #    },
#    #)

#    #theano_forward = theano.function(
#    #    inputs=[theano.Param(X_batch), theano.Param(y_batch)],
#    #    outputs=[loss_determ] + other_outputs,
#    #    updates=None,
#    #    givens={
#    #        X: X_batch,
#    #        y: y_batch,
#    #    },
#    #)

#    #theano_predict = theano.function(
#    #    inputs=[theano.Param(X_batch)],
#    #    outputs=other_outputs,
#    #    updates=None,
#    #    givens={
#    #        X: X_batch,
#    #    },
#    #)

#    return theano_backprop


#def create_sliced_network_output_func(model):
#    # Initialize symbolic input variables
#    X_batch = T.tensor4(name='X_batch')
#    # weird, idk why X and y exist
#    X = T.tensor4(name='X_batch')

#    output_layer = model.get_output_layer()

#    # Build expression to evalute network output without dropout
#    #newtork_output = output_layer.get_output(X_batch, deterministic=True)
#    newtork_output = layers.get_output(output_layer, X_batch, deterministic=True)
#    newtork_output.name = 'network_output'

#    theano_forward = theano.function(
#        inputs=[theano.Param(X_batch)],
#        outputs=[newtork_output],
#        givens={
#            X: X_batch,
#        }
#    )
#    theano_forward.name += ':theano_forward:sliced'
#    return theano_forward
