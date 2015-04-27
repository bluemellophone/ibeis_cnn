import numpy as np
import lasagne


class MultiImageSliceLayer(lasagne.layers.Layer):
    """
    orig CyclicSliceLayer
    References:
        https://github.com/benanne/kaggle-ndsb/blob/master/dihedral.py#L89

    This layer stacks rotations of 0, 90, 180, and 270 degrees of the input
    along the batch dimension.
    If the input has shape (batch_size, num_channels, r, c),
    then the output will have shape (4 * batch_size, num_channels, r, c).
    Note that the stacking happens on axis 0, so a reshape to
    (4, batch_size, num_channels, r, c) will separate the slice axis.
    """
    def __init__(self, input_layer):
        super(MultiImageSliceLayer, self).__init__(input_layer)

    def get_output_shape_for(self, input_shape):
        return (4 * input_shape[0],) + input_shape[1:]

    def get_output_for(self, input_, *args, **kwargs):
        return lasagne.utils.concatenate([
            #array_tf_0(input_),
            #array_tf_90(input_),
            #array_tf_180(input_),
            #array_tf_270(input_),
        ], axis=0)


class MultiImageRollLayer(lasagne.layers.Layer):
    """
    orig CyclicConvRollLayer


    This layer turns (n_views * batch_size, num_channels, r, c) into
    (n_views * batch_size, n_views * num_channels, r, c) by rolling
    and concatenating feature maps.
    It also applies the correct inverse transforms to the r and c
    dimensions to align the feature maps.

    References:
        https://github.com/benanne/kaggle-ndsb/blob/master/dihedral.py#L224
    """
    def __init__(self, input_layer):
        super(MultiImageRollLayer, self).__init__(input_layer)
        self.inv_tf_funcs = []  # array_tf_0, array_tf_270, array_tf_180, array_tf_90]
        self.compute_permutation_matrix()

    def compute_permutation_matrix(self):
        map_identity = np.arange(4)
        map_rot90 = np.array([1, 2, 3, 0])

        valid_maps = []
        current_map = map_identity
        for k in range(4):
            valid_maps.append(current_map)
            current_map = current_map[map_rot90]

        self.perm_matrix = np.array(valid_maps)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], 4 * input_shape[1]) + input_shape[2:]

    def get_output_for(self, input_, *args, **kwargs):
        s = input_.shape
        input_unfolded = input_.reshape((4, s[0] // 4, s[1], s[2], s[3]))

        permuted_inputs = []
        for p, inv_tf in zip(self.perm_matrix, self.inv_tf_funcs):
            input_permuted = inv_tf(input_unfolded[p].reshape(s))
            permuted_inputs.append(input_permuted)

        return lasagne.utils.concatenate(permuted_inputs, axis=1)  # concatenate long the channel axis


import theano.tensor as T


class CyclicPoolLayer(lasagne.layers.Layer):
    """
    Utility layer that unfolds the viewpoints dimension and pools over it.
    Note that this only makes sense for dense representations, not for
    feature maps (because no inverse transforms are applied to align them).
    """
    def __init__(self, input_layer, pool_function=T.mean):
        super(CyclicPoolLayer, self).__init__(input_layer)
        self.pool_function = pool_function

    def get_output_shape_for(self, input_shape):
        return (input_shape[0] // 4, input_shape[1])

    def get_output_for(self, input_, *args, **kwargs):
        unfolded_input = input_.reshape((4, input_.shape[0] // 4, input_.shape[1]))
        return self.pool_function(unfolded_input, axis=0)
