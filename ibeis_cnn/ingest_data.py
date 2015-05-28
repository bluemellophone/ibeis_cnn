from __future__ import absolute_import, division, print_function
from ibeis_cnn import utils
from os.path import join
import utool as ut
print, rrr, profile = ut.inject2(__name__, '[ibeis_cnn.ingest]')


def open_mnist_files(labels_fpath, data_fpath):
    """
    References:
        http://g.sweyla.com/blog/2012/mnist-numpy/
    """
    import struct
    #import os
    import numpy as np
    from array import array as pyarray
    with open(labels_fpath, 'rb') as flbl:
        magic_nr, size = struct.unpack(">II", flbl.read(8))
        lbl = pyarray("b", flbl.read())

    with open(data_fpath, 'rb') as fimg:
        magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = pyarray("B", fimg.read())
    digits = np.arange(10)

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = np.zeros((N, rows, cols), dtype=np.uint8)
    labels = np.zeros((N, 1), dtype=np.uint8)
    for i in range(len(ind)):
        images[i] = np.array(img[ ind[i] * rows * cols : (ind[i] + 1) * rows * cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]
    return images, labels


def grab_cached_mist_data(nets_dir):
    import numpy as np
    data_fpath = join(nets_dir, 'mnist_data.cPkl')
    labels_fpath = join(nets_dir, 'mnist_labels.cPkl')
    if not ut.checkpath(data_fpath):
        train_imgs_fpath = ut.grab_zipped_url('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz')
        train_lbls_fpath = ut.grab_zipped_url('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz')
        test_imgs_fpath = ut.grab_zipped_url('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz')
        test_lbls_fpath = ut.grab_zipped_url('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz')

        train_images, train_labels = open_mnist_files(train_lbls_fpath, train_imgs_fpath)
        test_images, test_labels = open_mnist_files(test_lbls_fpath, test_imgs_fpath)
        data = np.vstack((train_images, test_images))
        labels = np.append(train_labels, test_labels)
        utils.write_data_and_labels(data, labels, data_fpath, labels_fpath)
    return data_fpath, labels_fpath
