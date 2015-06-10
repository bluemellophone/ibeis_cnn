"""
Core functions for data augmentation


References:
    https://github.com/benanne/kaggle-ndsb/blob/master/data.py
"""
from __future__ import absolute_import, division, print_function
import functools
import numpy as np
import utool as ut
print, rrr, profile = ut.inject2(__name__, '[ibeis_cnn.augment]')


rot_transforms  = [functools.partial(np.rot90, k=k) for k in range(1, 4)]

flip_transforms = [np.fliplr, np.flipud]

all_transforms = [
    rot_transforms,
    flip_transforms,
]


default_augmentation_params = {
    'zoom_range': (1 / 1.1, 1.1),
    'rotation_range': (0, 360),
    'shear_range': (0, 0),
    'translation_range': (-4, 4),
    'do_flip': True,
    'allow_stretch': False,
}


TAU = 2 * np.pi


def random_affine_args(zoom_range=(1 / 1.1, 1.1),
                       max_tx=1.0,
                       max_ty=1.0,
                       max_shear=TAU / 16,
                       max_theta=TAU / 32,
                       enable_flip=False,
                       enable_stretch=False,
                       rng=np.random):
    r"""
    CommandLine:
        python -m ibeis_cnn.augment --test-random_affine_args

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_cnn.augment import *  # NOQA
        >>> zoom_range = (0.9090909090909091, 1.1)
        >>> max_tx = (-4, 4)
        >>> max_ty = (-4, 4)
        >>> max_shear = np.pi
        >>> enable_rotate = True
        >>> enable_flip = True
        >>> enable_stretch = True
        >>> rng = np.random.RandomState(0)
        >>> affine_args = random_affine_args(zoom_range, max_tx, max_ty, max_shear, enable_rotate, enable_flip, enable_stretch, rng)
        >>> sx, sy, theta, shear, tx, ty = affine_args
        >>> Aff = vt.affine_mat3x3(sx, sy, theta, shear, tx, ty)
        >>> result = ut.numpy_str2(Aff)
        >>> print(result)
        np.array([[ 0.972,  0.566,  0.359],
                  [ 0.308,  0.848, -0.611],
                  [ 0.   ,  0.   ,  1.   ]])
    """

    if zoom_range is None:
        sx = sy = 1.0
    else:
        log_zoom_range = [np.log(z) for z in zoom_range]

        if enable_stretch:
            sx = sy = np.exp(rng.uniform(*log_zoom_range))
        else:
            sx = np.exp(rng.uniform(*log_zoom_range))
            sy = np.exp(rng.uniform(*log_zoom_range))

    theta = 0.0 if max_theta is None else rng.uniform(-max_theta, max_theta)
    shear = 0.0 if max_shear is None else rng.uniform(-max_shear, max_shear)
    tx = 0.0 if max_tx is None else rng.uniform(-max_tx, max_tx)
    ty = 0.0 if max_ty is None else rng.uniform(-max_ty, max_ty)

    flip = enable_flip and (rng.randint(2) > 0)  # flip half of the time
    if flip:
        # shear 180 degrees + rotate 180 == flip
        theta += np.pi
        shear += np.pi

    affine_args = (sx, sy, theta, shear, tx, ty)
    return affine_args
    #Aff = vt.affine_mat3x3(sx, sy, theta, shear, tx, ty)
    #return Aff


def random_affine_kwargs(*args, **kwargs):
    affine_args = random_affine_args(*args, **kwargs)
    affine_keys = ['sx', 'sy', 'theta', 'shear', 'tx', 'ty']
    affine_kw = dict(zip(affine_keys, affine_args))
    return affine_kw


def affine_perterb(img, rng=np.random):
    r"""
    Args:
        img (ndarray[uint8_t, ndim=2]):  image data
        rng (module):

    Returns:
        ndarray: img_warped

    CommandLine:
        python -m ibeis_cnn.augment --test-affine_perterb --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_cnn.augment import *  # NOQA
        >>> import vtool as vt
        >>> img_fpath = ut.grab_test_imgpath('carl.jpg')
        >>> img = vt.imread(img_fpath)
        >>> rng = np.random   #.RandomState(0)
        >>> img_warped = affine_perterb(img, rng)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> pt.imshow(img_warped)
        >>> ut.show_if_requested()
    """
    import vtool as vt
    import cv2
    affine_args = random_affine_args(rng=rng)
    h1, w1 = img.shape[0:2]
    y1, x1 = h1 / 2.0, w1 / 2.0
    Aff = vt.affine_around_mat3x3(x1, y1, *affine_args)
    dsize = (w1, h1)
    img_warped = cv2.warpAffine(img, Aff[0:2], dsize, flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
    return img_warped


def test_transforms():
    r"""
    CommandLine:
        python -m ibeis_cnn.augment --test-test_transforms --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_cnn.augment import *  # NOQA
        >>> test_transforms()
    """
    from ibeis_cnn import ingest_data, utils, draw_results
    data, labels = ingest_data.testdata_patchmatch()
    cv2_data = utils.convert_theano_images_to_cv2_images(data)
    patches_ = cv2_data[::2]

    transform_list = ut.flatten(all_transforms)

    num_random = 5
    import vtool as vt
    for x in range(num_random):
        affine_kw = random_affine_kwargs()
        func = functools.partial(vt.affine_warp_around_center, **affine_kw)
        transform_list.append(func)

    orig_list   = []
    warped_list = []

    name_list = []

    for patch, func in zip(patches_, transform_list):
        if isinstance(func, functools.partial):
            name = ut.get_partial_func_name(func)
        else:
            name = ut.get_funcname(func)
        print(name)
        warped = func(patch)
        orig_list.append(patch)
        name_list.append(name)
        warped_list.append(warped)

    index_list = list(range(len(orig_list)))
    label_list = None
    tup = draw_results.get_patch_sample_img(orig_list, warped_list, label_list, {'text': name_list}, index_list, (1, len(index_list)))
    stacked_img, stacked_offsets, stacked_sfs = tup
    ut.quit_if_noshow()
    import plottool as pt
    pt.imshow(stacked_img)
    ut.show_if_requested()


def augment_siamese_patches(Xb, yb=None, rng=np.random):
    """
    CommandLine:
        python -m ibeis_cnn.augment --test-augment_siamese_patches --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_cnn.augment import *  # NOQA
        >>> from ibeis_cnn import ingest_data, utils, draw_results
        >>> data, labels = ingest_data.testdata_patchmatch()
        >>> cv2_data = utils.convert_theano_images_to_cv2_images(data)
        >>> batch_size = 128
        >>> Xb, yb = cv2_data[0:batch_size], labels[0:batch_size // 2]
        >>> Xb1, yb1 = augment_siamese_patches(Xb.copy(), yb.copy())
        >>> modified_indexes = np.where((Xb1 != Xb).sum(-1).sum(-1).sum(-1) > 0)[0]
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> #z = draw_results.get_sample_pairimg_from_X(Xb, 1)
        >>> pt.imshow(Xb[modified_indexes[0]], pnum=(2, 2, 1), title='before')
        >>> pt.imshow(Xb1[modified_indexes[0]], pnum=(2, 2, 2), title='after')
        >>> pt.imshow(Xb[modified_indexes[1]], pnum=(2, 2, 3))
        >>> pt.imshow(Xb1[modified_indexes[1]], pnum=(2, 2, 4))
        >>> ut.show_if_requested()
    """
    # Rotate corresponding patches together
    Xb1, Xb2 = Xb[::2], Xb[1::2]
    rot_transforms  = [functools.partial(np.rot90, k=k) for k in range(1, 4)]
    flip_transforms = [np.fliplr, np.flipud]
    prob_rotate = .3
    prob_flip   = .3

    num = len(Xb1)

    # Determine which examples will be augmented
    rotate_flags   = rng.uniform(0.0, 1.0, size=num) <= prob_rotate
    flip_flags     = rng.uniform(0.0, 1.0, size=num) <= prob_flip

    # Determine which functions to use
    rot_fn_list  = [rot_transforms[rng.randint(len(rot_transforms))]
                    if flag else None for flag in rotate_flags]
    flip_fn_list = [flip_transforms[rng.randint(len(flip_transforms))]
                    if flag else None for flag in flip_flags]

    for index, func_list in enumerate(zip(rot_fn_list, flip_fn_list)):
        for func in func_list:
            if func is not None:
                Xb1[index] = func(Xb1[index])
                Xb2[index] = func(Xb2[index])
    return Xb, yb


def stacked_img_pairs(Xb, modified_indexes, label_list=None, num=None):
    from ibeis_cnn import draw_results
    if num is None:
        num = len(modified_indexes)
    num = min(len(modified_indexes), num)
    patch_list1 = Xb[0::2]
    patch_list2 = Xb[1::2]
    per_row = 8
    cols = int(num / per_row)
    #print('len(patch_list1) = %r' % (len(patch_list1),))
    #print('len(patch_list2) = %r' % (len(patch_list2),))
    #print('len(modified_indexes) = %r' % (len(modified_indexes),))
    #print('modified_indexes = %r' % ((modified_indexes),))
    tup = draw_results.get_patch_sample_img(patch_list1, patch_list2, label_list, {}, modified_indexes, (cols, per_row))
    stacked_img, stacked_offsets, stacked_sfs = tup
    return stacked_img
    pass


def augment_siamese_patches2(Xb, yb=None, rng=np.random):
    """
    CommandLine:
        python -m ibeis_cnn.augment --test-augment_siamese_patches2 --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_cnn.augment import *  # NOQA
        >>> from ibeis_cnn import ingest_data, utils, draw_results
        >>> data, labels = ingest_data.testdata_patchmatch()
        >>> cv2_data = utils.convert_theano_images_to_cv2_images(data)
        >>> batch_size = 128
        >>> Xb, yb = cv2_data[0:batch_size], labels[0:batch_size // 2]
        >>> rng = np.random.RandomState(0)
        >>> Xb1, yb1 = augment_siamese_patches2(Xb.copy(), yb.copy(), rng=rng)
        >>> modified_indexes = np.where((Xb1[::2] != Xb[::2]).sum(-1).sum(-1).sum(-1) > 0)[0]
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> orig_stack = stacked_img_pairs(Xb, modified_indexes, None)
        >>> warp_stack = stacked_img_pairs(Xb1, modified_indexes, None)
        >>> pt.imshow(orig_stack, pnum=(1, 2, 1), title='before')
        >>> pt.imshow(warp_stack, pnum=(1, 2, 2), title='after')
        >>> ut.show_if_requested()
    """
    import vtool as vt
    # Rotate corresponding patches together
    Xb1, Xb2 = Xb[::2], Xb[1::2]
    prob_perterb = .7

    num = len(Xb1)

    # Determine which examples will be augmented
    perterb_flags = rng.uniform(0.0, 1.0, size=num) <= prob_perterb

    perterb_ranges = dict(
        zoom_range=None,
        max_tx=None,
        max_ty=None,
        max_shear=None,
        max_theta=None,
        enable_flip=False,
        enable_stretch=False,
    )
    perterb_ranges.update(
        dict(
            #zoom_range=(1 / 1.1, 1.1),
            max_tx=5,
            max_ty=5,
            #max_shear=TAU / 16,
            max_shear=TAU / 32,
            #max_theta=TAU / 32,
            #max_theta=TAU / 32,
            max_theta=TAU,
            enable_flip=True,
        )
    )

    index_list = np.where(perterb_flags)[0]

    perterb_kw_list = [
        random_affine_kwargs(rng=rng, **perterb_ranges)
        for index in index_list
    ]

    import cv2
    borderMode = cv2.BORDER_REFLECT101
    borderMode = cv2.BORDER_REFLECT_101
    borderMode = cv2.BORDER_WRAP
    borderMode = cv2.BORDER_CONSTANT
    borderMode = cv2.BORDER_REFLECT
    #borderMode = cv2.BORDER_REPLICATE
    flags = cv2.INTER_LANCZOS4

    for index, kw in zip(index_list, perterb_kw_list):
        Xb1[index] = vt.affine_warp_around_center(Xb1[index], borderMode=borderMode, flags=flags, **kw)
        Xb2[index] = vt.affine_warp_around_center(Xb2[index], borderMode=borderMode, flags=flags, **kw)
    return Xb, yb


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis_cnn.augment
        python -m ibeis_cnn.augment --allexamples
        python -m ibeis_cnn.augment --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()