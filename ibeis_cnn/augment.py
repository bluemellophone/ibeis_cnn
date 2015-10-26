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
    #np.random.shuffle(modified_indexes)
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


def show_augmented_patches(Xb_orig, Xb, yb_orig, yb, shadows=None):
    """
    from ibeis_cnn.augment import *  # NOQA
    std_ = center_std
    mean_ = center_mean
    """
    import plottool as pt
    import vtool as vt
    Xb_orig = vt.rectify_to_float01(Xb_orig)
    Xb_ = vt.rectify_to_float01(Xb)

    #num_examples = len(Xb_orig) // 2
    # only look at ones that were actually augmented
    diff1 = np.abs((Xb_[0::2] - Xb_orig[0::2]))
    diff2 = np.abs((Xb_[1::2] - Xb_orig[1::2]))
    diff_batches1 = diff1.sum(-1).sum(-1).sum(-1)
    diff_batches2 = diff2.sum(-1).sum(-1).sum(-1)
    diff_batches = diff_batches1 + diff_batches2
    modified_indexes = np.where(diff_batches > 0)[0]
    #import vtool as vt                       nnnnnnnnnnnnnnnnnnnn
    #nonmodified_flags = ~vt.other.index_to_boolmask(modified_indexes_, num_examples)
    #print(ut.debug_consec_list(modified_indexes_))
    # hack
    #modified_indexes = np.arange(num_examples)
    Xb_orig = vt.rectify_to_uint8(Xb_orig)
    Xb_ = vt.rectify_to_uint8(Xb_)

    orig_stack = stacked_img_pairs(Xb_orig, modified_indexes, yb_orig)
    warp_stack = stacked_img_pairs(Xb_, modified_indexes, yb)
    if shadows is not None:
        # hack
        shadow_stack = stacked_img_pairs(shadows, modified_indexes, yb)

    fnum = None
    fnum = pt.ensure_fnum(fnum)
    pt.figure(fnum)
    #next_pnum = pt.make_pnum_nextgen(nRows=2 + (shadows is not None), nCols=1)
    next_pnum = pt.make_pnum_nextgen(nCols=2 + (shadows is not None), nRows=1)
    pt.imshow(orig_stack, pnum=next_pnum(), title='before')
    pt.imshow(warp_stack, pnum=next_pnum(), title='after')

    if shadows is not None:
        pt.imshow(shadow_stack, pnum=next_pnum(), title='shadow_stack')


def testdata_augment():
    from ibeis_cnn import ingest_data, utils
    import vtool as vt
    dataset = ingest_data.grab_siam_dataset()
    cv2_data, labels = dataset.load_subset('valid')
    batch_size = 128
    Xb, yb = utils.random_xy_sample(cv2_data, labels, batch_size / 2, 2, seed=0)
    Xb = vt.rectify_to_float01(Xb)
    Xb_orig = Xb.copy()
    yb_orig = yb.copy()
    return Xb_orig, yb_orig, Xb, yb

@profile
def augment_affine(Xb, yb=None, rng=np.random):
    """
    CommandLine:
        python -m ibeis_cnn.augment --test-augment_affine --show --db PZ_MTEST

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_cnn.augment import *  # NOQA
        >>> from ibeis_cnn import ingest_data, utils, draw_results
        >>> Xb_orig, yb_orig, Xb, yb = testdata_augment()
        >>> rng = np.random.RandomState(0)
        >>> Xb, yb = augment_affine(Xb, yb, rng=rng)
        >>> ut.quit_if_noshow()
        >>> show_augmented_patches(Xb_orig, Xb, yb_orig, yb)
        >>> ut.show_if_requested()
    """
    assert Xb.max() <= 1.0, 'max/min = %r, %r' % (Xb.min(), Xb.max())
    assert Xb.min() >= 0.0, 'max/min = %r, %r' % (Xb.min(), Xb.max())
    import vtool as vt
    Xb1, Xb2 = Xb[::2], Xb[1::2]
    affprob_perterb = .7

    num = len(Xb1)
    # Determine which examples will be augmented
    affperterb_flags = rng.uniform(0.0, 1.0, size=num) < affprob_perterb

    affperterb_ranges = dict(
        zoom_range=None,
        max_tx=None,
        max_ty=None,
        max_shear=None,
        max_theta=None,
        enable_flip=False,
        enable_stretch=False,
    )
    affperterb_ranges.update(
        dict(
            #zoom_range=(1.0, 1.7),
            zoom_range=(1.0, 1.3),
            #zoom_range=(.7, 1.7),
            max_tx=2,
            max_ty=2,
            max_shear=TAU / 32,
            max_theta=TAU,
            enable_stretch=True,
            enable_flip=True,
        )
    )
    index_list = np.where(affperterb_flags)[0]

    affperterb_kw_list = [
        random_affine_kwargs(rng=rng, **affperterb_ranges)
        for index in index_list
    ]

    #lighting_perterb_ranges = dict(
    #    darken=(-.01, .01),
    #)

    import cv2
    #borderMode = cv2.BORDER_REFLECT101
    #borderMode = cv2.BORDER_REFLECT_101
    #borderMode = cv2.BORDER_WRAP
    #borderMode = cv2.BORDER_CONSTANT
    #borderMode = cv2.BORDER_REFLECT
    borderMode = cv2.BORDER_CONSTANT
    #borderMode = cv2.BORDER_REPLICATE
    flags = cv2.INTER_LANCZOS4
    borderValue = [.5] * Xb1.shape[-1]

    for index, kw in zip(index_list, affperterb_kw_list):
        Xb1[index] = np.clip(vt.affine_warp_around_center(
            Xb1[index], borderMode=borderMode, flags=flags,
            borderValue=borderValue, **kw), 0, 1)
        Xb2[index] = np.clip(vt.affine_warp_around_center(
            Xb2[index], borderMode=borderMode, flags=flags,
            borderValue=borderValue, **kw), 0, 1)
    with ut.embed_on_exception_context:
        assert Xb.max() <= 1.0, 'max/min = %r, %r' % (Xb.min(), Xb.max())
        assert Xb.min() >= 0.0, 'max/min = %r, %r' % (Xb.min(), Xb.max())
        #bad_xs = np.where((Xb > 1).sum(axis=-1).sum(axis=-1).sum(axis=-1) > 0)[0]
    Xb = Xb.astype(np.float32)
    return Xb, yb


@profile
def augment_shadow(Xb, yb=None, rng=np.random, return_shadowmaps=False):
    """
    CommandLine:
        python -m ibeis_cnn.augment --test-augment_shadow --show --db PZ_MTEST

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_cnn.augment import *  # NOQA
        >>> from ibeis_cnn import ingest_data, utils, draw_results
        >>> Xb_orig, yb_orig, Xb, yb = testdata_augment()
        >>> rng = np.random.RandomState(0)
        >>> Xb, yb, shadows = augment_shadow(Xb, yb, rng=rng, return_shadowmaps=True)
        >>> ut.quit_if_noshow()
        >>> show_augmented_patches(Xb_orig, Xb, yb_orig, yb, shadows)
        >>> ut.show_if_requested()
    """

    import vtool as vt
    # Rotate corresponding patches together
    Xb1, Xb2 = Xb[::2], Xb[1::2]
    perlin_perteb = .5

    def perlin_noise01(size):
        #scale = 128.0
        #scale = 80.0
        scale = size[0] * 1.5
        noise = (vt.perlin_noise(size, scale=scale, rng=rng).astype(np.float32)) / 255.0
        noise = np.transpose(noise[None, :], (1, 2, 0))
        return noise

    num = len(Xb1)

    perlinperterb_flags = rng.uniform(0.0, 1.0, size=num) < perlin_perteb

    #perlin_min = 0.2
    #perlin_max = 0.9
    #perlin_range = perlin_max - perlin_min
    if return_shadowmaps:
        shadows = np.empty(Xb.shape, dtype=Xb.dtype)
        shadows1, shadows2 =  shadows[::2], shadows[1::2]

    for index in np.where(perlinperterb_flags)[0]:
        img1 = Xb1[index]
        img2 = Xb2[index]
        # TODO: TAKE IN NORMALIZED POINTS
        noise1 = perlin_noise01(img1.shape[0:2])
        noise2 = perlin_noise01(img2.shape[0:2])

        #noise1 = np.clip(noise1 / .7, 0, 1)
        #noise2 = np.clip(noise2 / .7, 0, 1)
        noise1 = np.clip((noise1 ** .7 - .15) / .75, .1, 1)
        noise2 = np.clip((noise2 ** .7 - .15) / .75, .1, 1)

        if return_shadowmaps:
            shadows1[index] = noise1
            shadows2[index] = noise2

        #noise1[noise1 > .6] = 1
        #noise2[noise2 > .6] = 1

        #alpha1 = ((rng.rand() * perlin_range) + perlin_min)
        #alpha2 = ((rng.rand() * perlin_range) + perlin_min)
        #Xb1[index] = img1 ** (noise1 * alpha1)
        #Xb2[index] = img2 ** (noise2 * alpha2)
        #alpha1 = alpha2 = .5
        alpha1 = alpha2 = .5
        Xb1[index] = vt.blend_images_multiply(img1, noise1, alpha1)
        Xb2[index] = vt.blend_images_multiply(img2, noise2, alpha2)

        # Linear Burn
        #Xb1[index] = np.clip(img1 + noise1 - 1.0, 0, 1)
        #Xb2[index] = np.clip(img2 + noise2 - 1.0, 0, 1)

        #Xb1[index] = vt.blend_images_average(img1, noise1, alpha1)
        #Xb2[index] = vt.blend_images_average(img2, noise2, alpha2)

        #Xb1[index] = noise1
        #Xb2[index] = noise2
    #with ut.embed_on_exception_context:
    assert Xb.max() <= 1.0, 'max/min = %r, %r' % (Xb.min(), Xb.max())
    assert Xb.min() >= 0.0, 'max/min = %r, %r' % (Xb.min(), Xb.max())
    #bad_xs = np.where((Xb > 1).sum(axis=-1).sum(axis=-1).sum(axis=-1) > 0)[0]
    if return_shadowmaps:
        return Xb, yb, shadows
    return Xb, yb


@profile
def augment_gamma(Xb, yb=None, rng=np.random):
    """
    CommandLine:
        python -m ibeis_cnn.augment --test-augment_gamma --show --db PZ_MTEST

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_cnn.augment import *  # NOQA
        >>> from ibeis_cnn import ingest_data, utils, draw_results
        >>> Xb_orig, yb_orig, Xb, yb = testdata_augment()
        >>> rng = np.random.RandomState(0)
        >>> Xb, yb = augment_gamma(Xb, yb, rng=rng)
        >>> ut.quit_if_noshow()
        >>> show_augmented_patches(Xb_orig, Xb, yb_orig, yb)
        >>> ut.show_if_requested()
    """
    #import vtool as vt
    # Rotate corresponding patches together
    Xb1, Xb2 = Xb[::2], Xb[1::2]
    gamma_perterb = .5
    num = len(Xb1)

    # Determine which examples will be augmented
    gammaperterb_flags = rng.uniform(0.0, 1.0, size=num) < gamma_perterb

    # Modify exposure
    #perlin_min = 0.6
    #gamma_min = 1.7
    gamma_max = 1.7
    gamma_min = .4
    gamma_range = gamma_max - gamma_min
    for index in np.where(gammaperterb_flags)[0]:
        img1 = Xb1[index]
        img2 = Xb2[index]
        gamma1 = ((rng.rand() * gamma_range) + gamma_min)
        gamma2 = ((rng.rand() * gamma_range) + gamma_min)
        #print('gamma1 = %r' % (gamma1,))
        #print('gamma2 = %r' % (gamma2,))
        Xb1[index] = img1 ** (gamma1)
        Xb2[index] = img2 ** (gamma2)
    assert Xb.max() <= 1.0, 'max/min = %r, %r' % (Xb.min(), Xb.max())
    assert Xb.min() >= 0.0, 'max/min = %r, %r' % (Xb.min(), Xb.max())
    return Xb, yb


@profile
def augment_siamese_patches2(Xb, yb=None, rng=np.random):
    """
    CommandLine:
        python -m ibeis_cnn.augment --test-augment_siamese_patches2 --show --db=PZ_MTEST
        python -m ibeis_cnn.augment --test-augment_siamese_patches2 --show --colorspace='bgr'

        # Shows what augumentation looks like durring trainging
        python -m ibeis_cnn.train --test-pz_patchmatch --ds pzmtest --weights=new --arch=siaml2 --train --monitor --DEBUG_AUGMENTATION

    TODO:
        zoom in only if a true positive
        slightly different transforms for each image in the pair

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_cnn.augment import *  # NOQA
        >>> from ibeis_cnn import ingest_data, utils, draw_results
        >>> Xb_orig, yb_orig, Xb, yb = testdata_augment()
        >>> rng = np.random.RandomState(0)
        >>> Xb, yb = augment_siamese_patches2(Xb, yb, rng=rng)
        >>> ut.quit_if_noshow()
        >>> show_augmented_patches(Xb_orig, Xb, yb_orig, yb)
        >>> ut.show_if_requested()
    """
    augment_affine(Xb, yb, rng)
    augment_shadow(Xb, yb, rng)
    augment_gamma(Xb, yb, rng)
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
