import random

import numpy as np
from skimage.io import imread
from tifffile import imread


def data_generator(GT_image_dr, lowSNR_image_dr, wf_image_dr, patch_size, n_patches, n_channel, threshold, ratio, lp1,
                   lp2,
                   augment=False,
                   shuffle=False, add_noise=False):
    gt = imread(GT_image_dr).astype(np.float64)
    low = imread(lowSNR_image_dr).astype(np.float64)
    wf = imread(wf_image_dr).astype(np.float64)
    if len(gt.shape) == 3:
        gt = np.reshape(gt, (gt.shape[0], 1, gt.shape[1], gt.shape[2]))
        low = np.reshape(low, (low.shape[0], 1, low.shape[1], low.shape[2]))
        wf = np.reshape(wf, (wf.shape[0], 1, wf.shape[1], wf.shape[2]))
    if len(gt.shape) == 2:
        gt = np.reshape(gt, (1, 1, gt.shape[0], gt.shape[1]))
        low = np.reshape(low, (1, 1, low.shape[0], low.shape[1]))
        wf = np.reshape(wf, (1, 1, wf.shape[0], wf.shape[1]))
    print(gt.shape)
    # m = 5
    img_size = gt.shape[2]

    gt = gt / (gt.max(axis=(-1, -2))).reshape((gt.shape[0], gt.shape[1], 1, 1))
    low = low / (low.max(axis=(-1, -2))).reshape((low.shape[0], low.shape[1], 1, 1))
    wf = wf / (wf.max(axis=(-1, -2))).reshape((wf.shape[0], wf.shape[1], 1, 1))

    if add_noise:
        # gt = np.append(gt, gt, axis=0)
        # gt = np.append(gt, gt, axis=0)
        # wf = np.zeros(gt.shape)
        # low = np.zeros(gt.shape)
        for i in range(len(gt)):
            # wf[i] = np.random.poisson(gt[i] / lp1, size=gt[i].shape)
            low[i] = np.random.poisson(wf[i] / lp1, size=wf[i].shape)

    low = low / (low.max(axis=(-1, -2))).reshape((low.shape[0], low.shape[1], 1, 1))

    m = gt.shape[0]
    x = np.empty((m * n_patches * n_patches, patch_size, patch_size, 1), dtype=np.float64)
    w = np.empty((m * n_patches * n_patches, patch_size, patch_size, 1), dtype=np.float64)
    y = np.empty((m * n_patches * n_patches, patch_size, patch_size, 1), dtype=np.float64)

    rr = np.floor(np.linspace(0, img_size - patch_size, n_patches)).astype(np.int32)
    cc = np.floor(np.linspace(0, gt.shape[3] - patch_size, n_patches)).astype(np.int32)

    count = 0
    for l in range(m):
        for j in range(n_patches):
            for k in range(n_patches):
                x[count, :, :, 0] = low[l, n_channel, rr[j]:rr[j] + patch_size, cc[k]:cc[k] + patch_size]
                w[count, :, :, 0] = wf[l, n_channel, rr[j]:rr[j] + patch_size, cc[k]:cc[k] + patch_size]
                y[count, :, :, 0] = gt[l, n_channel, rr[j]:rr[j] + patch_size, cc[k]:cc[k] + patch_size]
                count = count + 1

    # if add_noise:
    #     gt = np.append(gt, gt, axis=0)
    #     gt = np.append(gt, gt, axis=0)
    #     low = np.zeros(gt.shape)
    #     print(low.shape)
    #     for i in range(len(gt)):
    #         low[i] = np.random.poisson(gt[i] / lp, size=gt[i].shape)

    if augment:
        count = x.shape[0]
        xx = np.zeros((4 * count, patch_size, patch_size, 1), dtype=np.float64)
        ww = np.zeros((4 * count, patch_size, patch_size, 1), dtype=np.float64)
        yy = np.zeros((4 * count, patch_size, patch_size, 1), dtype=np.float64)

        xx[0:count, :, :, :] = x
        xx[count:2 * count, :, :, :] = np.flip(x, axis=1)
        xx[2 * count:3 * count, :, :, :] = np.flip(x, axis=2)
        xx[3 * count:4 * count, :, :, :] = np.flip(x, axis=(1, 2))

        ww[0:count, :, :, :] = w
        ww[count:2 * count, :, :, :] = np.flip(w, axis=1)
        ww[2 * count:3 * count, :, :, :] = np.flip(w, axis=2)
        ww[3 * count:4 * count, :, :, :] = np.flip(w, axis=(1, 2))

        yy[0:count, :, :, :] = y
        yy[count:2 * count, :, :, :] = np.flip(y, axis=1)
        yy[2 * count:3 * count, :, :, :] = np.flip(y, axis=2)
        yy[3 * count:4 * count, :, :, :] = np.flip(y, axis=(1, 2))
    else:
        xx = x
        ww = w
        yy = y

    norm_x = np.linalg.norm(yy, axis=(1, 2))
    norm_x = norm_x / norm_x.max()
    ind_norm = np.where(norm_x > threshold)[0]
    print(len(ind_norm))

    xxx = np.empty((len(ind_norm), xx.shape[1], xx.shape[2], xx.shape[3]))
    www = np.empty((len(ind_norm), xx.shape[1], xx.shape[2], xx.shape[3]))
    yyy = np.empty((len(ind_norm), xx.shape[1], xx.shape[2], xx.shape[3]))

    for i in range(len(ind_norm)):
        xxx[i] = xx[ind_norm[i]]
        www[i] = ww[ind_norm[i]]
        yyy[i] = yy[ind_norm[i]]

    aa = np.linspace(0, len(xxx) - 1, len(xxx))
    random.shuffle(aa)
    aa = aa.astype(int)

    xxs = np.empty(xxx.shape, dtype=np.float64)
    wws = np.empty(www.shape, dtype=np.float64)
    yys = np.empty(yyy.shape, dtype=np.float64)

    if shuffle:
        for i in range(len(xxx)):
            xxs[i] = xxx[aa[i]]
            wws[i] = www[aa[i]]
            yys[i] = yyy[aa[i]]
    else:
        xxs = xxx
        wws = www
        yys = yyy

    xxs[xxs < 0] = 0
    for i in range(len(xxs)):
        for j in range(n_channel):
            xxs[i, :, :, j] = xxs[i, :, :, j] / xxs[i, :, :, j].max()
            wws[i, :, :, j] = wws[i, :, :, j] / wws[i, :, :, j].max()
            if yys[i, :, :, j].max() > 0:
                yys[i, :, :, j] = yys[i, :, :, j] / yys[i, :, :, j].max()

    ratio = ratio
    m1 = np.floor(xxs.shape[0] * ratio).astype(np.int32)
    x_train = xxs[0:m1]
    w_train = wws[0:m1]
    y_train = yys[0:m1]
    x_valid = xxs[m1::]
    w_valid = wws[m1::]
    y_valid = yys[m1::]

    print('The training set shape is:', x_train.shape)
    print('The validation set shape is:', x_valid.shape)
    return x_train, w_train, y_train, x_valid, w_valid, y_valid
