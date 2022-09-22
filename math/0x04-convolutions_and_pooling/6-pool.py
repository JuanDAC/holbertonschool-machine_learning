#!/usr/bin/env python3
"""
File that defines a function called pool
"""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Function should use a loop over i and j instead of np.sum or np.convolve
    Arguments:
      - images is a numpy.ndarray with shape (m, h, w) containing multiple
        * m is the number of images
        * h is the height in pixels of the images
        * w is the width in pixels of the images
        * c is the number of channels in the image
      - kernel_shape is a tuple of (kh, kw) containing the kernel shape for
        the pooling
        * kh is the height of the kernel
        * kw is the width of the kernel
      - stride is a tuple of (sh, sw)
        * sh is the stride for the height of the image
        * sw is the stride for the width of the image
      - mode indicates the type of pooling
        * max indicates max pooling
        * avg indicates average pooling
    Returns:
      - A numpy.ndarray containing the convolved images
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride
    ph = int(((h - kh) / sh) + 1)
    pw = int(((w - kw) / sw) + 1)
    pool = np.zeros((m, ph, pw, c))
    for i in range(ph):
        for j in range(pw):
            if mode == 'max':
                pool[:, i, j, :] = np.max(
                    images[:,
                           i * sh: i * sh + kh,
                           j * sw: j * sw + kw
                           ], axis=(1, 2))
            if mode == 'avg':
                pool[:, i, j, :] = np.average(
                    images[:,
                           i * sh: i * sh + kh,
                           j * sw: j * sw + kw
                           ], axis=(1, 2))
    return pool
