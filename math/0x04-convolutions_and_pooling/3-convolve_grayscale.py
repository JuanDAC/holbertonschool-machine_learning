#!/usr/bin/env python3
"""
File that contains the function convolve_grayscale
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Function should use a loop over i and j instead of np.sum or np.convolve
    Arguments:
      - images is a numpy.ndarray with shape (m, h, w) containing multiple
        grayscale images
        * m is the number of images
        * h is the height in pixels of the images
        * w is the width in pixels of the images
      - kernel is a numpy.ndarray with shape (kh, kw) containing the
        kernel for the convolution
        * kh is the height of the kernel
        * kw is the width of the kernel
      - padding is a tuple of (ph, pw)
        * ph is the padding for the height of the image
        * pw is the padding for the width of the image
    Returns:
      - A numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph = int(((h - 1) * sh + kh - h) / 2)
        pw = int(((w - 1) * sw + kw - w) / 2)
    if type(padding) == tuple:
        ph, pw = padding

    convolved_h = int(((h + (2 * ph) - kh) / sh) + 1)
    convolved_w = int(((w + (2 * pw) - kw) / sw) + 1)
    images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')
    conv = np.zeros((m, convolved_h, convolved_w))
    for i in range(convolved_h):
        for j in range(convolved_w):
            conv[:, i, j] = np.sum(
                images[:,
                       i * sh: i * sh + kh,
                       j * sw: j * sw + kw
                       ] * kernel, axis=(1, 2))
    return conv
