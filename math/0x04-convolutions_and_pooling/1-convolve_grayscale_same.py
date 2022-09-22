#!/usr/bin/env python3
"""
Function that performs a same convolution on grayscale images
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Function that performs a same convolution on grayscale images
    Arguments:
      - images is a numpy.ndarray with shape (m, h, w) containing
        multiple grayscale images
        * m is the number of images
        * h is the height in pixels of the images
        * w is the width in pixels of the images
      - kernel is a numpy.ndarray with shape (kh, kw) containing
        the kernel for the convolution
        * kh is the height of the kernel
        * kw is the width of the kernel
    Returns:
      - A numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph = 0 if kh % 2 != 0 else int(kh / 2)
    pw = 0 if kw % 2 != 0 else int(kw / 2)
    images_padding = np.pad(images, pad_width=(
        (0, 0), (ph, ph), (pw, pw)), mode='constant')
    convolved = np.zeros((m, h, w))
    image_base = np.arange(m)
    for i in range(h):
        for j in range(w):
            convolved[image_base, i, j] = np.sum(
                images_padding[image_base, i: i + kh, j: j + kw]
                * kernel, axis=(1, 2)
            )
    return convolved
