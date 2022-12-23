#!/usr/bin/env python3
"""
Randomly changes the brightness of an image
"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """
    Randomly changes the brightness of an image
    Arguments:
        * image is a 3D tf.Tensor containing the image to change
        * max_delta is the maximum amount the image should be brightened
        (or darkened)
    Returns:
        * the changed image
    """
    return tf.image.adjust_brightness(image, max_delta)
