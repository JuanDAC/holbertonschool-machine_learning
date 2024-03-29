#!/usr/bin/env python3
"""
Changes the hue of an image
"""
import tensorflow as tf


def change_hue(image, delta):
    """
    Changes the hue of an image
    Arguments:
        * image is a 3D tf.Tensor containing the image to change
        * delta is the amount the hue should change
    Returns:
        * the changed image
    """
    return tf.image.adjust_hue(image, delta)
