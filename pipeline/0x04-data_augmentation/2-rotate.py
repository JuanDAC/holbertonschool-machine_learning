#!/usr/bin/env python3
"""
Rotates an image by 90 degrees counter-clockwise
"""
import tensorflow as tf


def rotate_image(image):
    """
    Rotates an image by 90 degrees counter-clockwise
    Arguments:
        * image is a 3D tf.Tensor containing the image to rotate
    Returns:
        * the rotated image
    """
    return tf.image.rot90(image, k=1)
