"""
Flips an image horizontally
"""
import tensorflow as tf


def flip_image(image):
    """
    Flips an image horizontally
    Arguments:
        * image is a 3D tf.Tensor containing the image to flip
    Returns:
        * the flipped image
    """
    return tf.image.flip_left_right(image)
