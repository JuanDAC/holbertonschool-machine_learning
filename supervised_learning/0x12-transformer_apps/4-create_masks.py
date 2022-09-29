#!/usr/bin/env python3
"""File that contains the class Dataset"""
import tensorflow as tf
import tensorflow_datasets as tfds


class Dataset:
    """Class that loads and preps a dataset for machine translation"""

    def __init__(self):
        """
        Constructor should call the instance method
        tokenize_dataset to create the instance attributes tokenizer_pt
        and tokenizer_en update the data_train and data_validate
        attributes by tokenizing the examples

        batch_size is the batch size for training/validation
        max_len is the maximum number of tokens allowed per example sentence
        update the data_train attribute by performing the following actions:
        filter out all examples that have either sentence with more than max_len tokens
        cache the dataset to increase performance
        shuffle the entire dataset
        prefetch the dataset using tf.data.experimental.AUTOTUNE to increase performance
        update the data_validate attribute by performing the following actions:
        filter out all examples that have either sentence with more than max_len tokens
        split the dataset into padded batches of size batch_size
        """
        self.data_train = tfds.load(
            'ted_hrlr_translate/pt_to_en', split='train', as_supervised=True)
        self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en', split='validation',
            as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)
        self.data_train = self.data_train.filter(self.filter_max_length)
        self.data_valid = self.data_valid.filter(self.filter_max_length)
        self.data_train = self.data_train.cache()
        self.data_train = self.data_train.shuffle(1000)
        self.data_train = self.data_train.padded_batch(
            64, padded_shapes=([None], [None]))
        self.data_train = self.data_train.prefetch(
            tf.data.experimental.AUTOTUNE)
        self.data_valid = self.data_valid.padded_batch(
            64, padded_shapes=([None], [None]))

    def filter_max_length(self, x, y, max_length=10):
        """
        Method that filters out examples that are longer than max_length
        Arguments:
            - x is the tf.Tensor containing the input sentence
            - y is the tf.Tensor containing the target sentence
            - max_length is the maximum number of tokens allowed per example sentence
        Returns:
            - True or False
        """
        return tf.logical_and(tf.size(x) <= max_length,
                              tf.size(y) <= max_length)

    def tokenize_dataset(self, data):
        """
        Method should use
            tfds.deprecated.text.SubwordTextEncoder.build_from_corpus
        Arguments:
            - data tf.data.Dataset whose examples are formatted
                    as a tuple (pt, en)
        Returns:
            - tokenizer_pt, tokenizer_en
        """
        tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder\
            .build_from_corpus(
                (pt.numpy() for pt, en in data), target_vocab_size=2**15)
        tokenizer_en = tfds.deprecated.text.SubwordTextEncoder\
            .build_from_corpus(
                (en.numpy() for pt, en in data), target_vocab_size=2**15)
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        method should use the instance attributes tokenizer_pt and tokenizer_en
        to tokenize the sentences
        Arguments:
            - pt tf.Tensor containing the Portuguese sentence
            - en tf.Tensor containing the corresponding English sentence
        Returns:
            - pt_tokens, en_tokens
        """
        pt_tokens = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(
            pt.numpy()) + [self.tokenizer_pt.vocab_size + 1]
        en_tokens = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(
            en.numpy()) + [self.tokenizer_en.vocab_size + 1]
        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        Method should return a tuple (pt, en) containing the Portuguese
        and the English tokens, respectively
        Arguments:
            - pt tf.Tensor containing the Portuguese sentence
            - en tf.Tensor containing the corresponding English sentence
        Returns:
            - pt, en
        """
        pt_encoded, en_encoded = tf.py_function(
            self.encode, [pt, en], [tf.int64, tf.int64])
        pt_encoded.set_shape([None])
        en_encoded.set_shape([None])
        return pt_encoded, en_encoded

    def create_masks(inputs, target):
        """
        Method should use the following masks:
            - encoder_mask is the tf.Tensor padding mask of shape
                (batch_size, 1, 1, seq_len_in) to be applied in the encoder
            - combined_mask is the tf.Tensor of shape
                (batch_size, 1, seq_len_out, seq_len_out) used in the
                1st attention block in the decoder to pad and mask future
                tokens in the input received by the decoder. It takes
                the maximum between a look ahead mask and the decoder
                target padding mask.
            - decoder_mask is the tf.Tensor padding mask of shape
                (batch_size, 1, 1, seq_len_in) used in the 2nd attention
                block in the decoder.
            - look_ahead_mask is the tf.Tensor lower triangular
                tf.linalg.band_part of shape (seq_len_out, seq_len_out)
                with values 1 along the lower diagonal and 0 everywhere else
            - decoder_target_padding_mask is the tf.Tensor padding mask
                of shape (batch_size, 1, 1, seq_len_in) to be applied in
                the 1st attention block in the decoder.
        Returns:
            - encoder_mask, combined_mask, decoder_mask
        """
        encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
        encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]
        size = target.shape[1]
        look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        decoder_target_padding_mask = tf.cast(
            tf.math.equal(target, 0), tf.float32)
        decoder_target_padding_mask = decoder_target_padding_mask[:,
                                                                  tf.newaxis, tf.newaxis, :]
        combined_mask = tf.maximum(
            decoder_target_padding_mask, look_ahead_mask)
        decoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
        decoder_mask = decoder_mask[:, tf.newaxis, tf.newaxis, :]
        return encoder_mask, combined_mask, decoder_mask
