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
