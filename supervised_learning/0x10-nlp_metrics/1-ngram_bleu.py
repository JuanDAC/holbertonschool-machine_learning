#!/usr/bin/env python3
"""
File that contains the function ngram_bleu
"""
import numpy as np


def transform_grams(references, sentence, size_n_gram):
    """
    Function that calculates the n-gram BLEU score for a sentence:
    Arguments:
        - references is a list of reference translations
            * each reference translation is a list of the words in the
              translation
        - sentence is a list containing the model proposed sentence
        - n is the size of the n-gram to use for evaluation
    Returns:
        - the n-gram BLEU score
    """
    references = [ref[i:i+size_n_gram]
                  for ref in references for i in range(len(ref))]
    sentence = [sentence[i:i+size_n_gram] for i in range(len(sentence))]
    references = list(map(tuple, references))
    sentence = list(map(tuple, sentence))
    return references, sentence
