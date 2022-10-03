#!/usr/bin/env python3
"""
File that contains the function ngram_bleu
"""
import numpy as np


def transform_grams(references, sentence, n):
    """
    Function that transforms references and sentence based on grams
        Args:
        references is a list of reference translations
        each reference translation is a list of the words in the
        translation
        sentence is a list containing the model proposed sentence
        n is the size of the n-gram to use for evaluation
        Returns: the n-gram BLEU score
    """
    if n == 1:
        return references, sentence

    ngram_sentence = []
    sentence_length = len(sentence)
    for i in range(sentence_length - n + 1):
        ngram_sentence.append(tuple(sentence[i:i + n]))

    ngram_references = []
    for reference in references:
        ngram_reference = []
        reference_length = len(reference)
        for i in range(reference_length - n + 1):
            ngram_reference.append(tuple(reference[i:i + n]))
        ngram_references.append(ngram_reference)

    return ngram_references, ngram_sentence


def ngram_bleu(references, sentence, n):
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
    references, sentence = transform_grams(references, sentence, n)

    clipped_count = 0
    count = 0
    for word in sentence:
        if word in references:
            clipped_count += 1
        count += 1

    if clipped_count == 0:
        return 0

    precision = clipped_count / count
    brevity_penalty = 1
    len_sentence = len(sentence)
    list_references = []
    for reference in references:
        len_reference = len(reference)
        list_references.append(
            ((abs(len_reference - len_sentence)), len_reference))

    reference_len = sorted(list_references, key=lambda x: x[0])

    if len_sentence > reference_len[0][1]:
        brevity_penalty = np.exp(1 - (len_sentence / reference_len[0][1]))

    return brevity_penalty * precision
