#!/usr/bin/env python3
"""
File that calculates the unigram BLEU score for a sentence
"""

import numpy as np


def count_appearances(references, sentence):
    """
    Function that counts the appearances of each word in the sentence
    Args:
      -  references: list of reference translations
      -  sentence: list containing the model proposed sentence
    Returns:
      - count of each word of references in the sentence
    """

    sen = list(set(sentence))
    count_dict = {}

    for reference in references:
        for word in reference:
            if word in sen and word not in count_dict.keys():
                count_dict[word] = reference.count(word)
                continue
            if word in sen:
                new = reference.count(word)
                old = count_dict[word]
                count_dict[word] = max(new, old)

    return count_dict.values()


def clipping(references, sentence):
    """
    Function that calculates the clipping
    Arguments:
      - reference: length of the reference
      - sentence: length of the sentence
    Returns:
      - clipped count of each word of references in the sentence
    """
    len_sentence = len(sentence)
    list_references = []
    for reference in references:
        len_reference = len(reference)
        list_references.append(
            ((abs(len_reference - len_sentence)), len_reference))

    # Precision
    reference_len = sorted(list_references, key=lambda x: x[0])
    reference_len = reference_len[0][1]

    return reference_len, len_sentence


def uni_bleu(references, sentence):
    """
    Function that calculates the unigram BLEU score for a sentence
    Arguments:
      - references is a list of reference translations
        * each reference translation is a list of the words in the translation
      - sentence is a list containing the model proposed sentence
        * each word in the sentence should be considered as a possible match
    Returns:
      - the unigram BLEU score
    """
    count_values = count_appearances(references, sentence)
    reference_clipping, len_sentence = clipping(references, sentence)

    if len_sentence > reference_clipping:
        bp = 1

    if len_sentence <= reference_clipping:
        bp = np.exp(1 - (float(reference_clipping) / len_sentence))

    bleu_score = bp * np.exp(np.log(sum(count_values)) / len_sentence)

    return bleu_score
