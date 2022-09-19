#!/usr/bin/env python3
"""
File that calculates the unigram BLEU score for a sentence
"""

import numpy as np


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
    ```
    """
    sentence_length = len(sentence)
    references_length = []
    words = {}
    for reference in references:
        references_length.append(len(reference))
        for word in reference:
            if word not in words:
                words[word] = 1
    for word in sentence:
        if word in words:
            words[word] += 1
    words = np.array(list(words.values()))
    words = np.where(words > 1, 1, 0)
    words = np.sum(words)
    best_match = np.argmin(
        np.abs(np.array(references_length) - sentence_length))
    best_match = references[best_match]
    overlap = 0
    for word in sentence:
        if word in best_match:
            overlap += 1
    bleu_score = overlap / sentence_length
    if sentence_length > len(best_match):
        bp = 1
    else:
        bp = np.exp(1 - (len(best_match) / sentence_length))
    return (bp * bleu_score) * 2
