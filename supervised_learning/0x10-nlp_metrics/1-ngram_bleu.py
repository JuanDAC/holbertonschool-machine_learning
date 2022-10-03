#!/usr/bin/env python3
"""
File that contains the function ngram_bleu
"""
import numpy as np


def ngram_bleu(references, sentence, n):
    """
    Function that calculates the n-gram BLEU score for a sentence:
    Arguments:
        - references is a list of reference translations
            * each reference translation is a list of the words in the translation
        - sentence is a list containing the model proposed sentence
        - n is the size of the n-gram to use for evaluation
    Returns:
        - the n-gram BLEU score
    """
    len_sentence = len(sentence)
    list_references = []
    for reference in references:
        len_reference = len(reference)
        list_references.append(
            ((abs(len_reference - len_sentence)), len_reference))

    reference_len = sorted(list_references, key=lambda x: x[0])

    reference_len = reference_len[0][1]

    count_dict = {}
    for reference in references:
        for i in range(len(sentence) - n + 1):
            ngram = tuple(sentence[i:i + n])
            if ngram not in count_dict.keys():
                count_dict[ngram] = reference.count(ngram)
                continue
            new = reference.count(ngram)
            old = count_dict[ngram]
            count_dict[ngram] = max(new, old)

    count_dict = count_dict.values()
    count_dict = sum(count_dict)
    precision = count_dict / len_sentence

    if len_sentence > reference_len:
        bp = 1
    else:
        bp = np.exp(1 - (reference_len / len_sentence))

    bleu_score = bp * precision

    return bleu_score
