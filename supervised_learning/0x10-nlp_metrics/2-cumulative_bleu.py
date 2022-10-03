#!/usr/bin/env python3
"""
File that contains the function cumulative_bleu
"""
import numpy as np


def cumulative_bleu(references, sentence, n):
    """
    Function that calculates the cumulative n-gram BLEU score for a sentence:
    Arguments:
        - references is a list of reference translations
            * each reference translation is a list of the words in the translation
        - sentence is a list containing the model proposed sentence
        - n is the size of the largest n-gram to use for evaluation
    All n-gram scores should be weighted evenly
    Returns:
        - the cumulative n-gram BLEU score
    """
    weights = np.ones(n) / n
    p_n = np.zeros(n)
    for i in range(n):
        p_n[i] = ngram_bleu(references, sentence, i + 1)
    return np.exp(np.sum(weights * np.log(p_n)))


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
    ngram_references, ngram_sentence = transform_grams(references, sentence, n)
    ngram_sentence_length = len(ngram_sentence)
    sentence_length = len(sentence)

    if ngram_sentence_length == 0:
        return 0

    count_dict = {}
    for reference in ngram_references:
        for word in reference:
            if word in ngram_sentence and word not in count_dict.keys():
                count_dict[word] = reference.count(word)
                continue
            if word in ngram_sentence:
                new = reference.count(word)
                old = count_dict[word]
                count_dict[word] = max(new, old)

    count = np.sum(list(count_dict.values()))
    return np.exp(np.log(count / ngram_sentence_length))
