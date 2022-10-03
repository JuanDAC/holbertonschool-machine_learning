#!/usr/bin/env python3
"""
File that contains the function cumulative_bleu
"""
import numpy as np


def transform_grams(references, sentence, n):
    """
    Transforms references and sentence based on grams
    Arguments:
        - references is a list of reference translations
            * each reference translation is a list of the words in the
                translation
        - sentence is a list containing the model proposed sentence
        - n is the size of the n-gram to use for evaluation
    Returns: the n-gram BLEU score
        - references, sentence
    """
    if n == 1:
        return references, sentence

    ngram_sentence = []
    sentence_length = len(sentence)

    for i, word in enumerate(sentence):
        count = 0
        w = word
        for j in range(1, n):
            if sentence_length > i + j:
                w += " " + sentence[i + j]
                count += 1
        if count == j:
            ngram_sentence.append(w)

    ngram_references = []

    for ref in references:
        ngram_ref = []
        ref_length = len(ref)

        for i, word in enumerate(ref):
            count = 0
            w = word
            for j in range(1, n):
                if ref_length > i + j:
                    w += " " + ref[i + j]
                    count += 1
            if count == j:
                ngram_ref.append(w)
        ngram_references.append(ngram_ref)

    return ngram_references, ngram_sentence


def precision(references, sentence, n):
    """
    Calculates the precision for n-gram BLEU score for a sentence
    Arguments:
        - references contains reference translations
        - sentence contains the model proposed sentence
        - n the size of the n-gram to use for evaluation
    Returns:
        - the precision for n-gram BLEU score
    """
    ngram_references, ngram_sentence = transform_grams(
        references, sentence, n)
    ngram_sentence_length = len(ngram_sentence)

    sentence_dictionary = {word: ngram_sentence.count(word) for
                           word in ngram_sentence}
    references_dictionary = {}

    for ref in ngram_references:
        for gram in ref:
            if references_dictionary.get(gram) is None or \
                    references_dictionary[gram] < ref.count(gram):
                references_dictionary[gram] = ref.count(gram)

    matchings = {word: 0 for word in ngram_sentence}

    for ref in ngram_references:
        for gram in matchings.keys():
            if gram in ref:
                matchings[gram] = sentence_dictionary[gram]

    for gram in matchings.keys():
        if references_dictionary.get(gram) is not None:
            matchings[gram] = min(
                references_dictionary[gram], matchings[gram])

    precision = sum(matchings.values()) / ngram_sentence_length

    return precision


def brevity_penalty(references, sentence):
    """
    Function that calculates the brevity penalty for a sentence
    Arguments:
        - references is a list of reference translations
            * each reference translation is a list of the words in the
                translation
        - sentence is a list containing the model proposed sentence
    Returns: the brevity penalty for sentence
    """
    c = len(sentence)
    r = 0
    for ref in references:
        if abs(len(ref) - c) < abs(r - c):
            r = len(ref)

    if c > r:
        bp = 1
    else:
        bp = np.exp(1 - r / c)

    return bp


def cumulative_bleu(references, sentence, n):
    """
    Calculates the cumulative n-gram BLEU score for a sentence
    Arguments:
        - references list of reference translations
        - sentence contains the model proposed sentence
        - n the size of the larget n-gram to use for evaluation
    Returns:
        - score the cumulative n-gram BLEU score
    """
    precisions = [0] * n
    for i in range(n):
        precisions[i] = precision(references, sentence, i + 1)

    if len(precisions) > 0:
        bp = brevity_penalty(references, sentence)
        s = np.exp(np.sum((1 / n) * np.log(precisions)))
        score = bp * s
    else:
        score = 0

    return score
