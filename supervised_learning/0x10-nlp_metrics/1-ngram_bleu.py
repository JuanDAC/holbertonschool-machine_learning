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

    clipped_count = sum(count_dict.values())
    precision = clipped_count / ngram_sentence_length

    reference_len = []
    for reference in ngram_references:
        reference_len.append(len(reference))
    closest_ref = min(reference_len, key=lambda x: abs(x - sentence_length))

    if sentence_length > closest_ref:
        bp = 1
    else:
        bp = np.exp(1 - (closest_ref / sentence_length))

    bleu_score = bp * precision

    return bleu_score
