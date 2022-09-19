#!/usr/bin/env python3

def transform_grams(references, sentence, size_n_gram):
    """
    Function that transforms references and sentence based on grams
      Arguments:
        references - list of reference translations
        each reference translation is a list of the words in the
        translation
        sentence - list containing the model proposed sentence
        size_n_gram - size of the n-gram to use for evaluation
      Returns: 
      - n-gram BLEU score
    """
    # Create a dictionary of n-grams in sentence
    sentence_grams = {}
    for i in range(len(sentence) - size_n_gram + 1):
        n_gram = tuple(sentence[i:i + size_n_gram])
        if n_gram not in sentence_grams:
            sentence_grams[n_gram] = 1
        else:
            sentence_grams[n_gram] += 1

    # Create a dictionary of n-grams in references
    references_grams = {}
    for reference in references:
        for i in range(len(reference) - size_n_gram + 1):
            n_gram = tuple(reference[i:i + size_n_gram])
            if n_gram not in references_grams:
                references_grams[n_gram] = 1
            else:
                references_grams[n_gram] += 1

    # Calculate the overlap
    overlap = {}
    for n_gram in sentence_grams:
        if n_gram in references_grams:
            overlap[n_gram] = min(sentence_grams[n_gram],
                                  references_grams[n_gram])

    # Calculate the BLEU score
    bleu_score = sum(overlap.values()) / sum(sentence_grams.values())

    return bleu_score
