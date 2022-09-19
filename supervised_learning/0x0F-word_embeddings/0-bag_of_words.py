#!/usr/bin/env python3
"""
File that contains the bag_of_words function
"""


def bag_of_words(sentences, vocab=None):
    """
    Function that creates a bag of words embedding matrix
    Arguments:
     - sentences is a list of sentences to analyze
     - vocab is a list of the vocabulary words to use for the analysis
       If None, all words within sentences should be used
    Returns:
     - embeddings, features
       - embeddings is a numpy.ndarray of shape (s, f) containing the embeddings
         - s is the number of sentences in sentences
         - f is the number of features analyzed
       - features is a list of the features used for embeddings
    """
    if not isinstance(sentences, list) or len(sentences) == 0:
        return None, None

    if vocab is not None:
        if not isinstance(vocab, list) or len(vocab) == 0:
            return None, None

    words = {}
    for sentence in sentences:
        if not isinstance(sentence, str):
            return None, None

        for word in sentence.split():
            if word not in words:
                words[word] = 1
            else:
                words[word] += 1

    if vocab is not None:
        for word in vocab:
            if not isinstance(word, str):
                return None, None

            if word not in words:
                words[word] = 1
            else:
                words[word] += 1

    features = sorted(words.keys())
    embeddings = np.zeros((len(sentences), len(features)))

    for i, sentence in enumerate(sentences):
        for j, feature in enumerate(features):
            embeddings[i][j] = sentence.split().count(feature)

    return embeddings, features
