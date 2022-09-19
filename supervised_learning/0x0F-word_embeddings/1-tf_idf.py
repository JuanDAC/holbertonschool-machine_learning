#!/usr/bin/env python3

"""
File that contains the tf_idf function

"""

from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    Function that creates a bag of words embedding matrix
    Arguments:
      - sentences is a list of sentences to analyze
      - vocab is a list of the vocabulary words to use for the analysis
    Returns:
      - embeddings, featuresj
    """
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform(sentences)
    features = vectorizer.get_feature_names()
    embeddings = X.toarray()
    return embeddings, features
