#!/usr/bin/env python3
"""
File that contains the function semantic_search
"""
import numpy as np
import os
import tensorflow_hub as hub


def semantic_search(corpus_path, sentence):
    """
    Function that performs semantic search on a corpus of documents
    Arguments:
        - corpus_path is the path to the corpus of reference documents
                      on which to perform semantic search
        - sentence is the sentence from which to perform semantic search
    Returns:
        - the reference text of the document most similar to sentence
    """
    documents = [sentence]
    for file in os.listdir(corpus_path):
        if not file.endswith('.md'):
            continue
        with open(os.path.join(corpus_path, file)) as f:
            documents.append(f.read())
    embed = hub.load(
        "https://tfhub.dev/google/universal-sentence-encoder-large/5")
    embeddings = embed(documents)
    corr = np.inner(embeddings, embeddings)
    closest = np.argmax(corr[0, 1:])
    return documents[closest + 1]
