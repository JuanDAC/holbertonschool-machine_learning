#!/usr/bin/env python3
"""
Defines function that answers questions from multiple reference texts on loop
"""


import numpy as np
import os
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(corpus_path):
    """
        Based on the previous tasks, write a function that
        answers questions from multiple reference texts:
        Arguments:
            - corpus_path is the path to the corpus of reference documents
    """
    tokenizer = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad')
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")
    documents = []
    for file in os.listdir(corpus_path):
        if not file.endswith('.md'):
            continue
        with open(os.path.join(corpus_path, file)) as f:
            documents.append(f.read())
    embed = hub.load(
        "https://tfhub.dev/google/universal-sentence-encoder-large/5")
    embeddings = embed(documents)
    while (1):
        question = input("Q: ")
        if question.lower() in ["exit", "quit", "goodbye", "bye"]:
            print("A: Goodbye")
            exit()
        question_tokens = tokenizer.tokenize(question)
        question_embeddings = embed([question])
        corr = np.inner(embeddings, question_embeddings)
        closest = np.argmax(corr[0])
        reference = documents[closest]
        reference_tokens = tokenizer.tokenize(reference)
        tokens = ['[CLS]'] + question_tokens + \
            ['[SEP]'] + reference_tokens + ['[SEP]']
        input_word_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_word_ids)
        input_type_ids = [0] * (1 + len(question_tokens) + 1) + \
            [1] * (len(reference_tokens) + 1
                   )
        input_word_ids, input_mask, input_type_ids = map(
            lambda t: tf.expand_dims(
                tf.convert_to_tensor(t, dtype=tf.int32), 0),
            (input_word_ids, input_mask, input_type_ids),
        )
        outputs = model([input_word_ids, input_mask, input_type_ids])
        short_start = tf.argmax(outputs[0][0][1:]) + 1
        short_end = tf.argmax(outputs[1][0][1:]) + 1
        answer_tokens = tokens[short_start:short_end+1]
        answer = tokenizer.convert_tokens_to_string(answer_tokens)
        if answer == '':
            print("A: Sorry, I do not understand your question.")
        else:
            print("A: {}".format(answer))
