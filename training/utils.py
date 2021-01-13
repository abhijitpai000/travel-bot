"""
Utility functions for training.

- words_to_index - Generates word-index mapping.
- make_bow_vector - Generates bag of words tensor.

"""
import torch
import numpy as np


def words_to_index(vocab):
    """
    Generates words-index mapping for training.

    Parameters
    ----------
        vocab (list): Tokenized text.

    Returns
    -------
        words_to_idx (dict): Mapping of words and indexes.
    """
    word_to_idx = {}

    for token in vocab:
        if token not in word_to_idx:
            word_to_idx[token] = len(word_to_idx)

    return word_to_idx


def make_bow_vector(query_tokens, words_to_idx):
    """
    Generate Bag of Words vectors.

    Parameters
    ----------
        query_tokens (list): Tokenized input text.
        words_to_idx (dict): Mapping of words and indexes.

    Returns
    -------
        bow_vector (tensor): Bag of words vector.
    """
    bow_vec = torch.zeros(len(words_to_idx), dtype=torch.float)

    for tokens in query_tokens:
        bow_vec[words_to_idx[tokens]] += 1

    return bow_vec


def ohe_sequence(sequence, batch_size, seq_size, dict_size):
    """
    One Hot Encode sequential input data for Action Detector.

    Parameters
    ----------
        sequence (np.array): Sequential data.
        batch_size (int): Number of dimensions.
        seq_size (int): Max Length - 1.
        dict_size (int): Total unique words in the vocab.

    Returns
    -------
        bow_vec (tensor)
    """
    bow_vec = np.zeros((batch_size, seq_size, dict_size), dtype=np.float32)

    for flow_num in range(batch_size):
        for action_num in range(seq_size):
            bow_vec[flow_num, action_num, sequence[flow_num][action_num]] = 1

    return bow_vec


def prediction_category(prediction, intents):
    """
    Extract intent from model's prediction.

    Parameters
    ---------
        prediction (tensor): Model's prediction for a given input.
        intents (list): List of intents.

    Returns
    -------
        predicted_category (str)

    """
    top_n, top_idx = prediction.topk(1)
    top_intent_idx = top_idx.item()
    return intents[top_intent_idx]
