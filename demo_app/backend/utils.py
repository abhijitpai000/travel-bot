"""
Utility functions for Production

- tokenize_input - Tokenizes Sentences.
- make_bow_vector - Generates bag of words tensor.

"""
import torch
import numpy as np
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')


def tokenize_input(input_seq):
    """
    Tokenize input text.

    Parameters
    ----------
        input_seq (str): Input text sequence.

    Returns
    -------
        input_tokens (list)
    """
    tokens = word_tokenize(input_seq)
    return [word.lower() for word in tokens]


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
        if tokens not in words_to_idx:
            tokens = "unk"
            bow_vec[words_to_idx[tokens]] += 1
        else:
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
