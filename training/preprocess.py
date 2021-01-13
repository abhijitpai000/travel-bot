"""
Module to pre-process data for training.

Functions:
- make_dataset: Generates appropriate data required for training from raw file.
    - tokenize_input: Tokenize input text.
"""

import torch
from torch.utils.data import Dataset
import json
from nltk.tokenize import word_tokenize

# Local Imports
from .utils import words_to_index, make_bow_vector, ohe_sequence


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


def vectorize_data(training_data, words_to_idx, target_labels):
    """
    Vectorize data into train_X, train_y.

    Parameters
    ----------
        training_data (list): List of tuples (instances, labels)
        words_to_idx (dict): Words to Indexes mapping.
        target_labels (list): Target Labels

    Returns
    -------
        train_X (tensor), train_y (tensor)
    """
    # Generating Vectorized train_X, train_y.
    train_X = []
    train_y = []

    for instance, label in training_data:
        # train_X.
        bow_vec = make_bow_vector(instance, words_to_idx)
        train_X.append(bow_vec)

        # train_y.
        target = target_labels.index(label)
        train_y.append(target)

    # Changing train_y to Long Tensor.
    train_y = torch.LongTensor(train_y)

    return train_X, train_y


def process_intents_data(file_name):
    """
    Pre-process intent classifier training data.

    Parameters
    ----------
        file_name (str): Ex: "intents_data.json" file name for pre-processing.

    Returns
    -------
        vocab (list), intents (list), train_X (tensor), train_y (tensor)
    """

    # Reading .json file for pre-processing.
    data_file_path = f"training/data/{file_name}"
    with open(data_file_path) as f:
        data = json.load(f)

    # Segregating data for training.
    intents = []
    vocab = []
    training_data = []

    for i in data["intents"]:
        tag = i["intent"]
        intents.append(tag)
        for j in i["query"]:
            tokens = tokenize_input(j)
            vocab.extend(tokens)
            training_data.append((tokens, tag))

    # Removing duplicates from vocab.
    vocab = set(vocab)
    words_to_idx = words_to_index(vocab)

    # Vectorized data.
    train_X, train_y = vectorize_data(training_data, words_to_idx, intents)

    return words_to_idx, intents, train_X, train_y


def process_actions_data(filename="actions_data.json"):
    """
    Pre-process next action predictor training data.

    Parameters
    ----------
        file_name (str): Ex: "actions_data.json" file name for pre-processing.

    Returns
    -------
        actions_to_idx (dict), idx_to_actions (dict),
        dict_size (int) , input_seq (tensor),
        target_seq (tensor)
    """
    data_file_path = f"training/data/{filename}"
    with open(data_file_path) as f:
        data = json.load(f)

    # Create training data.
    corpus = []
    for flow in data["flows"]:
        corpus.append(flow)

    # Add Padding.
    max_len = 6

    for i in range(len(corpus)):
        while len(corpus[i]) < max_len:
            corpus[i].append("padding")

    # Create vocab.
    vocab = set([actions for flows in corpus for actions in flows])
    actions_to_idx = {actions: idx for idx, actions in enumerate(vocab)}
    idx_to_actions = {idx: actions for idx, actions in enumerate(vocab)}

    # Creating sequential data.
    input_seq = []
    target_seq = []

    for i in range(len(corpus)):
        # input_seq = first to last-1
        input_seq.append(corpus[i][:-1])

        # target_seq = second to last
        target_seq.append(corpus[i][1:])

    # Sequential data to integers.
    for i in range(len(corpus)):
        input_seq[i] = [actions_to_idx[action] for action in input_seq[i]]
        target_seq[i] = [actions_to_idx[action] for action in target_seq[i]]

    # One Hot Encoding sequential data.
    batch_size = len(corpus)
    seq_size = max_len - 1
    dict_size = len(actions_to_idx)

    input_seq = ohe_sequence(sequence=input_seq,
                             batch_size=batch_size,
                             seq_size=seq_size,
                             dict_size=dict_size)

    # Converting to Tensors.
    input_seq = torch.from_numpy(input_seq)
    target_seq = torch.LongTensor(target_seq)

    return actions_to_idx, idx_to_actions, dict_size, input_seq, target_seq


class IntentClassifierDataset(Dataset):
    def __init__(self, train_X, train_y):
        """Constructor method"""
        self.train_X = train_X
        self.train_y = train_y
        self.n_samples = len(train_X)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.train_X[idx], self.train_y[idx]
