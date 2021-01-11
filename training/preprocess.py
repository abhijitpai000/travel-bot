"""
Module to pre-process datasets for training.

Functions:
- preprocess_data: Generates appropriate data required for training from raw file.
    - tokenize_input: Tokenize input text.
"""

import json
from pathlib import Path
from nltk.tokenize import word_tokenize


def tokenize_input(input_seq):
    """
    Tokenize input text.

    Parameters
    ----------
        input_seq (list): Input text sequence.

    Returns
    -------
        input_tokens (list)
    """
    return [word.lower() for word in word_tokenize(input_seq)]


def preprocess_data(file_name):
    """
    Pre-process training datasets.

    Parameters
    ----------
        file_name (str): Ex: "data.json" file name for pre-processing.

    Returns
    -------
        training_data (dict), vocab (list), intents (list)
    """

    # Reading .json file for pre-processing.
    data_file_path = Path.cwd() / f"datasets/{file_name}"
    with open(data_file_path) as f:
        data = json.load(f)

    # Segregating data for training.
    intents = []
    vocab = []
    training_data = {}

    for i in data["intents"]:
        tag = i["intent"]
        intents.append(tag)
        for j in i["query"]:
            tokens = tokenize_input(j)
            vocab.extend(tokens)
        training_data[tag] = vocab

    # Removing duplicates from vocab.
    vocab = [word for word in vocab if word not in vocab]

    return training_data, vocab, intents
