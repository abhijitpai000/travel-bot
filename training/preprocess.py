"""
Module to pre-process the training data.
"""

import json
from pathlib import Path


# from nltk.tokenize import word_tokenize
#
# # Local Imports.
# from .utils import word_to_index, make_bow_vector


def tokenize_input(input_seq):
    """
    Tokenize input text.

    Parameters
    ----------
        input_seq (list): Input text sequence.

    Returns
    -------
        #todo: add return.
    """
    pass


def preprocess_data():
    """
    Pre-process training data.

    Returns
    -------
        #todo: add return
    """
    training_file_path = Path.cwd() / "data/training_data.json"

    with open(training_file_path) as f:
        training_data = json.load(f)
    return


if __name__ == '__main__':
    preprocess_data()
