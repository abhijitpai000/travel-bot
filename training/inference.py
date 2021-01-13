"""
Command Line Utility for testing trained model.
"""
import torch
import numpy as np

# Local Imports.
from .models import MultiLayerPerceptron, VanillaRNN
from .utils import make_bow_vector,ohe_sequence, prediction_category
from .preprocess import tokenize_input


def intent_detection(user_input):
    """
    Intent Detection.

    Parameters
    ----------
        user_input (str): User Input.

    """
    # Loading Model State.
    model_data = torch.load("training/data/model_data.pth")
    model_state = model_data["model_state"]
    input_size = model_data["input_size"]
    hidden_size = model_data["hidden_size"]
    output_size = model_data["output_size"]
    intents = model_data["intents"]
    words_to_idx = model_data["words_to_idx"]

    model = MultiLayerPerceptron(input_size,
                                 hidden_size,
                                 output_size)
    model.eval()
    model.load_state_dict(model_state)

    tokens = tokenize_input(user_input)
    bow_vec = make_bow_vector(tokens)
    prediction = model(bow_vec)
    intent_detected = prediction_category(prediction, intents)

    return intent_detected


def action_prediction(intent):
    """
    Predict Next Action based on Intent

    Parameters
    ---------
        intent (list): Current User Intent.

    Returns
    -------
        action.
    """

    # Load Model data.
    model_data = torch.load("training/data/actions_predictor_data.pth")
    model_state = model_data["model_state"]
    actions_to_idx = model_data["actions_to_idx"]
    idx_to_actions = model_data["idx_to_actions"]
    dict_size = model_data["dict_size"]

    # Processing intent.
    intent = np.array([[actions_to_idx[intents] for intents in intent]])

    intent = ohe_sequence(sequence=intent,
                          batch_size=1,
                          seq_size=intent.shape[1],
                          dict_size=dict_size)

    intent = torch.from_numpy(intent)

    # Initializing model.
    model = VanillaRNN(input_size=dict_size,
                       hidden_size=12,
                       output_size=dict_size,
                       n_layers=1)
    model.load_state_dict(model_state)
    model.eval()
    out, _ = model(intent)

    _, action_idx = torch.topk(out, k=1)
    action_idx = [idxs.item() for idxs in action_idx]

    return [idx_to_actions[idx] for idx in action_idx]
