import torch
import numpy as np

# Local Imports.
from .config import INTENT_CLASSIFIER_DATA, ACTIONS_PREDICTOR_DATA
from .models import MultiLayerPerceptron, VanillaRNN
from .utils import tokenize_input, make_bow_vector, ohe_sequence


def intent_classifier(user_input):
    """
    Predicts user's intent.

    Parameters
    ----------
        user_input (str): User Input.

    Returns
    -------
        intent_detected (str)
    """
    model_data = torch.load(INTENT_CLASSIFIER_DATA)

    model_state = model_data["model_state"]
    input_size = model_data["input_size"]
    hidden_size = model_data["hidden_size"]
    output_size = model_data["output_size"]
    words_to_idx = model_data["words_to_idx"]
    intents = model_data["intents"]

    model = MultiLayerPerceptron(input_size=input_size,
                                 hidden_size=hidden_size,
                                 output_size=output_size)
    model.load_state_dict(model_state)
    model.eval()

    # Preprocess user input text.
    input_tokens = tokenize_input(user_input)
    bow_vec = make_bow_vector(query_tokens=input_tokens,
                              words_to_idx=words_to_idx)

    # Generate Predictions.
    prediction = model(bow_vec)
    _, predicted_idx = torch.topk(prediction, k=1)
    predicted_idx = predicted_idx.item()

    return intents[predicted_idx]


def action_predictor(intent_seq):
    """
    Predicts next best action based on intent detected.

    Parameters
    ----------
        intent_seq (array): Series of intents captured.

    Returns
    -------
        action (str)
    """
    model_data = torch.load(ACTIONS_PREDICTOR_DATA)

    model_state = model_data["model_state"]
    actions_to_idx = model_data["actions_to_idx"]
    idx_to_actions = model_data["idx_to_actions"]
    dict_size = model_data["dict_size"]

    model = VanillaRNN(input_size=dict_size,
                       hidden_size=12,
                       output_size=dict_size,
                       n_layers=1)
    model.load_state_dict(model_state)
    model.eval()

    # Process intent_seq.
    intents = np.array([[actions_to_idx[intents] for intents in intent_seq]])
    intents = ohe_sequence(sequence=intents,
                           batch_size=1,
                           seq_size=intents.shape[1],
                           dict_size=dict_size)
    intents = torch.from_numpy(intents)

    # Generating Predictions.
    prediction, _ = model(intents)
    _, action_idx = torch.topk(prediction, k=1)
    action_idx = [idx.item() for idx in action_idx]

    return idx_to_actions[action_idx[-1]]
