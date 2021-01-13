"""
Training Script.

"""
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

# Local Imports.
from .preprocess import process_intents_data,process_actions_data, IntentClassifierDataset
from .models import MultiLayerPerceptron, VanillaRNN


def train_intent_classifier(filename="intents_data.json"):
    """
    Function to Intent Classifier using Multi-Layer Perceptron.

    Parameters
    ----------
        filename (str with .json extension): Raw file name.

    Yields
    ------
        intent_classifier_data.pth.
    """
    words_to_idx, intents, train_X, train_y = process_intents_data(filename)

    input_size = len(words_to_idx)
    output_size = len(train_y)
    hidden_size = 8
    batch_size = 8
    learning_rate = 0.01
    num_epoch = 300

    # Initializing values.
    dataset = IntentClassifierDataset(train_X, train_y)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True)
    model = MultiLayerPerceptron(input_size=input_size,
                                 hidden_size=hidden_size,
                                 output_size=output_size)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate)

    t = tqdm(range(num_epoch), desc="Intent Classifier")

    for epoch in t:
        for instance, label in dataloader:
            # Forward Pass.
            output = model(instance)

            # Loss Computation.
            loss = loss_func(output, label)
            t.set_postfix({'Loss': loss.item()})

            # Clear Gradients.
            optimizer.zero_grad()

            # Back Propagate.
            loss.backward()

            # Update Parameters.
            optimizer.step()

    # Saving Learned Parameters.
    model_data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "words_to_idx": words_to_idx,
        "intents": intents,

    }
    file_path = Path.cwd() / "training/data/intent_classifier_data.pth"
    torch.save(model_data, file_path)
    return


def train_action_predictor(filename="actions_data.json"):
    """
    Function to Next Action Predictor using Vanilla RNN.

    Parameters
    ----------
        filename (str with .json extension): Raw file name.

    Yields
    ------
        actions_predictor_data.pth.
    """
    actions_to_idx, idx_to_actions, dict_size, input_seq, target_seq = process_actions_data(filename)
    learning_rate = 0.001
    num_epochs = 1000

    # Initializing Model Parameters.

    model = VanillaRNN(input_size=dict_size,
                       hidden_size=12,
                       output_size=dict_size,
                       n_layers=1)

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    t = tqdm(range(num_epochs), desc="Actions Predictor")

    # Training Loop.

    for epoch in t:
        # Forward Pass.
        out, hidden = model(input_seq)

        # Loss.
        loss = loss_func(out, target_seq.view(-1))
        t.set_postfix({'Loss': loss.item()})

        # Clear Gradients.
        optimizer.zero_grad()

        # Back-propagation
        loss.backward()

        # Update Parameters
        optimizer.step()

    # Saving Model Parameters.
    model_data = {
        "model_state": model.state_dict(),
        "actions_to_idx": actions_to_idx,
        "dict_size": dict_size,
        "idx_to_actions": idx_to_actions
    }

    file_path = Path.cwd() / "training/data/actions_predictor_data.pth"
    torch.save(model_data, file_path)


