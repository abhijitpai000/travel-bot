"""
Models classes for training.

Intent Classification
- Multi-Layer Perceptron

Next Action Detection
- Vanilla RNN

"""
import torch
import torch.nn as nn


class MultiLayerPerceptron(nn.Module):
    """Three-Layer Perceptron"""

    def __init__(self, input_size, hidden_size, output_size):
        """Constructor Method"""

        super(MultiLayerPerceptron, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, input_seq):
        """Forward Pass Method"""
        out = self.linear1(input_seq)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        return out


class VanillaRNN(nn.Module):
    """Single Layer Vanilla RNN"""

    def __init__(self, input_size, hidden_size, output_size, n_layers):
        super(VanillaRNN, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers

        # Single Layer RNN.
        self.rnn = nn.RNN(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=n_layers,
                          batch_first=True)

        # Fully Connected Layer.
        self.fcl = nn.Linear(hidden_size,
                             output_size)

    def forward(self, input_seq):
        """Forward Pass"""
        batch_size = input_seq.size(0)

        # Initial Hidden layer with zeros.
        hidden = self.initHidden(batch_size)

        # Input and Hidden to RNN Layer.
        out, hidden = self.rnn(input_seq, hidden)

        # Output to FCL.
        out = out.contiguous().view(-1, self.hidden_size)
        out = self.fcl(out)

        return out, hidden

    def initHidden(self, batch_size):
        """First Hidden layer with zeroes initiated"""
        zero_hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        return zero_hidden
