import torch.nn as nn
import torch
import torch.nn.functional as F

"""
Imitation learning network
"""


class CNN(nn.Module):

    def __init__(self, history_length=0, n_classes=5):
        super(CNN, self).__init__()
        # TODO : define layers of a convolutional neural network
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        # Calculate the exact flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 96, 96)
            conv_out = self.conv_layers(dummy)
            self.flattened_size = conv_out.numel() // conv_out.shape[0]

        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size, 250),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(250, n_classes),
        )

    def forward(self, x):
        # TODO: compute forward pass
        x = self.conv_layers(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x
