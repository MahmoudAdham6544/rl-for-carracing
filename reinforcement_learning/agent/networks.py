import torch.nn as nn
import torch
import torch.nn.functional as F


"""
CartPole network
"""


class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=400):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


"""
Carracing network
"""


class CNN(nn.Module):

    def __init__(self, input_shape=(1, 96, 96), n_classes=5):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 96, 96)
            conv_out = self.conv_layers(dummy)
            self.flattened_size = conv_out.numel() // conv_out.shape[0]

        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size, 250), nn.ReLU(), nn.Linear(250, n_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x
