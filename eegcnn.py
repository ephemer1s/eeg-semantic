import torch
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            nn.Conv1d(1, 128, 3, 2, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )
        self.conv2 = torch.nn.Sequential(
            nn.Conv1d(128, 128, 3, 2, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )
        self.conv3 = torch.nn.Sequential(
            nn.Conv1d(128, 64, 3, 2, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )
        self.conv4 = torch.nn.Sequential(
            nn.Conv1d(64, 64, 3, 2, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )
        self.conv5 = torch.nn.Sequential(
            nn.Conv1d(64, 32, 3, 2, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )
        self.conv6 = torch.nn.Sequential(
            nn.Conv1d(32, 32, 3, 2, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )
        self.conv7 = torch.nn.Sequential(
            nn.Conv1d(32, 16, 3, 2, 1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )
        self.conv8 = torch.nn.Sequential(
            nn.Conv1d(16, 16, 3, 2, 1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )
        self.mlp = torch.nn.Linear(3,1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.mlp(x)
        return x
