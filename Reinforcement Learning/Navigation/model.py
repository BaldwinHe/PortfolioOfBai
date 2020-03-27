import torch
from torch import nn
import torch.nn.functional as F

class QNetworkFC(nn.Module):
    def __init__(self,in_channels = 37, out_channels = 4, fc1_channels = 64, fc2_channels = 128, fc3_channels = 256):
        super(QNetworkFC, self).__init__()
        self.fc1 = nn.Linear(in_channels, fc1_channels)
        self.fc2 = nn.Linear(fc1_channels, fc2_channels)
        self.fc3 = nn.Linear(fc2_channels, fc3_channels)
        self.fc4 = nn.Linear(fc3_channels,fc2_channels)
        self.fc5 = nn.Linear(fc2_channels,fc1_channels)
        self.fc6 = nn.Linear(fc1_channels, out_channels)

    def forward(self,state):
        x1 = self.fc1(state)
        x = F.relu(x1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x + x1)
        return x

