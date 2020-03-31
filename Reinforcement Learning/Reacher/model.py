import torch
from torch import nn
import torch.nn.functional as F

class ActorNet(nn.Module):
    def __init__(self, state_dim, action_dim, max_action_value, fc_channel_base=64):
        super(ActorNet,self).__init__()
        self.fc1 = nn.Linear(state_dim, fc_channel_base)
        self.fc2 = nn.Linear(fc_channel_base, fc_channel_base*2)
        self.fc3 = nn.Linear(fc_channel_base*2, fc_channel_base*4)
        self.fc4 = nn.Linear(fc_channel_base*4, action_dim)
        self.max_action_value = max_action_value

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        out = self.max_action_value * F.tanh(self.fc4(x))
        return out

class CriticNet(nn.Module):
    def __init__(self, state_dim, action_dim,fc_channel_base=64):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, fc_channel_base)
        self.fc2 = nn.Linear(fc_channel_base, fc_channel_base*2)
        self.fc3 = nn.Linear(fc_channel_base*2, fc_channel_base*4)
        self.fc4 = nn.Linear(fc_channel_base*4, 1)

    def forward(self, state, action):
        x = F.relu(self.fc1(torch.cat([state, action], dim=1)))
        x = F.relu((self.fc2(x)))
        x  =F.relu(self.fc3(x))
        out = self.fc4(x)
        return out

