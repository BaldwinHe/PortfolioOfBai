import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class ActorNet(nn.Module):
    def __init__(self, state_dim, action_dim, max_action_value,seed, fc_channel_base=256):
        super(ActorNet,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_dim, fc_channel_base)
        self.fc2 = nn.Linear(fc_channel_base, fc_channel_base*4)
        self.fc3 = nn.Linear(fc_channel_base*4, fc_channel_base*8)
        self.fc4 = nn.Linear(fc_channel_base*8, action_dim)
        self.max_action_value = max_action_value
        self.reset_parameters()


    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        out = self.max_action_value * F.tanh(self.fc4(x))
        return out

class CriticNet(nn.Module):
    def __init__(self, state_dim, action_dim,seed,fc_channel_base=258):
        super(CriticNet, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_dim + action_dim, fc_channel_base)
        self.fc2 = nn.Linear(fc_channel_base, fc_channel_base*4)
        self.fc3 = nn.Linear(fc_channel_base*4, fc_channel_base*8)
        self.fc4 = nn.Linear(fc_channel_base*8, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        x = torch.cat([state.to(device), action.to(device)], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu((self.fc2(x)))
        x  =F.relu(self.fc3(x))
        out = self.fc4(x)
        return out

