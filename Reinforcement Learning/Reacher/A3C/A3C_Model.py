import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class ActorCritic(torch.nn.Module):

    def normalized_columns_initializer(weights, std=1.0):
        out = torch.randn(weights.size())
        out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
        return out

    # Xavier initialization
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            weight_shape = list(m.weight.data.size())
            dim_in = weight_shape[1]
            dim_out = weight_shape[0]
            weight_bound = np.sqrt(6. / (dim_in + dim_out))
            m.weight.data.uniform_(-weight_bound, weight_bound)
            m.bias.data.fill_(0)

    def __init__(self, state_dims, actions_dims, max_action_value,base_channel=128):
        super(ActorCritic).__init__()
        self.max_action_value = max_action_value

        # Linear
        self.fc1 = nn.Linear(state_dims,base_channel);
        self.fc2 = nn.Linear(base_channel,base_channel*2)
        self.fc3 = nn.Linear(base_channel*2,base_channel*4)

        # LSTM
        self.lstm = nn.LSTMCell(base_channel*4, base_channel*6)

        self.actor = nn.Linear(base_channel*6, actions_dims)
        self.critic = nn.Linear(base_channel*6, 1)

        self.init_parameters()

        self.train()

    def forward(self, input, h_last, c_last):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        h_last, c_last = self.lstm(x, (h_last, c_last))

        critic_output = self.critic(h_last)
        actor_output = self.actor(h_last)) * self.max_action_value

        return critic_output, actor_output, (h_last, c_last)


    def init_parameters(self):
        self.apply(self.weights_init)

        self.actor.weight.data = self.normalized_columns_initializer(
            self.actor.weight.data, 0.01)

        self.critic.weight.data = self.normalized_columns_initializer(
            self.critic.weight.data, 1.0)

        self.actor.bias.data.fill_(0)
        self.critic.bias.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        self.lstm.bias_ih.data.fill_(0)