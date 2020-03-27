import numpy as np
import random
from collections import namedtuple, deque

from model import *

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 1e-3
LR = 5e-5
UPDATE_EVERY = 4

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

class AgentFC():
    """Interacts with ans learns from the environment"""

    def __init__(self, state_size, action_size):
        """Initialize an Agent object

        :param state_size(int): dimension of each state
        :param action_size(int): dimension of each action
        """

        self.state_size = state_size
        self.action_size = action_size

        self.qnet_local = QNetworkFC(state_size, action_size).to(device)
        self.qnet_target = QNetworkFC(state_size, action_size).to(device)

        self.optimizer = optim.Adam(self.qnet_local.parameters(), lr=LR)
        self.criterion = torch.nn.MSELoss()

        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)

        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EXERY time steps
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if(self.t_step == 0):
            # If enough sample are available in memory, get random subset and Learn
            if (len(self.memory) > BATCH_SIZE):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """ Return actions for given state per current policy

        :param state(array_like): current state
        :param eps(float): epsilon, for epsilon-greedy action selection
        :return: selected action
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnet_local.eval()
        with torch.no_grad():
            action_values = self.qnet_local(state)
        self.qnet_local.train()

        # Epsilon-greedy action selection
        if( random.random() > eps):
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """ Update actions for given state per current policy

        :param experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
        :param gamma (float): discount factor
        """

        states, actions, rewards, next_states, dones = experiences    

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnet_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnet_local(states).gather(1, actions)

        # Compute loss
        loss = self.criterion(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.soft_update(self.qnet_local, self.qnet_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """ Soft update model parameters
        Q_target = tau * Q_local + (1 - tau) * Q_target

        :param local_model (Pytorch model): weights will be copied from
        :param target_model (Pytorch model): weights will be copied to
        :param tau (float): interpolation parameter
        """

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)


class ReplayBuffer():
    """Fixed-size buffer to store experience tuples"""
    def __init__(self, action_size, buffer_size, batch_size):
        """Initialize a ReplayBuffer object

        :param action_size(int): dimension of each action
        :param buffer_size(int): maximum size of buffer
        :param batch_size(int: size of each training batch
        """

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experence = namedtuple("Experience", field_names=["state", "action","reward", "next_state","done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory"""
        e = self.experence(state, action, reward, next_state, done)
        self.memory.append(e)

    def map_reward(self, reward):
        """modify reward to get yello banana quickly"""
        if reward > 0:
            return 2.0
        elif reward < 0:
            return -1.5
        else :
            return 0

    def sample(self):
        """Randomly sample a batch of experience from memory"""

        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""

        return len(self.memory)
