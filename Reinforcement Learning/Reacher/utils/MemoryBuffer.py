import random
import numpy as np
import torch
from collections import namedtuple, deque
from utils.SumTree import SumTree
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

class PrioritizedReplayBuffer():
    alpha = 0.6
    beta = 0.4
    epsilon = 0.00001
    beta_increment_per_sampling = 0.01
    abs_err_upper = 1.0  # clipped abs error

    def __init__(self, buffer_size, batch_size):
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action","reward", "next_state","done"])
        self.memory = SumTree(buffer_size)

    def get_priority(self, reward):
        return (np.abs(reward) + self.epsilon) ** self.alpha

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory"""
        # priority = self.get_priority(reward)
        e = self.experience(state, action, reward, next_state, done)
        max_priority = np.max(self.memory.tree_data[-self.memory.buffer_size:])
        if(max_priority == 0):
            max_priority = self.abs_err_upper
        self.memory.add(max_priority, e)

    def update(self, idx, error):
        error += self.epsilon
        clipped_errors = np.minimum(error, self.abs_err_upper)
        priority = clipped_errors ** self.alpha
        self.memory.update(idx, priority)

    def sample(self):
        experiences = []
        priorities = []
        indexs = []
        priority_segment = self.memory.total / self.batch_size

        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)

        for i in range(self.batch_size):
            low_bound, high_bound = i * priority_segment, (i + 1) * priority_segment

            value = np.random.uniform(low_bound, high_bound)
            leaf_index, priority, data = self.memory.get_leaf(value)

            indexs.append(leaf_index)
            priorities.append(priority)
            experiences.append(data)

        priorities = priorities / self.memory.total
        weights = np.power(priorities * self.memory.N, -self.beta)
        weights /= weights.max()

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones, indexs, weights)

    def __len__(self):
        return self.memory.N


class ReplayBuffer():
    """Fixed-size buffer to store experience tuples"""
    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object
        :param buffer_size(int): maximum size of buffer
        :param batch_size(int: size of each training batch
        """

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action","reward", "next_state","done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory"""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experience from memory"""

        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""

        return len(self.memory)