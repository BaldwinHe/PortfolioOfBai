import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = 0.54
        self.eps = 1.0
        self.eps_decay = 0.9997
        self.eps_min = 0.01
        self.gamma = 0.87
        self.policy_s = np.zeros(self.nA)

    def generate_action_pro(self, state):
        """ Generate action probability based on Q and current state

        Parameters
        ----------
        state: current state

        Returns
        -------
        policy_s： action probability
        """
        self.policy_s = np.ones(self.nA) * (self.eps/self.nA)
        best_a = np.argmax(self.Q[state])
        self.policy_s[best_a] = 1 - self.eps + self.eps/self.nA
        return self.policy_s

    def update_eps(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        if(state in self.Q):
            return np.random.choice(self.nA,p=self.generate_action_pro(state))
        else:
            return np.random.choice(self.nA)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """

        self.Q[state][action] += self.alpha * (reward +
                                               self.gamma * np.sum(self.generate_action_pro(next_state)
                                                                   * self.Q[next_state]) -
                                               self.Q[state][action])