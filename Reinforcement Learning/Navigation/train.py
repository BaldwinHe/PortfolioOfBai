import collections
import torch
from agent import *
from unityagents import UnityEnvironment
import numpy as np
from config import *

def train(agent, env ,net='FC', n_episodes=30000, max_t=10000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """ Deep Q-Learning.
    :param agent (Agent) : learning agent
    :param env (Environment) : enviroment
    :param net (str) : network type
    :param n_episodes (int): maximum number of training episodes
    :param max_t (int): maximum number of timesteps per episode
    :param eps_start (float): starting value of epsilon, for epsilon-greedy action selection
    :param eps_end (float): minimum value of epsilon
    :param eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # number of actions
    action_size = brain.vector_action_space_size

    scores = []  # list containing scores from each episode
    scores_window = collections.deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon

    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0

        for t in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            score += reward
            state = next_state
            if (done):
                break

        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay*eps)
        print('\rEpisode {}\tMean Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            np.save('scores_{}_{}.npy'.format(net, i_episode - 100),np.array(scores))
            torch.save(agent.qnet_local.state_dict(), 'checkpoint{}_{}.pth'.format(net, i_episode - 100))
            print('\rEpisode {}\tMean Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 15.0:
            print('\nEnvironment solved in {:d} episodes!\tMean Score: {:.2f}'.format(i_episode - 100,
                                                                          np.mean(scores_window)))
            np.save('scores_{}_{}.npy'.format(net, i_episode - 100),np.array(scores))
            torch.save(agent.qnet_local.state_dict(), 'checkpoint_s_{}_{}.pth'.format(net, i_episode - 100))
            break
    print('TRAIN DONE!')


if __name__ == '__main__':
    env = UnityEnvironment(file_name=args.unity)

    agent = AgentFC(37, 4)

    train(agent, env, 'FC')
