import collections

import numpy as np
from unityagents import UnityEnvironment
from tqdm import tqdm
from agent import *
from config import *


def train(agent, env, n_episodes=300000, max_t=100000):
    # get the default brain
    brain_name = env.brain_names[0]

    total_scores = []  # list containing scores from each episode
    scores_window = collections.deque(maxlen=100)  # last 100 scores

    for i_episode in range(1, 1 + n_episodes):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        agent.reset()
        scores = np.zeros(12)

        for i in range(max_t):
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            agent.step(states, actions, rewards, next_states, dones)
            if i % 20 == 0:
                agent.sample_and_learn()
            scores += rewards
            states = next_states
            if (np.any(dones)):
                break
        scores_window.append(scores)
        total_scores.append(scores)
        agent.save('actor_checkpoint.pth','critic_checkpoint.pth')

        print('\rEpisode {}\tMean Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tMean Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 2000.0:
            print('\nEnvironment solved in {:d} episodes!\tMean Score: {:.2f}'.format(i_episode - 100,
                                                                                      np.mean(scores_window)))
            np.save('scores_{}.npy'.format( i_episode - 100), np.array(scores))
            agent.save('actor_checkpoint_{}.pth'.format(i_episode - 100),'critic_checkpoint_{}.pth'.format(i_episode - 100))
            break
    print('TRAIN DONE!')

if __name__ == '__main__':
    env = UnityEnvironment(file_name=args.unity)

    agent = Agent(129,20,1,args.seed)

    train(agent, env)