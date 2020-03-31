import sys

import torch
from agent import *
from config import *
from unityagents import UnityEnvironment

if __name__ == '__main__':
    agent = Agent(33, 4, 1)
    agent.load(args.actor, args.critic)
    env = UnityEnvironment(file_name=args.unity)

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    state = env_info.vector_observations[0]  # get the current state
    score = 0  # initialize the score
    t_max = 1000
    while t_max > 0:
        action = agent.act(state) # select an action
        env_info = env.step(action)[brain_name]  # send the action to the environment
        next_state = env_info.vector_observations[0]  # get the next state
        reward = env_info.rewards[0]  # get the reward
        done = env_info.local_done[0]  # see if episode has finished
        score += reward  # update the score
        state = next_state  # roll over the state to next time step
        print("{}-Score: {}".format(t_max, score))
        sys.stdout.flush()
        t_max -= 1
        if done:  # exit loop if episode finished
            break