import sys

from unityagents import UnityEnvironment
import numpy as np
from agent import *
from config import *

if __name__ == '__main__':
    agent = Agent(33,4,1,args.seed)
    agent.load(args.actor, args.critic)
    env = UnityEnvironment(file_name=args.unity)

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    states = env_info.vector_observations
    scores = 0  # initialize the score
    t_max = 1000
    while t_max > 0:
        actions = agent.act(states) # select an action
        env_info = env.step(actions)[brain_name]  # send the action to the environment
        next_states = env_info.vector_observations  # get the next state
        rewards = env_info.rewards[0]  # get the reward
        dones = env_info.local_done  # see if episode has finished
        scores += rewards  # update the score
        states = next_states  # roll over the state to next time step
        print("{}-Score: {}".format(t_max, np.mean(scores)))
        sys.stdout.flush()
        t_max -= 1
