import os
import torch
from config import *
from unityagents import UnityEnvironment
from A3C.A3C_Model import ActorCritic
from utils.SharedAdam import SharedAdam
from torch import multiprocessing
from torch.autograd import Variable

class Env():
    def __init__(self):
        os.environ['OMP_NUM_THREADS'] = '1'
        self.env = UnityEnvironment(file_name=args.unity)
        self.brain_name = self.env.brain_names[0]
        self.shared_model = ActorCritic(33, 4, 1)
        self.env_info = None
        self.optimizer = SharedAdam(self.shared_model.parameters(), lr=args.lr_a3c)
        self.optimizer.share_memory()

    def train(self):
        # Code Reference: https://github.com/pytorch/examples/tree/master/mnist_hogwild

        processes = []

        # process = multiprocessing.Process(target=self.monitor, args=(args.num_processes, args, self.shared_model))
        # process.start()
        # processes.append(process)

        for rank in range(0, args.num_processes):
            process = multiprocessing.Process(target=agent, args=(rank))
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

    def ensure_shared_grads(self, model, shared_model):
        for param, shared_param in zip(model.parameters(), shared_model.parameters()):
            if shared_param.grad is not None: return
            shared_param._grad = param.grad

    def agent(self, rank):

        model = ActorCritic(33, 4, 1)
        model.train()

        self.env_info = self.env.reset(train_mode=True)[self.brain_name]
        state = self.env_info.vector_observations[rank][0]
        episode_length = 0
        done = self.env_info.local_done[rank][0]

        while True:
            episode_length += 1

            model.load_state_dict(self.shared_model.state_dict())

            if done:
                c_last = Variable(torch.zeros(1, 128*6))
                h_last = Variable(torch.zeros(1, 128 * 6))
            else:
                c_last = Variable(c_last)
                h_last = Variable(h_last)

            values, log_probs, rewards, entropies = self.step(state, (h_last, c_last))

    def step(self, state, h_last, c_last):
        value, action, (h_last, c_last) = self.model(state, h_last, c_last)

