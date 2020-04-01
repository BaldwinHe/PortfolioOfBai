import torch
import torch.nn.functional as F
from torch.autograd import Variable

from A3C.A3C_Model import ActorCritic

# ensure shared model with gradient
def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None: return
        shared_param._grad = param.grad


