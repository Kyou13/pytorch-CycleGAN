import torch
from torch import optim
from torch import nn
import random
import time
import datetime
import sys
import numpy as np
import itertools


def get_optim(params, target):

  assert isinstance(target, nn.Module) or isinstance(target, dict) or isinstance(target, itertools.chain)

  if isinstance(target, nn.Module):
    target = target.parameters()

  if params['optimizer'] == 'sgd':
    optimizer = optim.SGD(target, params['lr'], weight_decay=params['wd'])
  elif params['optimizer'] == 'momentum':
    optimizer = optim.SGD(
        target, params['lr'], momentum=0.9, weight_decay=params['wd'])
  elif params['optimizer'] == 'nesterov':
    optimizer = optim.SGD(target, params['lr'], momentum=0.9,
                          weight_decay=params['wd'], nesterov=True)
  elif params['optimizer'] == 'adam':
    optimizer = optim.Adam(target, params['lr'], betas=(
        params['beta1'], params['beta2']), weight_decay=params['wd'])
  elif params['optimizer'] == 'amsgrad':
    optimizer = optim.Adam(
        target, params['lr'], weight_decay=params['wd'], amsgrad=True)
  elif params['optimizer'] == 'rmsprop':
    optimizer = optim.RMSprop(target, params['lr'], weight_decay=params['wd'])
  else:
    raise ValueError

  return optimizer


def denorm(x):
  out = (x + 1) / 2
  return out.clamp(0, 1)


# args: torch.nn.modules
def weights_init(m):
  classname = m.__class__.__name__  # str
  if classname.find('Conv') != -1:
    # args: torch.Tensor
    # inplaceで動作するため`_`がついている
    # 正規分布
    nn.init.normal_(m.weight.data, 0.0, 0.02)
    if hasattr(m, "bias") and m.bias is not None:
      # 定数
      nn.init.constant_(m.bias.data, 0.0)
  elif classname.find('BatchNorm2d') != -1:
    nn.init.normal_(m.weight.data, 1.0, 0.02)
    nn.init.constant_(m.bias.data, 0)

class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)


class LambdaLR:
    def __init__(self, n_epochs, decay_start_epoch, offset=0):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)
