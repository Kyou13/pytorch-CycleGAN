import torch
from torch import optim
from torch import nn


def get_optim(params, target):

  assert isinstance(target, nn.Module) or isinstance(target, dict)

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
        params['beta1'], 0.999), weight_decay=params['wd'])
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
    nn.init.normal_(m.weight.data, 0.0, 0.02)
    if hasattr(m, "bias") and m.bias is not None:
      nn.init.constant_(m.bias.data, 0.0)
  elif classname.find('BatchNorm2d') != -1:
    nn.init.normal_(m.weight.data, 1.0, 0.02)
    nn.init.constant_(m.bias.data, 0)
