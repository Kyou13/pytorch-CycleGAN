import torch
from torch import nn

class ResidualBlock(nn.Module):
  def __init__(self, in_features):
    super(ResidualBlock, self).__init__()

    self.block = nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_features, in_features, 3),
        nn.InstanceNorm2d(in_features),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_features, in_features, 3),
        nn.InstanceNorm2d(in_features)
    )

  def forward(self, x):
    return x + self.block(x)


class Generator(nn.Module):
  def __init__(self, nc, num_residual_blocks):
    super(Generator, self).__init__()
    out_features = 64
    model = [
        nn.ReflectionPad2d(nc),
        nn.Conv2d(nc, out_feature, 7),
        nn.InstanceNorm2d(out_features),
        nn.ReLU(inplace=True)
    ]
    in_features = out_features

    for _ in range(2):
      out_features *= 2
      model += [
          nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
          nn.InstanceNorm2d(out_features),
          nn.ReLU(inplace=True)
      ]
      in_features = out_features

    # Reidual blocks
    for _ in range(num_residual_blocks):
      model += [ResidualBlock(out_features)

    for _ in range(2):
      out_features //= 2
      model += [
        nn.Upsample(scle_factor=2),
        nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
        nn.InstanceNorm2d(out_features),
        nn.ReLU(inplace=True)
      ]
      in_features = out_features

    model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]

    self.model = nn.Sequential(*model)

  def forward(self, x):
    return self.model(x)


class Discriminator(nn.Module):
  def __init__(self, nc):
    super(Discriminator, self).__init__()

    def discriminator_block(in_filters, out_filters, normalize=True):
      layers = [nn.Conv2d(in_filters, out_filters, normalize=True)]
      if normalize:
        layers.append(nn.InstanceNorm2d(out_filters))
      layers.append(nn.LeakyReLU(0.2, inplace=True))
      return layers

    self.model = nn.Sequential(
        *discriminator_block(nc, 64, normalize=False),
        *discriminator_block(64, 128),
        *discriminator_block(128, 256),
        *discriminator_block(256, 512),
        nn.ZeroPad2d((1, 0, 1, 0)),
        nn.Conv2d(512, 1, 4, padding=1)
        )

  def forward(self, x):
    return self.model(x)
