import os
import torch
import torchvision
import torch.nn as nn
import click
import datetime
import numpy as np
import itertools
from torchvision import transforms
from torchvision.utils import save_image
from src import models, utils
import matplotlib.pyplot as plt
from src.datasets import ImageDataset
from PIL import Image

params = {
    'seed': 123456789,
    'epochs': 200,
    'batch_size': 1,
    'optimizer': 'adam',
    'lr': 2e-4,
    'beta1': 0.5,
    'beta2': 0.999,
    'wd': 0,
    'img_height': 256,
    'img_width': 256,
    'channels': 3,
    'decay_epoch': 100,
    'sample_interval': 100,
    'checkpoint_interval': 10,
    'n_residual_blocks':9,
    'lambda_cyc': 10.0,
    'lambda_id': 5.0,
}


@click.group()
def main():
  np.random.seed(params['seed'])
  torch.manual_seed(params['seed'])
  torch.cuda.manual_seed_all(params['seed'])
  torch.backends.cudnn.benchmark = True


@main.command()
@click.option('--dataset', type=str, default='monet2photo')
def train(dataset):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  sample_dir = os.path.join('samples', dataset)
  weights_dir = os.path.join('weights', dataset)
  os.makedirs(sample_dir, exist_ok=True)
  os.makedirs(weights_dir, exist_ok=True)


  transforms_ = [
      transforms.Resize(int(params['img_height'] * 1.12), Image.BICUBIC),
      transforms.RandomCrop((params['img_height'], params['img_width'])),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
  ]

  # DataLoader
  data_loader = torch.utils.data.DataLoader(
      ImageDataset(os.path.join('data', dataset), transforms_=transforms_, unaligned=True),
      batch_size=params['batch_size'],
      shuffle=True
  )

  val_data_loader = torch.utils.data.DataLoader(
      ImageDataset(os.path.join('data', dataset), transforms_=transforms_, unaligned=True, mode='test'),
      batch_size=5,
      shuffle=True
  )

  # Models
  D_A = models.Discriminator(params['channels'])
  D_B = models.Discriminator(params['channels'])
  G_AB = models.Generator(params['channels'], params['n_residual_blocks'])
  G_BA = models.Generator(params['channels'], params['n_residual_blocks'])

  D_A = D_A.to(device)
  D_B = D_B.to(device)
  G_AB = G_AB.to(device)
  G_BA = G_BA.to(device)

  D_A.apply(utils.weights_init)
  D_B.apply(utils.weights_init)
  G_AB.apply(utils.weights_init)
  G_BA.apply(utils.weights_init)

  # Losses
  # TODO: cuda必要？
  criterion_GAN = nn.MSELoss()
  criterion_cycle = nn.L1Loss()
  criterion_identity = nn.L1Loss()

  # Optimizer
  optimizer_G = utils.get_optim(
      params,
      itertools.chain(G_AB.parameters(),G_BA.parameters()), 
  )
  optimizer_D_A = utils.get_optim(params, D_A)
  optimizer_D_B = utils.get_optim(params, D_B)

  # schedulers
  lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
      optimizer_G, lr_lambda=utils.LambdaLR(params['epochs'], decay_start_epoch=params['decay_epoch']).step
  )
  lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
      optimizer_D_A, lr_lambda=utils.LambdaLR(params['epochs'], decay_start_epoch=params['decay_epoch']).step
  )
  lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
      optimizer_D_B
      , lr_lambda=utils.LambdaLR(params['epochs'], decay_start_epoch=params['decay_epoch']).step
  )
  # Buffers of previously generated samples
  fake_A_buffer = utils.ReplayBuffer()
  fake_B_buffer = utils.ReplayBuffer()

  def sample_images(epochs):
      """Saves a generated sample from the test set"""
      imgs = next(iter(val_dataloader))
      G_AB.eval()
      G_BA.eval()
      with torch.no_grad():
        real_A = imgs["A"].to(device)
        fake_B = G_AB(real_A)
        real_B = imgs["B"].to(device)
        fake_A = G_BA(real_B)
      # Arange images along x-axis
      real_A = make_grid(real_A, nrow=5, normalize=True)
      real_B = make_grid(real_B, nrow=5, normalize=True)
      fake_A = make_grid(fake_A, nrow=5, normalize=True)
      fake_B = make_grid(fake_B, nrow=5, normalize=True)
      # Arange images along y-axis
      image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
      save_image(image_grid, os.path.join(samples_dir,"fake_images-%s.png" % (epochs)), normalize=False)

  losses_D = []
  losses_G = []
  total_step = len(data_loader)
  for epoch in range(params['epochs']):
    for i, images in enumerate(data_loader):
      real_A = images['A'].to(device)
      real_B = images['B'].to(device)

      b_size = real_A.size(0)

      # TODO: require_grad, G.train(), 自動化できないか
      real_labels = torch.ones((b_size, 1, 16, 16)).to(device)
      fake_labels = torch.zeros((b_size, 1, 16, 16)).to(device)

      # Train Generator
      optimizer_G.zero_grad()

      # GAN loss
      fake_B = G_AB(real_A)
      loss_GAN_AB = criterion_GAN(D_B(fake_B), real_labels)
      fake_A = G_BA(real_B)
      loss_GAN_BA = criterion_GAN(D_A(fake_A), real_labels)
      
      loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

      # Cycle loss
      recov_A = G_BA(fake_B)
      loss_cycle_A = criterion_cycle(recov_A, real_A)
      recov_B = G_AB(fake_A)
      loss_cycle_B = criterion_cycle(recov_B, real_B)

      loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

      # Total loss
      loss_G = loss_GAN + params['lambda_cyc'] * loss_cycle

      loss_G.backward()
      optimizer_G.step()

      # Train discriminator A
      optimizer_D_A.zero_grad()

      loss_real_A = criterion_GAN(D_A(real_A), real_labels)
      fake_A_ = fake_A_buffer.push_and_pop(fake_A)
      loss_fake_A = criterion_GAN(D_A(fake_A_.detach()), fake_labels)
      loss_D_A = (loss_real_A + loss_fake_A) / 2
      loss_D_A.backward()
      optimizer_D_A.step()

      # Train discriminator B
      optimizer_D_B.zero_grad()

      loss_real_B = criterion_GAN(D_B(real_B), real_labels)
      fake_B_ = fake_A_buffer.push_and_pop(fake_B)
      loss_fake_B = criterion_GAN(D_B(fake_B_.detach()), fake_labels)
      loss_D_B = (loss_real_B + loss_fake_B) / 2
      loss_D_B.backward()
      optimizer_D_B.step()

      loss_D = (loss_D_A + loss_D_B) / 2


      print('Epoch [{}/{}], step [{}/{}], loss_D: {:.4f}, loss_G: {:.4f}, D_A(x): {:.2f}, D_A(G_BA(z)): {:.2f}, D_B(x): {:.2f}, D_B(G_AB(z)): {:.2f}'
            .format(epoch, params['epochs'], i + 1, total_step, loss_D.item(), loss_G.item(),
                    loss_real_A.mean().item(), loss_real_B.mean().item(),
                    loss_real_B.mean().item(), loss_real_B.mean().item()))  

      losses_G.append(loss_G.item())
      losses_D.append(loss_D.item())

    if epoch % params['checkpoint_interval'] == 0:
      torch.save(G_AB.state_dict(), os.path.join(weights_dir, 'G_AB.ckpt'))
      torch.save(G_BA.state_dict(), os.path.join(weights_dir, 'G_BA.ckpt'))
      torch.save(D_A.state_dict(), os.path.join(weights_dir, 'D_A.ckpt'))
      torch.save(D_B.state_dict(), os.path.join(weights_dir, 'D_B.ckpt'))

    if (epoch + 1) == 1:
      save_image(utils.denorm(images), os.path.join(
          sample_dir, 'real_images.png'))
    sample_images(epoch + 1)


  plt.figure(figsize=(10, 5))
  plt.title("Generator and Discriminator Loss During Training")
  plt.plot(g_losses, label="Generator")
  plt.plot(d_losses, label="Discriminator")
  plt.xlabel("iterations")
  plt.ylabel("Loss")
  plt.legend()
  plt.savefig(os.path.join(sample_dir, 'loss.png'))


@main.command()
@click.option('--dataset', type=str, default='mnist')
def generate(dataset):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  sample_dir = os.path.join('samples', dataset)
  weights_dir = os.path.join('weights', dataset)
  os.makedirs(sample_dir, exist_ok=True)

  G = models.Generator(params['nz'], params['ngf'], params['nc'])
  G.load_state_dict(torch.load(os.path.join(weights_dir, 'G.ckpt')))
  G.eval()
  G = G.to(device)

  with torch.no_grad():
    z = torch.randn(params['batch_size'], params['nz'], 1, 1).to(device)
    fake_images = G(z)

  dt_now = datetime.datetime.now()
  now_str = dt_now.strftime('%y%m%d%H%M%S')
  save_image(utils.denorm(fake_images), os.path.join(
      sample_dir, 'fake_images_{}.png'.format(now_str)))
  print('Saved Image ' + os.path.join(sample_dir,
                                      'fake_images_{}.png'.format(now_str)))


if __name__ == '__main__':
  main()
