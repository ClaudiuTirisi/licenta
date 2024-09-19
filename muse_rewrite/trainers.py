from math import sqrt
from random import choice
from pathlib import Path
from shutil import rmtree
from functools import partial

from PIL import Image

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image

from vae import VQGanVAE

from einops import rearrange

from accelerate import Accelerator

from ema import EMA

def yes_or_no(question):
  answer = input(f'{question} (y/n)')
  return answer.lower() in ['yes', 'y']

def cycle(dl):
  while True:
    for data in dl:
      yield data

def accum_log(log, new_logs):
  for key, new_value in new_logs.items():
    old_value = log.get(key, 0.)
    log[key] = old_value + new_value

  return log

class ImageDataset(Dataset):
  def __init__(
      self,
      folder,
      image_size,
      exts = ['jpg', 'jpeg', 'png']
  ):
    super().__init__()
    self.folder = folder
    self.image_size = image_size
    self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

    print(f'Found {len(self.paths)} training samples at {folder}')

    self.transform = T.Compose([
      T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
      T.Resize(image_size),
      T.RandomHorizontalFlip(),
      T.CenterCrop(image_size),
      T.ToTensor()
    ])

  def __len__(self):
    return len(self.paths)

  def __getitem__(self, index):
    path = self.paths[index]
    img = Image.open(path)
    return self.transform(img)

class VQGanVAETrainer(nn.Module):
  def __init__(
      self,
      vae,
      *,
      folder,
      valid_folder = None,
      num_train_steps,
      batch_size,
      image_size,
      lr = 3e-4,
      grad_accum_every = 1,
      save_results_every = 100,
      save_model_every = 1000,
      results_folder = "./results",
      valid_frac = 0.05,
      random_split_seed = 42,
      use_ema = True,
      ema_beta = 0.995,
      ema_update_after_step = 0,
      ema_update_every = 1,
      apply_grad_penalty_every = 4,   
  ):
    super().__init__()

    self.vae = vae
    self.register_buffer('steps', torch.Tensor([0]))

    self.accelerator = Accelerator()

    self.num_train_steps = num_train_steps
    self.batch_size = batch_size
    self.grad_accum_every = grad_accum_every

    all_parameters = set(vae.parameters())
    discr_parameters = set(vae.discr.parameters())
    vae_parameters = all_parameters - discr_parameters

    self.vae_parameters = vae_parameters

    self.optim = Adam(vae_parameters, lr = lr)
    self.discr_optim = Adam(discr_parameters, lr = lr)

    self.ds = ImageDataset(folder, image_size)

    if valid_frac >0:
      train_size = int((1 - valid_frac) * len(self.ds))
      valid_size = len(self.ds) - train_size
      self.ds, self.valid_ds = random_split(self.ds, [train_size, valid_size], generator = torch.Generator().manual_seed(random_split_seed))
      print(f'Split dataset into {len(self.ds)} samples for training and {len(self.valid_ds)} samples for validating')
    else:
      self.valid_ds = self.ds
      print(f'Training with shared training and validation dataset of {len(self.ds)} samples')

    self.dl = DataLoader(
      self.ds,
      batch_size = batch_size,
      shuffle = True
    )

    self.valid_dl = DataLoader(
      self.valid_ds,
      batch_size = batch_size,
      shuffle = True
    )

    self.vae, self.optim, self.discr_optim, self.dl, self.valid_dl = self.accelerator.prepare(self.vae, self.optim, self.discr_optim, self.dl, self.valid_dl)

    self.use_ema = use_ema
    if use_ema:
      self.ema_vae = EMA(vae, update_after_step = ema_update_after_step, update_every = ema_update_every)
      self.ema_vae = self.accelerator.prepare(self.ema_vae)

    self.dl_iter = cycle(self.dl)
    self.valid_dl_iter = cycle(self.valid_dl)

    self.save_model_every = save_model_every
    self.save_results_every = save_results_every

    self.apply_grad_penalty_every = apply_grad_penalty_every

    self.results_folder = Path(results_folder)

    if len([*self.results_folder.glob("**/*")]) > 0 and yes_or_no("Do you want to clear previous experiment checkpoints and results?"):
      rmtree(str(self.results_folder))

    self.results_folder.mkdir(parents = True, exist_ok = True)

  def save(self, path):
    pkg = dict(
      model = self.accelerator.get_state_dict(self.vae),
      optim = self.optim.state_dict(),
      discr_optim = self.discr_optim.state_dict()
    )
    torch.save(pkg, path)

  def load(self, path):
    path = Path(path)
    assert path.exists()
    pkg = torch.load(path)

    vae = self.accelerator.unwrap_model(self.vae)
    vae.load_state_dict(pkg['model'])

    self.optim.load_state_dict(pkg['optim'])
    self.discr_optim.load_state_dict(pkg['discr_optim'])

  @property
  def device(self):
    return self.accelerator.device
  
  def train_step(self):
    device = self.device

    steps = int(self.steps.item())
    apply_grad_penalty = not (steps % self.apply_grad_penalty_every)

    self.vae.train()
    discr = self.vae.discr
    
    if self.use_ema:
      ema_vae = self.ema_vae

    if self.use_ema:
      ema_vae = self.ema_vae

    logs = {}
    for _ in range(self.grad_accum_every):
      img = next(self.dl_iter)
      img = img.to(device)

      with self.accelerator.autocast():
        loss = self.vae(
          img,
          add_gradient_penalty = apply_grad_penalty,
          return_loss = True
        )
        
        self.accelerator.backward(loss / self.grad_accum_every)

        accum_log(logs, {'loss': loss.item() / self.grad_accum_every})

    self.optim.step()
    self.optim.zero_grad()

    self.discr_optim.zero_grad()
    for _ in range(self.grad_accum_every):
      img = next(self.dl_iter)
      img = img.to(device)

      loss = self.vae(img, return_discr_loss = True)
      self.accelerator.backward(loss / self.grad_accum_every)

      accum_log(logs, {'discr_loss': loss.item() / self.grad_accum_every})

    self.discr_optim.step()

    print(f"{steps}: vae loss: {logs['loss']} - discr loss: {logs['discr_loss']}")

    if self.use_ema:
      ema_vae.update()

    if not (steps % self.save_results_every) or steps == self.num_train_steps - 1:
      vaes_to_evaluate = [(self.vae, str(steps))]

      if self.use_ema:
        vaes_to_evaluate.append((ema_vae.ema_model, f'{steps}.ema'))

      for model, filename in vaes_to_evaluate:
        model.eval()

        valid_data = next(self.valid_dl_iter)
        valid_data = valid_data.to(device)

        recons = model(valid_data, return_recons = True)

        imgs_and_recons = torch.stack((valid_data, recons), dim = 0)
        imgs_and_recons = rearrange(imgs_and_recons, 'r b ... -> (b r) ...')

        imgs_and_recons = imgs_and_recons.detach().cpu().float().clamp(0., 1.)
        grid = make_grid(imgs_and_recons, nrow = 2, normalize = True, value_range = (0, 1))

        logs['reconstructions'] = grid

        save_image(grid, str(self.results_folder / f'{filename}.png'))

      print(f'{steps}: saving to {str(self.results_folder)}')

      self.accelerator.wait_for_everyone()
      if not (steps % self.save_model_every) or steps == self.num_train_steps - 1:
        state_dict = self.accelerator.unwrap_model(self.vae).state_dict()
        model_path = str(self.results_folder / f'vae.{steps}.pt')
        self.accelerator.save(state_dict, model_path)

        if self.use_ema:
          ema_state_dict = self.accelerator.unwrap_model(self.ema_vae).state_dict()
          model_path = str(self.results_folder / f'vae.{steps}.ema.pt')
          self.accelerator.save(ema_state_dict, model_path)

        print(f'{steps}: saving model to {str(self.results_folder)}')

    self.steps += 1
    return logs

  def train(self):
    
    while self.steps < self.num_train_steps:
      logs = self.train_step()

    print('training complete')

