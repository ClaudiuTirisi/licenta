"""
Parameters:
 - folder_path = path to folder of .pt and .txt files containing image tokens and image captions
 - 
"""
from torch.profiler import profile, record_function, ProfilerActivity
from custom_datasets import ImageTokensTextDataset

import sys
sys.path.append("..")
sys.path.append("external")

from pathlib import Path
from shutil import rmtree

import torch
from torch import nn
from torch.utils.data import DataLoader

from torchvision.utils import make_grid, save_image

from accelerate import Accelerator

from tqdm import trange

from transformers import Adafactor

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

def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

  
class LowResTrainer(nn.Module):
  def __init__(
      self,
      masked_model,
      vae_model,
      train_folder,
      valid_folder,
      num_train_steps,
      batch_size,
      fmap_size = 256,
      lr = 1e-4,
      weight_decay = 0.045,
      grad_accum_every = 1,
      save_results_every = 100,
      save_model_every = 1000,
      results_folder = "./results-masked",
  ):
    super().__init__()

    self.masked_model = masked_model
    self.fmap_size = fmap_size

    print(f'Training model with {count_parameters(self.masked_model)} parameters')

    self.vae_model = vae_model
    self.register_buffer('steps', torch.Tensor([0]))

    self.accelerator = Accelerator()

    self.num_train_steps = num_train_steps
    self.batch_size = batch_size
    self.grad_accum_every = grad_accum_every

    all_parameters = set(masked_model.parameters())

    self.optim = Adafactor(all_parameters, 
                           weight_decay = weight_decay, 
                           clip_threshold=1., 
                           warmup_init = True, 
                           scale_parameter = False)

    self.ds = ImageTokensTextDataset(train_folder)
    self.valid_ds = ImageTokensTextDataset(valid_folder)

    print(f'Training with {len(self.ds)} training samples and {len(self.valid_ds)} validation samples')

    self.dl = DataLoader(
      self.ds,
      batch_size = batch_size,
      drop_last = True,
    )

    self.valid_dl = DataLoader(
      self.valid_ds,
      batch_size = batch_size,
      drop_last = True,
    )

    self.masked_model, self.optim, self.dl, self.valid_dl = self.accelerator.prepare(self.masked_model, self.optim, self.dl, self.valid_dl)

    self.vae_model.eval()

    self.dl_iter = cycle(self.dl)
    self.valid_dl_iter = cycle(self.valid_dl)

    self.save_model_every = save_model_every
    self.save_results_every = save_results_every

    self.results_folder = Path(results_folder)

    if len([*self.results_folder.glob("**/*")]) > 0 and yes_or_no("Do you want to clear previous experiment checkpoints and results?"):
      rmtree(str(self.results_folder))

    self.results_folder.mkdir(parents = True, exist_ok = True)

  def save(self, path):
    pkg = dict(
      model = self.accelerator.get_state_dict(self.masked_model),
      optim = self.optim.state_dict(),
    )
    torch.save(pkg, path)

  def load(self, path):
    path = Path(path)
    assert path.exists()
    pkg = torch.load(path)

    masked_model = self.accelerator.unwrap_model(self.masked_model)
    masked_model.load_state_dict(pkg['model'])

    self.optim.load_state_dict(pkg['optim'])

  @property
  def device(self):
    return self.accelerator.device
  
  def train_step(self):
        device = self.device

        steps = int(self.steps.item())

        self.masked_model.train()

        logs = {}
        for _ in range(self.grad_accum_every):
          indices, texts = next(self.dl_iter)
          indices = indices.to(device)

          with self.accelerator.autocast():
            loss = self.masked_model(
              images_or_ids = indices,
              texts = list(texts)
            )
            
            self.accelerator.backward(loss / self.grad_accum_every)

            accum_log(logs, {'loss': loss.item() / self.grad_accum_every})

        self.optim.step()
        self.optim.zero_grad(set_to_none = True)

        print(f"{steps}: Cross Entropy loss: {logs['loss']}")

        if (not (steps % self.save_results_every) or steps == self.num_train_steps - 1):
          
          self.masked_model.eval()
          valid_indices, valid_texts = next(self.valid_dl_iter)
          valid_indices = valid_indices.to(device)

          with torch.no_grad():
            output_image_ids = self.masked_model.generate(list(valid_texts), fmap_size = self.fmap_size)
            output_images = self.vae_model.decode_from_ids(output_image_ids, self.batch_size, self.fmap_size).detach().cpu().float().clamp(0, 1)

          grid = make_grid(output_images, nrow = 2, normalize = True, value_range = (0, 1))

          save_image(grid, str(self.results_folder / f'{steps}.png'))
          
          with open(str(self.results_folder / f'{steps}.txt'), 'w+') as f:
            f.write(";".join(valid_texts))

          print(f'{steps}: saving to {str(self.results_folder)}')

          self.accelerator.wait_for_everyone()
          if not (steps % self.save_model_every) or steps == self.num_train_steps - 1:
            state_dict = self.accelerator.unwrap_model(self.masked_model).state_dict()
            model_path = str(self.results_folder / f'maskgit.{steps}.pt')
            self.accelerator.save(state_dict, model_path)

            print(f'{steps}: saving model to {str(self.results_folder)}')

        self.steps += 1

  def train(self):
    
    for step in trange(self.num_train_steps, position = 0, leave = True):
      #with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
      #  with record_function("model_inference"):
          logs = self.train_step()

      #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    print('training complete')

