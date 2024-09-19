from math import sqrt
from random import choice
from pathlib import Path
from shutil import rmtree
from functools import partial

from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image

from vae import VQGanVAE

from einops import rearrange

from accelerate import Accelerator

from ema import EMA

import json
from tqdm import tqdm, trange

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

class ImageTextDataset(Dataset):
  def __init__(
      self,
      folder,
      token_folder,
      annotations_path,
      image_size,
      tokenizer = None,
      exts = ['jpg', 'jpeg', 'png']
  ):
    super().__init__()
    self.image_size = image_size
    self.tokenizer = tokenizer
    self.token_folder = Path(token_folder)

    if yes_or_no("Do you want to clear token folder and recompute tokens? (yes/no)"):
      rmtree(str(self.token_folder))
      self.token_folder.mkdir(parents = True, exist_ok = True)
      self.should_compute_tokens = True
    else:
      self.should_compute_tokens = False
      
    
    self.transform = T.Compose([
      T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
      T.Resize((image_size, image_size)),
      # image might not fit the text after flip and crop
      # T.RandomHorizontalFlip(),
      # T.CenterCrop(image_size),
      T.ToTensor()
    ])

    image_paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

    image_annotations = json.load(open(annotations_path))["annotations"]

    image_annotations_keyed = dict()
    for annotation in image_annotations:
      image_id = annotation["image_id"]
      caption = annotation["caption"]
      if image_id not in image_annotations_keyed:
        image_annotations_keyed[image_id] = []
      image_annotations_keyed[image_id].append(caption)

    self.data = []
    for path in tqdm(image_paths):
      image_id = path.name.split(".")[0]
      encoded_path = f'{token_folder}/{image_id}.pt'
      for text in image_annotations_keyed[int(image_id)]:
        image_data = {
          "path": path,
          "encoded_path": encoded_path,
          "texts": text,
        }
        self.data.append(image_data)

      if self.should_compute_tokens:
          with torch.no_grad():
              _, indices, _ = self.tokenizer.encode(self.transform(Image.open(path)).unsqueeze(0).cuda())
              torch.save(indices[0], encoded_path)

    print(f'Found {len(self.data)} training samples at {folder}')

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    path = self.data[index]["path"]
    texts = self.data[index]["texts"]
    encoded_img = torch.load(self.data[index]["encoded_path"])
    img = Image.open(path)
    return encoded_img, self.transform(img), texts
  
class MaskGitTrainer(nn.Module):
  def __init__(
      self,
      maskgit,
      *,
      image_folder,
      token_folder,
      caption_file,
      num_train_steps,
      batch_size,
      image_size = 256,
      lr = 1e-4,
      weight_decay = 0.045,
      grad_accum_every = 1,
      save_results_every = 100,
      save_model_every = 1000,
      results_folder = "./results-maskgit",
      random_split_seed = 42, 
      valid_frac = 0.05,
  ):
    super().__init__()

    self.maskgit = maskgit
    self.register_buffer('steps', torch.Tensor([0]))

    self.accelerator = Accelerator()

    self.num_train_steps = num_train_steps
    self.batch_size = batch_size
    self.grad_accum_every = grad_accum_every

    all_parameters = set(maskgit.parameters())

    self.optim = Adafactor(all_parameters, 
                           weight_decay = weight_decay, 
                           clip_threshold=1., 
                           warmup_init = True, 
                           scale_parameter = False)

    self.ds = ImageTextDataset(image_folder, token_folder, caption_file, image_size, tokenizer = maskgit.vae)

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
      shuffle = True,
      drop_last = True
    )

    self.valid_dl = DataLoader(
      self.valid_ds,
      batch_size = batch_size,
      shuffle = True
    )

    self.maskgit, self.optim, self.dl, self.valid_dl = self.accelerator.prepare(self.maskgit, self.optim, self.dl, self.valid_dl)

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
      model = self.accelerator.get_state_dict(self.maskgit),
      optim = self.optim.state_dict(),
    )
    torch.save(pkg, path)

  def load(self, path):
    path = Path(path)
    assert path.exists()
    pkg = torch.load(path)

    maskgit = self.accelerator.unwrap_model(self.maskgit)
    maskgit.load_state_dict(pkg['model'])

    self.optim.load_state_dict(pkg['optim'])

  @property
  def device(self):
    return self.accelerator.device
  
  def train_step(self):
    device = self.device

    steps = int(self.steps.item())

    self.maskgit.train()

    logs = {}
    for _ in range(self.grad_accum_every):
      encoded_img, _, texts = next(self.dl_iter)
      encoded_img = encoded_img.to(device)

      with self.accelerator.autocast():
        loss = self.maskgit(
          images_or_ids = encoded_img,
          texts = list(texts)
        )
        
        self.accelerator.backward(loss / self.grad_accum_every)

        accum_log(logs, {'loss': loss.item() / self.grad_accum_every})

    self.optim.step()
    self.optim.zero_grad(set_to_none = True)

    print(f"{steps}: transformer loss: {logs['loss']}")

    if steps > 10 and (not (steps % self.save_results_every) or steps == self.num_train_steps - 1):
      
      self.maskgit.eval()
      valid_encoded, valid_images, valid_texts = next(self.valid_dl_iter)
      valid_images = valid_images.to(device)

      output_images = self.maskgit.generate(valid_texts).detach().cpu().float().clamp(0., 1.)
      grid = make_grid(output_images, nrow = 2, normalize = True, value_range = (0, 1))

      save_image(grid, str(self.results_folder / f'{steps}.png'))
      
      with open(str(self.results_folder / f'{steps}.txt'), 'w+') as f:
        f.write(";".join(valid_texts))

      print(f'{steps}: saving to {str(self.results_folder)}')

      self.accelerator.wait_for_everyone()
      if not (steps % self.save_model_every) or steps == self.num_train_steps - 1:
        state_dict = self.accelerator.unwrap_model(self.maskgit).state_dict()
        model_path = str(self.results_folder / f'maskgit.{steps}.pt')
        self.accelerator.save(state_dict, model_path)

        print(f'{steps}: saving model to {str(self.results_folder)}')

    self.steps += 1
    return logs

  def train(self):
    
    for step in trange(self.num_train_steps, position = 0, leave = True):
      if self.steps >= self.num_train_steps:
        break
      logs = self.train_step()

    print('training complete')

