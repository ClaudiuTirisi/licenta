"""
https://github.com/Taited/clip-score

Calculates the CLIP Scores

The CLIP model is a contrasitively learned language-image model. There is
an image encoder and a text encoder. It is believed that the CLIP model could 
measure the similarity of cross modalities. Please find more information from 
https://github.com/openai/CLIP.

The CLIP Score measures the Cosine Similarity between two embedded features.
This repository utilizes the pretrained CLIP Model to calculate 
the mean average of cosine similarities. 

See --help to see further details.

Code apapted from https://github.com/mseitzer/pytorch-fid and https://github.com/openai/CLIP.

Copyright 2023 The Hong Kong Polytechnic University

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import os.path as osp
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import clip
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--batch-size", type = int, default = 50, help ='Batch size to use')

parser.add_argument("--clip-model", type = str, default = "ViT-B/32", help="CLIP model to use")

parser.add_argument("--num-workers", type=int, help=("Number of processes to use for data loading. Default to `min(8, num_cpus)`"))

parser.add_argument("--device", type = str, default = None, help = "Device to use. Like cuda, cuda:0 or cpu")

parser.add_argument("--real_flag", type = str, default = "img", help = "The modality of real path. Default to img")

parser.add_argument("--fake_flag", type = str, default = "txt", help = "The modality of fake path. Defaults to txt")

parser.add_argument("--real_path", type = str, help = "Paths to the generated images or .npz statistic files")

parser.add_argument("--fake_path", type=str, help="Paths to generated iamges or .npz statistic files")

IMAGE_EXTENSIONS = {"bmp", "jpg", "jpeg", "pgm", "png", "ppm", "tif", "tiff", "webp"}

TEXT_EXTENSIONS = {"txt"}

class DummyDataset(Dataset):
  FLAGS = ["img", "txt"]
  def __init__(
    self, 
    real_path,
    fake_path,
    real_flag = "img",
    fake_flag = "img",
    transform = None,
    tokenizer = None
  ):
    super().__init__()
    assert real_flag in self.FLAGS and fake_flag in self.FLAGS, "CLIP only supports img and txt modalities"

    self.real_folder = self._combine_without_prefix(real_path)
    self.real_flag = real_flag
    self.fake_folder = self._combine_without_prefix(fake_path)
    self.fake_flag = fake_flag
    self.transform = transform
    self.tokenizer = tokenizer
    assert self._check()

  def __len__(self):
    return len(self.real_folder)
  
  def _check(self):
    # works because real_folder and fake_folder are both sorted
    for idx in range(len(self)):
      real_name = self.real_folder[idx].split('.')
      fake_name = self.fake_folder[idx].split('.')
      if fake_name != real_name:
        return False
      
    return True
  
  def __getitem__(self, index):
    if index >= len(self):
      raise IndexError
    
    real_path = self.real_folder[index]
    fake_path = self.fake_folder[index]
    real_data = self._load_modality(real_path, self.real_flag)
    fake_data = self._load_modality(fake_path, self.fake_flag)

  def _load_modality(self, path, modality):
    if modality == 'img':
      data = self._load_img(path)
    elif modality == 'txt':
      data = self._load_txt(path)
    else:
      raise TypeError(f"Got unexpected modality: {modality}")
    return data
  
  def _load_img(self, path):
    img = Image.open(path)
    if self.transform is not None:
      img = self.transform(img)
    return img
  
  def _load_txt(self, path):
    with open(path, 'r') as fp:
      data = fp.read()

    if self.tokenizer is not None:
      data = self.tokenizer(data).squeeze()

    return data
 
  def _combine_without_prefix(self, folder_path, prefix = '.'):
    """
    Make a sorted list of all the files in a folder except for those
    whose name starts with a given character
    By default ignores files starting with a .
    """
    folder = []
    for name in os.listdir(folder_path):
      if name[0] == prefix:
        continue
      folder.append(osp.join(folder_path, name))

    folder.sort()
    return folder

def forward_modality(model, data, flag):
  device = next(model.parameters()).device
  data = data.to(device)
  if flag == 'img':
    return model.encode_image(data)
  if flag == 'txt':
    return model.encode_text(data)
  raise TypeError

@torch.no_grad()
def calculate_clip_score(dataloader, model, real_flag, fake_flag):
  score_acc = 0.
  sample_num = 0.
  # presumably some sort of maximum value of a logit?
  logit_scale = model.logit_scale.exp()

  for batch_data in tqdm(dataloader):
    real, fake = batch_data
    real_features = forward_modality(model, real, real_flag)
    fake_features = forward_modality(model, fake, fake_flag)

    real_features = real.features / real_features.norm(dim = 1, keepdim = True).to(torch.float32)
    fake_features = fake_features / fake_features.norm(dim = 1, keepdim = True).to(torch.float32)

    # dot product * logit_scale
    score = logit_scale * (fake_features * real_features).sum()
    score_acc += score
    sample_num += real.shape[0]
  
  return score_acc / sample_num

def main():
  args = parser.parse_args()

  if args.device is None:
    device = ["cpu", "cuda"][torch.cuda.is_available()]
    device = torch.device(device)
  else:
    device = torch.device(args.device)

  if args.num_workers is None:
    try:
      num_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
      num_cpus = os.cpu_count()

    num_workers = min(num_cpus, 8) if num_cpus is None else 0
  else:
    num_workers = args.num_workers

  model, preprocess = clip.load(args.clip_model, device = device)

  dataset = DummyDataset(args.real_path, args.fake_path, args.real_flag, args.fake_flag, transform = preprocess, tokenizer = clip.tokenize)
  dataloader = DataLoader(dataset, args.batch_size, num_workers = num_workers, pin_memory = True)

  clip_score = calculate_clip_score(dataloader, model, args.real_flag, args.fake_flag)
  clip_score = clip_score.cpu().item()
  print(f"CLIP score: {clip_score}")

if __name__ == '__main__':
    main()
