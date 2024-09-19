import sys
from pathlib import Path
sys.path.append("external")
sys.path.append("external/vqgan-taming")

from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from einops import rearrange
import torch

from muse_maskgit_pytorch.t5 import t5_encode_text, DEFAULT_T5_NAME

def encode_text(texts):
  return t5_encode_text(texts, DEFAULT_T5_NAME, output_device = "cuda")

from vqgan import VQModel

from custom_datasets import ImageTextNameDataset

from vqganconfig import vqgan_config

def precompute(dataset, vae, t5_encode_fn, save_to, batch_size):
  """
  Text embeds take too much space, so only precompute image tokens
  """
  dataloader = DataLoader(dataset, batch_size = batch_size, num_workers = 8)

  for images, texts, file_names in tqdm(dataloader):
    texts = list(texts)
    file_names = list(file_names)
    with torch.no_grad():
      _, _, indices = vae.encode(images.cuda())
    
    # values range from 0 to 16384, int16 should be enough
    indices = indices.short()

    for i in range(batch_size):
      with open(f'{save_to}/{file_names[i]}.txt', "w") as f:
          f.write(texts[i])
      with open(f'{save_to}/{file_names[i]}-indices.txt', "w") as f:
          f.write(str(indices[i].tolist()))

vae = VQModel(**vqgan_config)
vae.init_from_ckpt(path = "models/pretrained-vqgan.ckpt")

vae.cuda()

dataset = ImageTextNameDataset("../CC3M/", image_size = 256)

precompute(dataset, vae, encode_text, "../cc3m-precomputed", batch_size = 64)
