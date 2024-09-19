import sys
sys.path.append("external")
sys.path.append("external/vqgan-taming")
from masked_model import MaskedModel, SequenceModelWrapper
from lowres_trainer import LowResTrainer

from vqgan import VQModel

from masked_transformer import Transformer

import torch
import math

from muse_maskgit_pytorch import TransformerBlocks

import torch.nn as nn

import torch

import math

from vqganconfig import vqgan_config

from mambaCText import MambaCText
import sys
sys.path.append("external")
sys.path.append("external/vqgan-taming")
from masked_model import MaskedModel, SequenceModelWrapper
from lowres_trainer import LowResTrainer

from vqgan import VQModel

from masked_transformer import Transformer

import torch
import math

import torch.nn as nn

import torch

import math

from vqganconfig import vqgan_config

def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)

vae = VQModel(**vqgan_config)
vae.init_from_ckpt(path = "models/pretrained-vqgan.ckpt")

vae.cuda()

from muse_maskgit_pytorch import MaskGit, MaskGitTransformer

transformer = MaskGitTransformer(
    actual_model = MambaCText(
      token_size = 1024,
      depth = 12,
      d_state = 16,
    ),
    num_tokens = 16384,       # must be same as codebook size above
    seq_len = 16*16,            # must be equivalent to fmap_size ** 2 in vae
    dim = 1024,                # model dimension
)

base_maskgit = MaskGit(
    transformer = transformer, # transformer
    image_size = 256,          # image size
    cond_drop_prob = 0.25,     # conditional dropout, for classifier free guidance
    self_token_critic = True,
    no_mask_token_prob = 0.25,
).cuda()

trainer = LowResTrainer(
  masked_model = base_maskgit,
  vae_model = vae,
  train_folder = "../dataset/cc3m-train-256x256",
  valid_folder = "../dataset/cc3m-valid-256x256",
  num_train_steps = 1500000,
  batch_size = 8,
  results_folder = "../results/mamba-text-results",
  save_results_every = 1000,
  save_model_every = 1000,
  fmap_size = 16,
)

trainer.train()
