import sys
sys.path.append("external")
sys.path.append("external/vqgan-taming")
from masked_model import MaskedModel, SequenceModelWrapper
from lowres_trainer import LowResTrainer

from vqgan import VQModel

from masked_transformer import Transformer

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
    num_tokens = 16384,       # must be same as codebook size above
    seq_len = 16*16,            # must be equivalent to fmap_size ** 2 in vae
    dim = 1024,                # model dimension
    depth = 12,                # depth
    dim_head = 64,            # attention head dimension
    heads = 4,                # attention heads,
    ff_mult = 2,              # feedforward expansion factor
)

base_maskgit = MaskGit(
    transformer = transformer, # transformer
    image_size = 256,          # image size
    cond_drop_prob = 0.25,     # conditional dropout, for classifier free guidance
    self_token_critic = True,
    no_mask_token_prob = 0.25,
).cuda()

base_maskgit.load("../results/original-transformer_results/maskgit.400000.pt")

trainer = LowResTrainer(
  masked_model = base_maskgit,
  vae_model = vae,
  train_folder = "../dataset/cc3m-train-256x256",
  valid_folder = "../dataset/cc3m-valid-256x256",
  num_train_steps = 1100000,
  batch_size = 8,
  results_folder = "../results/original-transformer_results-part2",
  save_results_every = 100,
  save_model_every = 10000,
  fmap_size = 16,
)

trainer.train()