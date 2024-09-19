from muse_maskgit_pytorch import TransformerBlocks

import torch.nn as nn

import torch

import math

import sys
sys.path.append("..")

from masked_model import MaskedModel, SequenceModelWrapper

class Transformer(nn.Module):
  def __init__(
    self,
    token_size,
    depth,
    # if None, dim_out = dim_in
    dim_head = 64,
    ff_mult = 4,
  ):
    super().__init__()
    self.token_size = token_size

    self.transformer_blocks = TransformerBlocks(dim = token_size, depth = depth, dim_head = dim_head, ff_mult = ff_mult)

  def forward(self, x, context, context_mask):
    return self.transformer_blocks(x, context, context_mask)

def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)

