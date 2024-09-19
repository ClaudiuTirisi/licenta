from collections import namedtuple
import os, hashlib
import requests
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torchvision import models

def Normalize(in_channels):
  return torch.nn.GroupNorm(num_groups = 32, num_channels = in_channels, eps = 1e-6, affine = True)

def nonlinearity(x):
  return x * torch.sigmoid(x)

class ResnetBlock(nn.Module):
  def __init__(
    self,
    *,
    in_channels,
    out_channels = None,
  ):
    super().__init__()
    self.in_channels = in_channels
    out_channels = in_channels if out_channels is None else out_channels
    self.out_channels = out_channels

    self.norm1 = Normalize(in_channels)
    self.conv1 = torch.nn.Conv2d(
      in_channels,
      out_channels,
      kernel_size = 3,
      stride = 1,
      padding = 1
    )

    self.norm2 = Normalize(out_channels)

    self.conv2 = torch.nn.Conv2d(
      out_channels,
      out_channels,
      kernel_size = 3,
      stride = 1,
      padding = 1
    )

    if self.in_channels != self.out_channels:
      # sort of a linear layer
      self.nin_shortcut = torch.nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size = 1,
        stride = 1,
        padding = 0
      )

  def forward(self, x):
    h = x
    h = self.norm1(h)
    h = nonlinearity(h)
    h = self.conv1(h)

    h = self.norm2(h)
    h = nonlinearity(h)
    h = self.conv2(h)

    if self.in_channels != self.out_channels:
      x = self.nin_shortcut(x)

    return x + h

class AttnBlock(nn.Module):
  def __init__(self, in_channels):
    super().__init__()
    self.in_channels = in_channels

    self.norm = Normalize(in_channels)
    self.q = torch.nn.Conv2d(
      in_channels,
      in_channels,
      kernel_size = 1,
      stride = 1,
      padding = 0
    )

    self.k = torch.nn.Conv2d(
      in_channels,
      in_channels,
      kernel_size = 1,
      stride = 1,
      padding = 0
    )

    self.v = torch.nn.Conv2d(
      in_channels,
      in_channels,
      kernel_size = 1,
      stride = 1,
      padding = 0
    )

    self.proj_out = torch.nn.Conv2d(
      in_channels,
      in_channels,
      kernel_size = 1,
      stride = 1,
      padding = 0
    )

  def forward(self, x):
    h_ = x
    h_ = self.norm(h_)
    q = self.q(h_)
    k = self.k(h_)
    v = self.v(h_)

    b, c, h, w = q.shape
    q = q.reshape(b, c, h*w)
    q = q.permute(0, 2, 1)
    k = k.reshape(b, c, h*w)
    w_ = torch.bmm(q, k)
    w_ = w_ *  (int(c)**(-0.5))
    w_ = torch.nn.functional.softmax(w_, dim = 2)

    v = v.reshape(b, c, h*w)
    w_ = w_.permute(0, 2, 1)
    h_ = torch.bmm(v, w_)
    h_ = h_.reshape(b, c, h, w)
    
    h_ = self.proj_out(h_)

    return x+h_

class Downsample(nn.Module):
  def __init__(self, in_channels):
    super().__init__()
    self.conv = torch.nn.Conv2d(
      in_channels,
      in_channels,
      kernel_size = 3,
      stride = 2,
      padding = 0
    )

  def forward(self, x):
    pad = (0, 1, 0, 1)
    x = torch.nn.functional.pad(x, pad, mode = "constant", value = 0)
    x = self.conv(x)
    return x

class Upsample(nn.Module):
  def __init__(
      self,
      in_channels
  ):
    super().__init__()
    self.conv = torch.nn.Conv2d(
      in_channels,
      in_channels,
      kernel_size = 3,
      stride = 1,
      padding = 1
    )

  def forward(self, x):
    x = torch.nn.functional.interpolate(
      x,
      scale_factor = 2.0,
      mode = "nearest"
    )
    x = self.conv(x)
    return x

class Encoder(nn.Module):
  def __init__(
      self,
      *,
      ch, # 128
      ch_mult = (1, 2, 4, 8), # [1, 1, 2, 2, 4]
      num_res_blocks, # 2
      attn_resolutions, # 16
      in_channels, # 3
      resolution, # 256
      z_channels, # 256
      **ignore_kwargs
  ):
    super().__init__()
    self.ch = ch
    self.temb_ch = 0
    self.num_resolutions = len(ch_mult)
    self.num_res_blocks = num_res_blocks
    self.resolution = resolution
    self.in_channels = in_channels

    # downsampling

    self.conv_in = torch.nn.Conv2d(
      in_channels,
      self.ch,
      kernel_size = 3,
      stride = 1,
      padding = 1,
    )

    curr_res = resolution

    # (1, 1, 1, 2, 2, 4)
    in_ch_mult = (1, ) + tuple(ch_mult)

    self.down = nn.ModuleList()
    for i_level in range(self.num_resolutions):
      block = nn.ModuleList()
      attn = nn.ModuleList()
      block_in = ch*in_ch_mult[i_level]
      block_out = ch*ch_mult[i_level]
      for i_block in range(self.num_res_blocks):
        block.append(
          ResnetBlock(
            in_channels = block_in,
            out_channels = block_out,
          )
        )
        block_in = block_out

        if curr_res in attn_resolutions:
          attn.append(AttnBlock(block_in))

      down = nn.Module()
      down.block = block
      down.attn = attn

      if i_level != self.num_resolutions - 1:
        down.downsample = Downsample(block_in)
        curr_res = curr_res // 2
      
      self.down.append(down)

    # middle

    self.mid = nn.Module()
    self.mid.block_1 = ResnetBlock(
      in_channels = block_in,
      out_channels = block_in,
    )
    self.mid.attn_1 = AttnBlock(block_in)

    self.mid.block_2 = ResnetBlock(
      in_channels = block_in,
      out_channels = block_in,
    )

    # end

    self.norm_out = Normalize(block_in)
    self.conv_out = torch.nn.Conv2d(
      block_in,
      z_channels,
      kernel_size = 3,
      stride = 1,
      padding = 1
    )

  def forward(self, x):

    # downsampling
    hs = [self.conv_in(x)]
    for i_level in range(self.num_resolutions):
      for i_block in range(self.num_res_blocks):
        h = self.down[i_level].block[i_block](hs[-1])
        if len(self.down[i_level].attn) > 0:
          h = self.down[i_level].attn[i_block](h)
        hs.append(h)
      if i_level != self.num_resolutions - 1:
        hs.append(self.down[i_level].downsample(hs[-1]))

    # middle
    h = hs[-1]
    h = self.mid.block_1(h)
    h = self.mid.attn_1(h)
    h = self.mid.block_2(h)

    # end
    h = self.norm_out(h)
    h = nonlinearity(h)
    h = self.conv_out(h)
    return h

class Decoder(nn.Module):
  def __init__(
    self,
    *,
    ch, # 128
    out_ch = 3,
    ch_mult = (1, 2, 4, 8), # [1, 1, 2, 2, 4]
    num_res_blocks, # 2
    attn_resolutions, # 16
    in_channels, # 3
    resolution, # 256
    z_channels, # 256
    give_pre_end = False,
    **ignore_kwargs
  ):
    super().__init__()
    self.ch = ch
    self.num_resolutions = len(ch_mult)
    self.num_res_blocks = num_res_blocks
    self.resolution = resolution
    self.in_channels = in_channels

    in_ch_mult = (1, ) + tuple(ch_mult)
    block_in = ch*ch_mult[self.num_resolutions - 1]
    curr_res = resolution // 2**(self.num_resolutions - 1)
    self.z_shape = (1, z_channels, curr_res, curr_res)

    self.conv_in = torch.nn.Conv2d(
      z_channels,
      block_in,
      kernel_size = 3,
      stride = 1,
      padding = 1
    )

    self.mid = nn.Module()
    self.mid.block_1 = ResnetBlock(
      in_channels = block_in,
      out_channels = block_in,
      )

    self.mid.attn_1 = AttnBlock(block_in)
    self.mid.block_2 = ResnetBlock(
      in_channels = block_in,
      out_channels = block_in,
    )

    self.up = nn.ModuleList()
    for i_level in reversed(range(self.num_resolutions)):
      block = nn.ModuleList()
      attn = nn.ModuleList()
      block_out = ch*ch_mult[i_level]
      for i_block in range(self.num_res_blocks + 1):
        block.append(
          ResnetBlock(
            in_channels=block_in,
            out_channels=block_out,
          )
        )
        block_in = block_out
        if curr_res in attn_resolutions:
          attn.append(AttnBlock(block_in))
      up = nn.Module()
      up.block = block
      up.attn = attn
      if i_level != 0:
        up.upsample = Upsample(block_in)
        curr_res = curr_res * 2
      self.up.insert(0, up)

    self.norm_out = Normalize(block_in)
    self.conv_out = torch.nn.Conv2d(
      block_in,
      out_ch,
      kernel_size = 3,
      stride = 1,
      padding = 1
    )

  def forward(self, z):
    h = self.conv_in(z)

    h = self.mid.block_1(h)
    h = self.mid.attn_1(h)
    h = self.mid.block_2(h)

    for i_level in reversed(range(self.num_resolutions)):
      for i_block in range(self.num_res_blocks + 1):
        h = self.up[i_level].block[i_block](h)
        if len(self.up[i_level].attn) > 0:
          h = self.up[i_level].attn[i_block](h)
      
      if i_level != 0:
        h = self.up[i_level].upsample(h)

    h = self.norm_out(h)
    h = nonlinearity(h)
    h = self.conv_out(h)
    return h

