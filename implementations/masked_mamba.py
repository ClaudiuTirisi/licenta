import torch.nn as nn

import torch

import math

from mamba_ssm import Mamba

from torchtune.modules import RMSNorm

from mamba_ssm.ops.triton.layernorm import layer_norm_fn
from einops import repeat

from functools import partial

import sys
sys.path.append("..")

from masked_model import MaskedModel, SequenceModelWrapper

class MambaIT(nn.Module):
  def __init__(
      self,
      token_size,
      depth,
      d_state = 16,
      d_conv = 4,
      expand = 2,
  ):
    super().__init__()
    self.token_size = token_size
    self.mamba_layers = nn.ModuleList([Mamba(d_model = token_size, d_state = d_state, d_conv = d_conv, expand = expand) for _ in range(depth)])
    self.norm = RMSNorm(token_size)
    self.apply(self._init_weights)
       

  def _init_weights(
    module,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
  ):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))

  def forward(
      self,
      x,
      context,
      context_mask
  ):

    context_len = context.shape[1]

    context_mask = repeat(context_mask, 'b t -> b t s', s = self.token_size)
    context = torch.where(context_mask, context, torch.zeros_like(context))

    x = torch.cat((context, x), dim = -2)

    for mamba_layer in self.mamba_layers:
      x = x + mamba_layer(x)
      x = self.norm(x) + 1e-6

    return x[:, context_len:, :]

def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)

