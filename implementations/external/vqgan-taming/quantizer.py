import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from einops import rearrange

class VectorQuantizer2(nn.Module):
  def __init__(
    self,
    n_e, # n_embed = 256
    e_dim, # embed_dim = 16384
    beta, # 0.25
    remap = None,
    unknown_index = "random",
    sane_index_shape = False,
    legacy = True
  ):
    super().__init__()
    self.n_e = n_e
    self.e_dim = e_dim
    self.beta = beta
    self.legacy = legacy

    # 256 embedding vectors, each of size 16384
    self.embedding = nn.Embedding(self.n_e, self.e_dim)
    # uniform between -1/256 and 1/256
    self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)    

    self.re_embed = n_e
    self.sane_index_shape = False

  def remap_to_used(self, inds):
    pass

  def forward(self, z):
    z = rearrange(z, 'b c h w -> b h w c').contiguous()
    # 16 by 16384
    z_flattened = z.view(-1, self.e_dim)

    d = torch.sum(z_flattened ** 2, dim = 1, keepdim = True) + \
        torch.sum(self.embedding.weight**2, dim = 1) - 2 * \
        torch.einsum('bd, dn -> bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))
    
    # d[t][e] = distance from token t to embedding e
    
    min_encoding_indices = torch.argmin(d, dim = 1)

    # min_encoding_indices[i] = id of the closest embedding to token i
    z_q = self.embedding(min_encoding_indices).view(z.shape)

    min_encodings = None

    # beta = 0.25 so the first term dominates
    # this seems to imply we want the model to push the 
    # encoder output towards the embeddings
    # and less so the embeddings towards the output
    loss = torch.mean((z_q.detach() - z)**2) + self.beta * \
          torch.mean((z_q - z.detach()) ** 2)

    z_q = z + (z_q - z).detach()

    z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

    return z_q, loss,  min_encoding_indices

  def get_codebook_entry(self, indices, shape):
    z_q = self.embedding(indices)

    if shape is not None:
      z_q = z_q.view(shape)
      z_q = z_q.permute(0, 3, 1, 2).contiguous()

    return z_q