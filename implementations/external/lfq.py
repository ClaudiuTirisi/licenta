from math import log2, ceil
from collections import namedtuple

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.cuda.amp import autocast

from einops import rearrange, reduce, pack, unpack

def pack_one(t, pattern):
  return pack([t], pattern)

def unpack_one(t, ps, pattern):
  return unpack(t, ps, pattern)[0]

def log(t, eps = 1e-5):
  return t.clamp(min = eps).log()

def entropy(prob):
  return (-prob * log(prob)).sum(dim = -1)

class LFQ(nn.Module):
  def __init__(
      self,
      *,
      # encoded dim, as in, number of channels
      dim = None,
      # user-supplied
      codebook_size = None,
      entropy_loss_weight = 0.1,
      commitment_loss_weight = 0.25,
      # set to 4
      diversity_gamma = 1.,
      straight_through_activation = nn.Identity(),
      # always equal to 1
      num_codebooks = 1,
      keep_num_codebooks_dim = None,
      codebook_scale = 1.,
      frac_per_sample_entropy = 1.
  ):
    super().__init__()
    assert dim is not None or codebook_size is not None, 'either dim or codebook_size must be specified'
    assert dim is None or log2(codebook_size).is_integer(), 'codebook size must be power of 2 for lookup free quantization'

    if codebook_size is None:
      codebook_size = 2 ** dim

    codebook_dim = int(log2(codebook_size))

    codebook_dims = codebook_dim * num_codebooks

    if dim is None:
      dim = codebook_dims

    has_projections = dim != codebook_dims
    self.project_in = nn.Linear(dim, codebook_dims) if has_projections else nn.Identity()
    self.project_out = nn.Linear(codebook_dims, dim) if has_projections else nn.Identity()
    self.has_projections = has_projections

    self.dim = dim
    self.codebook_dim = codebook_dim
    self.num_codebooks = num_codebooks

    if keep_num_codebooks_dim is None:
      keep_num_codebooks_dim = num_codebooks > 1
    
    assert not (num_codebooks > 1 and not keep_num_codebooks_dim)

    self.keep_num_codebooks_dim = keep_num_codebooks_dim

    self.activation = straight_through_activation

    assert 0 < frac_per_sample_entropy <= 1.
    self.frac_per_sample_entropy = frac_per_sample_entropy

    self.diversity_gamma = diversity_gamma
    self.entropy_loss_weight = entropy_loss_weight

    self.codebook_scale = codebook_scale

    self.commitment_loss_weight = commitment_loss_weight

    self.register_buffer("mask", 2 ** torch.arange(codebook_dim - 1, -1, -1))
    self.register_buffer("zero", torch.tensor(0.), persistent = False)

    all_codes = torch.arange(codebook_size)

    # all numbers from 0 to codebook_size as bits
    # codebook_dim is chosen to be log2(codebook_size)
    # and bits.shape = codebook_size x codebook_dim
    # ensuring there's enough space for all numbers from 0 to codebook_size
    bits = ((all_codes[..., None].int() & self.mask) != 0).float()

    # turns the 0 bit into -1, and keeps 1 as 1
    codebook = self.bits_to_codes(bits)
    self.register_buffer("codebook", codebook, persistent = False)

  def bits_to_codes(self, bits):
    return bits * 2 - 1

  @property
  def dtype(self):
    return self.codebook.dtype
  
  def indices_to_codes(
      self,
      indices,
      project_out = True
  ):
      is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

      if not self.keep_num_codebooks_dim:
          indices = rearrange(indices, '... -> ... 1')

      # indices to codes, which are bits of either -1 or 1

      bits = ((indices[..., None].int() & self.mask) != 0).to(self.dtype)

      codes = self.bits_to_codes(bits)

      codes = rearrange(codes, '... c d -> ... (c d)')

      # whether to project codes out to original dimensions
      # if the input feature dimensions were not log2(codebook size)

      if project_out:
          codes = self.project_out(codes)

      # rearrange codes back to original shape

      if is_img_or_video:
          codes = rearrange(codes, 'b ... d -> b d ...')

      return codes


  @autocast(enabled = False)
  def forward(
    self,
    # (batch, channels, width, height)
    x,
    # unmodified
    inv_temperature = 100.,
    # always False
    return_loss_breakdown = False,
    # always None
    mask = None
  ):
    x = x.float()
    is_img_or_video = x.ndim >= 4

    # moves the channels dimension to the end
    # and flattens the spatial dimensions
    # afterwards, x[b][i][c] is the value of channel c, for the ith pixel in image b
    if is_img_or_video:
      x = rearrange(x, 'b d ... -> b ... d')
      x, ps = pack_one(x, 'b * d')

    assert x.shape[-1] == self.dim, "Incorrect number of channels passed to quantizer. Should be equal to number of channels the encoder outputs"

    # linear layer that operates on the channel dimension, that was now moved to the end
    # and is considered the feature dimension
    x = self.project_in(x)

    # splits x along the feature dimension among the codebooks
    # we only use one codebook, so x.shape = (b, n, 1, d)
    # essentially just adding a dimension
    x = rearrange(x, 'b n (c d) -> b n c d', c = self.num_codebooks)

    # replace positive values with 1, and negative values with -1
    original_input = x
    codebook_value = torch.ones_like(x)
    quantized = torch.where(x > 0, codebook_value, -codebook_value)

    # the above "binarization" leads to gradients that are equal to 0
    # when computing gradients in the backward phase we skip the gradient of the 
    # binarization function
    if self.training:
      x = x + (quantized - x).detach()
    else:
      x = quantized

    # ignoring the batch, each element in the sequence (the flattened spatial dimension)
    # is transformed into its corresponding binary number
    # by considering values greater than 0 as a 1-bit and everything else as a 0-bit
    indices = reduce((x > 0).int() * self.mask.int(), 'b n c d -> b n c', 'sum')

    if self.training:

      # distance[n][c] = -2 * sum over d of original_input[n][d] * self.codebook[c][d]
      # distance[n][c] = -2 original_input[n] dot codebook[c]
      distance = -2 * einsum('... i d, j d -> ... i j', original_input, self.codebook)
      
      # probability of each code in the codebook being chosen
      # assuming we are choosing codes based on cosine similarity
      # which is close to euclidean distance since the codes are
      # equal to -1 or 1
      prob = (-distance * inv_temperature).softmax(dim = -1)

      # consolidate all samples, by flattening across batch dimension
      prob = rearrange(prob, 'b n ... -> (b n) ...')

      # entropy for each sample
      # if high, that means the model doesn't strongly prefer one codebook entry
      # to represent the sample
      # if low, that means the model strongly prefers one codebook entry
      per_sample_entropy = entropy(prob).mean()

      # for each codebook entry, compute the probability of it being chosen
      avg_prob = reduce(prob, '... c d -> c d', 'mean')
      codebook_entropy = entropy(avg_prob).mean()

      # we want the model to clearly choose one codebook entry 
      # so we minimize per sample entropy
      # and we want the model to make use of all the codebook entries
      # so we maximize the codebook entropy

      entropy_aux_loss = per_sample_entropy - self.diversity_gamma * codebook_entropy
      
    else:
      entropy_aux_loss = per_sample_entropy = codebook_entropy = self.zero

    # we want the encoder to commit to a certain codebook representation
    # by having its output already be close to quantized
    if self.training:
      commit_loss = F.mse_loss(original_input, quantized.detach(), reduction = 'none')
      commit_loss = commit_loss.mean()
    else:
      commit_loss = self.zero
      
    x = rearrange(x, 'b n c d -> b n (c d)')
    
    x = self.project_out(x)

    if is_img_or_video:
      x = unpack_one(x, ps, 'b * d')
      x = rearrange(x, 'b ... d -> b d ...')
      indices = unpack_one(indices, ps, 'b * c')
      
    if not self.keep_num_codebooks_dim:
      indices = rearrange(indices, '... 1 -> ...')

    aux_loss = entropy_aux_loss * self.entropy_loss_weight + commit_loss * self.commitment_loss_weight

    if not return_loss_breakdown:
      return x, indices, aux_loss
    
    return x, indices, aux_loss, per_sample_entropy, codebook_entropy, commit_loss