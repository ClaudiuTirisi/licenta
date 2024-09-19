from pathlib import Path
import copy
import math
from math import sqrt
from functools import partial, wraps

from lfq import LFQ
import torch.nn as nn
import torch

import torch.nn.functional as F
from torch.autograd import grad

import torchvision

from einops import rearrange, pack , unpack

def group_dict_by_key(cond, d):
  """
  Splits dictionary d into 2 dictionaries such that
  the first dictionary contains key-pair values for which the cond(key) = True
  and the second dictionary contains everything else
  """
  return_val = [dict(), dict()]
  for key in d.keys():
    match = bool(cond(key))
    ind = int(not match)
    return_val[ind][key] = d[key]
  return (*return_val,)

def string_begins_with(prefix, string_input):
  return string_input.startswith(prefix)

def group_by_key_prefix(prefix, d):
  """
  Returns 2 dicts, first has keys that start with the given prefix
  and the second has the rest of the keys
  """
  return group_dict_by_key(partial(string_begins_with, prefix), d)

def groupby_prefix_and_trim(prefix, d):
  """
  group_by_key_prefix but it removed the prefix from the resulting dict
  """
  kwargs_with_prefix, kwargs = group_by_key_prefix(prefix, d)
  kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
  return kwargs_without_prefix, kwargs

def gradient_penalty(images, output, weight = 10):
  """
  L2 gradient penalty that constraints the Discriminator function
  to be 1-Lipschitz
  grad computes the gradient of outputs with respect to inputs
  assuming that outputs = L(f(inputs))
  The chain rule says that ouputs' = L'(f(inputs)) * f'(inputs)
  and grad requires that we pass the values of f'(inputs)
  Since we are interested in penalizing the gradients of the discriminator
  w.r.t to the inputs, we want the function f to be the identity
  so we pass a tensor on 1s as its gradient
  """
  gradients = grad(
    outputs = output,
    inputs = images,
    grad_outputs = torch.ones(output.size(), device = images.device),
    create_graph = True,
    retain_graph = True,
  )[0]

  gradients = rearrange(gradients, 'b ... -> b (...)')
  return weight * ((gradients.norm(2, dim = 1) - 1)**2).mean()

def grad_layer_wrt_loss(loss, layer):
  return grad(
    outputs = loss,
    inputs = layer,
    grad_outputs = torch.ones_like(loss),
    retain_graph = True
  )[0].detach()

def safe_div(numer, denom, eps = 1e-8):
  return numer / denom.clamp(min = eps)

def remove_vgg(fn):
  """
  Decorator that removes the vgg attribute,
  runs the function fn, and then reads it.
  Used to dump and load the state dict 
  since VGG is pretrained
  """
  @wraps(fn)
  def inner(self, *args, **kwargs):
    has_vgg = hasattr(self, '_vgg')
    if has_vgg:
      vgg = self._vgg
      delattr(self, '_vgg')
    out = fn(self, *args, **kwargs)
    
    if has_vgg:
      self._vgg = vgg

    return out

  return inner

class GLUResBlock(nn.Module):
  def __init__(self, channels, groups = 16):
    super().__init__()
    self.net = nn.Sequential(
      nn.Conv2d(channels, channels * 2, 3, padding = 1),
      # first half of x * sigmoid(second half of x)
      # halving alongside channel dimension
      nn.GLU(dim = 1),
      nn.GroupNorm(groups, channels),
      nn.Conv2d(channels, channels * 2, 3, padding = 1),
      nn.GLU(dim = 1),
      nn.GroupNorm(groups, channels),
      nn.Conv2d(channels, channels, 1)
    )
  
  def forward(self, x):
    return self.net(x) + x

class ResBlock(nn.Module):
  def __init__(self, channels, groups = 16):
    super().__init__()
    self.net = nn.Sequential(
      nn.Conv2d(channels, channels, 3, padding = 1),
      nn.GroupNorm(groups, channels),
      nn.LeakyReLU(0.1),
      nn.Conv2d(channels, channels, 3, padding = 1),
      nn.GroupNorm(groups, channels),
      nn.LeakyReLU(0.1),
      nn.Conv2d(channels, channels, 1)
    )

  def forward(self, x):
    return self.net(x) + x

class ResnetEncDec(nn.Module):
  def __init__(
    self,
    dim,
    *,
    channels = 3,
    layers = 4,
    layer_mults = None,
    num_resnet_blocks = 1,
    resnet_groups = 16,
    first_conv_kernel_size = 5
  ):
    super().__init__()

    # group norm is across channels, why is this check done for dim?
    assert dim % resnet_groups == 0, f'Dimension {dim} must divide the number of groups'

    self.layers = layers
    self.encoders = nn.ModuleList([])
    self.decoders = nn.ModuleList([])

    if layer_mults is None:
      layer_mults = [2 ** t for t in range(layers)]

    assert len(layer_mults) == layers, 'layer multipliers must be equal to number of layers'

    layer_dims = [dim * mult for mult in layer_mults]
    dims = (dim, *layer_dims)

    self.encoded_dim = dims[-1]
    
    # (input channels, output channels) pairs for convolutions ?
    dim_pairs = zip(dims[:-1], dims[1:])

    append = lambda arr, t: arr.append(t)
    prepend = lambda arr, t: arr.insert(0, t)

    # if num_resnet_blocks is a number
    # turns it into a tuple
    # filling in the missing spots with 0s
    # obtaining a tuple of length equal to the number of layers
    # so it considers that only the last layer contains resnet blocks?
    if not isinstance(num_resnet_blocks, tuple):
      num_resnet_blocks = (*((0, )*(layers - 1)), num_resnet_blocks)

    assert len(num_resnet_blocks) == layers, 'must  specify the number of resnet blocks for each layer'

    for layer_index, (dim_in, dim_out), layer_num_resnet_blocks in zip(range(layers), dim_pairs, num_resnet_blocks):
      append(self.encoders, nn.Sequential(nn.Conv2d(dim_in, dim_out, 4, stride = 2, padding = 1), nn.LeakyReLU(0.1)))
      prepend(self.decoders, nn.Sequential(nn.ConvTranspose2d(dim_out, dim_in, 4, 2, 1), nn.LeakyReLU(0.1)))

      for _ in range(layer_num_resnet_blocks):
        append(self.encoders, ResBlock(dim_out, groups = resnet_groups))
        prepend(self.decoders, GLUResBlock(dim_out, groups = resnet_groups))

    prepend(self.encoders, nn.Conv2d(channels, dim, first_conv_kernel_size, padding = first_conv_kernel_size // 2))
    append(self.decoders, nn.Conv2d(dim, channels, 1))

    """
    encoders: Conv2d(channels, dim)
              (Conv2d(dim, dim ** 2, stride = 2), n * Resnet(dim ** 2)
              (Conv2d(dim ** 2, dim ** 4, stride =2), n * Resnet(dim ** 4))
              .
              .
              .
              

    decoders: (GLUResBlock(dim_final)
              ConvTranspose2d(dim_final, dim_final // 2, kernel = 2))
              .
              .
              .
              .
              (GLUResBlock(dim ** 2)
              ConvTranspose2d(dim ** 2, dim, kernel = 2)
              )
              Conv2d(dim, channels)
              
    """    

  def get_encoded_fmap_size(self, image_size):
    return image_size // (2 ** self.layers)

  @property
  def last_dec_layer(self):
    return self.decoders[-1].weight

  def encode(self, x):
    for enc in self.encoders:
      x = enc(x)
    return x

  def decode(self, x):
    for dec in self.decoders:
      x = dec(x)
    return x    

def hinge_discr_loss(fake, real):
  """
  fake is the output of the discriminator when its input is the decoded image
  real is the output of the discriminator when its input is the real image
  where decoded image = Dec(Enc(real image))
  the discriminator is supposed to return 0 for images it considers fake
  and 1 for images it considers real
  the relu ensures clamps the output to be positive
  ignoring the relu, the loss becomes
  2 + fake - real
  so to minimize the loss, we must have fake = 0, meaning the discriminator considers the decoded images fake
  and we must have real = 1, which means the discriminator considers the original images real
  """
  return (F.relu(1 + fake) + F.relu(1 - real)).mean()

def hinge_gen_loss(fake):
  """
  The generator (which is the decoder) tries to pass its images as real
  so it wants fake to be equal to 1, meaning the discriminator considers them real
  which corresponds to maximizing fake, or minimizing -fake
  """
  return - fake.mean()

class Discriminator(nn.Module):
  def __init__(
      self,
      dims,
      channels = 3,
      groups = 16,
      init_kernel_size = 5
  ):
    super().__init__()
    dim_pairs = zip(dims[:-1], dims[1:])

    # project to dims[0] channels
    self.layers = nn.ModuleList([nn.Sequential(nn.Conv2d(channels, dims[0], init_kernel_size, padding = init_kernel_size // 2), nn.LeakyReLU(0.1))])

    for dim_in, dim_out in dim_pairs:
      self.layers.append(nn.Sequential(
        nn.Conv2d(dim_in, dim_out, 4, stride = 2, padding = 1),
        nn.GroupNorm(groups, dim_out),
        nn.LeakyReLU(0.1)
      ))

    dim = dims[-1]
    self.to_logits = nn.Sequential(
      nn.Conv2d(dim, dim, 1),
      nn.LeakyReLU(0.1),
      nn.Conv2d(dim, 1, 4)
    )

  # returns a 5x5 1-channel "image" if it receives a 32x32 image
  def forward(self, x):
    for net in self.layers:
      x = net(x)

    return self.to_logits(x)

class VQGanVAE(nn.Module):
  # only dim = 128 and codebook_size = 65536 were passed
  def __init__(
    self,
    *,
    dim,
    channels = 3,
    layers = 4,
    vgg = None,
    codebook_size = 65536,
    # always True
    use_vgg_and_gan = True,
    discr_layers = 4,
    **kwargs
  ):
    super().__init__()
    
    self.channels = channels
    self.codebook_size = codebook_size
    self.dim_divisor = 2 ** layers

    enc_dec_class = ResnetEncDec
    self.enc_dec = enc_dec_class(
      dim = dim,
      channels = channels,
      layers = layers
    )

    self.quantizer = LFQ(
      dim = self.enc_dec.encoded_dim,
      codebook_size = codebook_size,
      diversity_gamma = 4
    )

    # reconstruction loss
    # mean of |target - input|
    self.recon_loss_fn = F.l1_loss

    # vgg remaind None
    self._vgg = None
    self.discr = None
    self.use_vgg_and_gan = use_vgg_and_gan

    layer_mults = list(map(lambda t: 2 ** t, range(discr_layers)))
    layer_dims = [dim * mult for mult in layer_mults]
    dims = (dim, *layer_dims)

    self.discr = Discriminator(dims = dims, channels = channels)
    
    self.discr_loss = hinge_discr_loss
    self.gen_loss = hinge_gen_loss

  @property
  def device(self):
    return next(self.parameters()).device

  @property
  def vgg(self):
    if self._vgg is not None:
      return self._vgg
    
    # vgg takes in a 3-channel image and returns 4096 features
    vgg = torchvision.models.vgg16(pretrained = True)
    vgg.classifier = nn.Sequential(*vgg.classifier[:-2])
    self._vgg = vgg.to(self.device)

    return self._vgg

  @property
  def encoded_dim(self):
    """
    The number of channels obtained after running the encoder
    """
    return self.enc_dec.encoded_dim
  
  def get_encoded_fmap_size(self, image_size):
    """
    Resolution of image after encoding
    """
    return self.enc_dec.get_encoded_fmap_size(image_size)

  def copy_for_eval(self):
    """
    Creates a copy of the model from which it removes the discriminator
    and the VGG, and then sets it up for evaluation
    """
    device = next(self.parameters()).device
    vae_copy = copy.deepcopy(self.cpu())

    if vae_copy.use_vgg_and_gan:
      del vae_copy.discr
      del vae_copy._vgg

    vae_copy.eval()
    return vae_copy.to(device)

  @remove_vgg
  def state_dict(self, *args, **kwargs):
    return super().state_dict(*args, **kwargs)

  @remove_vgg
  def load_state_dict(self, *args, **kwargs):
    return super().load_state_dict(*args, **kwargs)

  def save(self, path):
    torch.save(self.state_dict(), path)

  def get_online_weights_from_ema_model(self, state_dict):
    return {k.split("online_model.")[1]:v for k, v in state_dict.items() if k.startswith("online_model.")}
  
  def get_ema_weights_from_ema_model(self, state_dict):
    return {k.split("ema_model.")[1]:v for k, v in state_dict.items() if k.startswith("ema_model.")}

  def remove_vgg_from_state_dict(self, state_dict):
    return {k:v for k, v in state_dict.items() if not k.startswith("_vgg")}

  def load(self, path, is_ema = True):
    path = Path(path)
    assert path.exists()
    state_dict = torch.load(str(path))
    if is_ema:
      weights = self.remove_vgg_from_state_dict(self.get_ema_weights_from_ema_model(state_dict))
    else:
      weights = self.remove_vgg_from_state_dict(state_dict)
    self.load_state_dict(weights)

  def encode(self, fmap):
    fmap = self.enc_dec.encode(fmap)
    fmap, indices, vq_aux_loss = self.quantizer(fmap)
    return fmap, indices, vq_aux_loss

  def decode(self, fmap):
    return self.enc_dec.decode(fmap)

  def decode_from_ids(self, ids):

      ids, ps = pack([ids], 'b *')

      fmap = self.quantizer.indices_to_codes(ids)

      fmap, = unpack(fmap, ps, 'b * c')

      fmap = rearrange(fmap, 'b h w c -> b c h w')
      return self.decode(fmap)
  
  def forward(
      self,
      img,
      return_loss = False,
      # whether the discriminator loss is returned
      return_discr_loss = False,
      # whether the reconstructed image is returned
      return_recons = False,
      add_gradient_penalty = True
  ):
    batch, channels, height, width, device = *img.shape, img.device

    assert (height % self.dim_divisor) == 0, f'For the encoder, height must divide {self.dim_divisor}'
    assert (width % self.dim_divisor) == 0, f'For the encoder, width must divide {self.dim_divisor}'

    fmap, indices, commit_loss = self.encode(img)
    fmap = self.decode(fmap)

    if not return_loss and not return_discr_loss:
      return fmap

    if return_discr_loss:
      fmap.detach_()
      img.requires_grad_()

      fmap_discr_logits, img_discr_logits = self.discr(fmap), self.discr(img)

      discr_loss = self.discr_loss(fmap_discr_logits, img_discr_logits)

      if add_gradient_penalty:
        gp = gradient_penalty(img, img_discr_logits)
        loss = discr_loss + gp

      if return_recons:
        return loss, fmap

      return loss

    recon_loss = self.recon_loss_fn(fmap, img)

    if not self.use_vgg_and_gan:
      if return_recons:
        return recon_loss, fmap

      return recon_loss
    
    img_vgg_input = img
    fmap_vgg_input = fmap

    img_vgg_feats = self.vgg(img_vgg_input)
    recon_vgg_feats = self.vgg(fmap_vgg_input)
    perceptual_loss = F.mse_loss(img_vgg_feats, recon_vgg_feats)

    gen_loss = self.gen_loss(self.discr(fmap))

    """
    Adaptive Loss for GANS: https://arxiv.org/pdf/2012.03149.pdf
    Gives more or less weight to the generator loss
    By comparing it with the perceptual loss (which can be thought of as somewhat of a discriminator loss)
    """
    last_dec_layer = self.enc_dec.last_dec_layer

    norm_grad_wrt_gen_loss = grad_layer_wrt_loss(gen_loss, last_dec_layer).norm(2)
    norm_grad_wrt_perceptual_loss = grad_layer_wrt_loss(perceptual_loss, last_dec_layer).norm(2) 

    adaptive_weight = safe_div(norm_grad_wrt_perceptual_loss, norm_grad_wrt_gen_loss)
    adaptive_weight.clamp_(max = 1e4)

    loss = recon_loss + perceptual_loss + commit_loss + adaptive_weight * gen_loss 

    if return_recons:
      return loss, fmap
    
    return loss
