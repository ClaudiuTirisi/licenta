from collections import namedtuple
import os, hashlib
import requests
from tqdm import tqdm

from enc_dec import Encoder, Decoder

from quantizer import VectorQuantizer2

from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torchvision import models

class ScalingLayer(nn.Module):
  def __init__(self):
    super(ScalingLayer, self).__init__()
    self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
    self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

  def forward(self, inp):
    return (inp - self.shift) / self.scale

class NetLinLayer(nn.Module):
  def __init__(
      self,
      chn_in,
      chn_out = 1,
      use_dropout = False
  ):
    super(NetLinLayer, self).__init__()
    layers = [nn.Dropout(), ] if use_dropout else []
    layers += [nn.Conv2d(chn_in, chn_out, 1, stride = 1, padding = 0, bias = False),]
    self.model = nn.Sequential(*layers)

class vgg16(torch.nn.Module):
  def __init__(
    self,
    requires_grad = False,
    pretrained = True
  ):
    super(vgg16, self).__init__()
    vgg_pretrained_features = models.vgg16(pretrained = pretrained).features

    self.slice1 = torch.nn.Sequential()
    self.slice2 = torch.nn.Sequential()
    self.slice3 = torch.nn.Sequential()
    self.slice4 = torch.nn.Sequential()
    self.slice5 = torch.nn.Sequential()
    self.N_slices = 5

    for x in range(4):
      self.slice1.add_module(str(x), vgg_pretrained_features[x])
    
    for x in range(4, 9):
      self.slice2.add_module(str(x), vgg_pretrained_features[x])

    for x in range(9, 16):
      self.slice3.add_module(str(x), vgg_pretrained_features[x])

    for x in range(16, 23):
      self.slice4.add_module(str(x), vgg_pretrained_features[x])

    for x in range(23, 30):
      self.slice5.add_module(str(x), vgg_pretrained_features[x])

    if not requires_grad:
      for param in self.parameters():
        param.requires_grad = False

  def forward(
    self,
    x
  ):
    h = self.slice1(x)
    h_relu1_2 = h
    h = self.slice2(h)
    h_relu_2_2 = h
    h = self.slice3(h)
    h_relu_3_3 = h
    h = self.slice4(h)
    h_relu_4_3 = h
    h = self.slice5(h)
    h_relu_5_3 = h

    vgg_outputs = namedtuple("VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"])

    out = vgg_outputs(h_relu1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3, h_relu_5_3)
    return out
    
URL_MAP = {
    "vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"
}

CKPT_MAP = {
    "vgg_lpips": "vgg.pth"
}

MD5_MAP = {
    "vgg_lpips": "d507d7349b931f0638a25a48a722f98a"
}


def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


def md5_hash(path):
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def get_ckpt_path(name, root, check=False):
    assert name in URL_MAP
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
        print("Downloading {} model from {} to {}".format(name, URL_MAP[name], path))
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], md5
    return path

class LPIPS(nn.Module):
  def __init__(self, use_dropout = True):
    super().__init__()
    self.scaling_layer = ScalingLayer
    self.chns = [64, 128, 256, 512, 512]
    self.net = vgg16(pretrained = True, requires_grad = False)

    self.lin0 = NetLinLayer(self.chns[0], use_dropout = use_dropout)
    self.lin1 = NetLinLayer(self.chns[1], use_dropout = use_dropout)
    self.lin2 = NetLinLayer(self.chns[2], use_dropout = use_dropout)
    self.lin3 = NetLinLayer(self.chns[3], use_dropout = use_dropout)
    self.lin4 = NetLinLayer(self.chns[4], use_dropout = use_dropout)

    self.load_from_pretrained()

    for param in self.parameters():
      param.requires_grad = False

  def load_from_pretrained(self, name = "vgg_lpips"):
    ckpt = get_ckpt_path(name, "taming/modules/autoencoder/lpips")
    self.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict = False)
    print("Loaded pretrained LPIPS loss from {}".format(ckpt))

  @classmethod
  def from_pretrained(cls, name = "vgg_lpips"):
    if name != "vgg_lpips":
      raise NotImplementedError

    model = cls()
    ckpt = get_ckpt_path(name)
    model.load_state_dict(torch.load(ckpt, map_location = torch.device("cpu")), strict = False)
    return model

  def forward(self, input, target):
    pass

def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    nn.init.normal_(m.weight.data, 0.0, 0.02)
  elif classname.find('BatchNorm') != -1:
    nn.init.normal_(m.weight.data, 1.0, 0.02)
    nn.init.constant_(m.bias.data, 0)

class NLayerDiscriminator(nn.Module):
  def __init__(
      self,
      input_nc = 3,
      ndf = 64,
      n_layers = 3,
  ):
    super(NLayerDiscriminator, self).__init__()
    norm_layer = nn.BatchNorm2d
    use_bias = False

    kw = 4
    padw = 1
    sequence = [
      nn.Conv2d(
        input_nc,
        ndf, 
        kernel_size = kw,
        stride = 2,
        padding = padw
      ),
      nn.LeakyReLU(0.2, True)
    ]

    nf_mult = 1
    nf_mult_prev = 1
    for n in range(1, n_layers):
      nf_mult_prev = nf_mult
      nf_mult = min(2**n, 8)
      sequence += [
        nn.Conv2d(
          ndf * nf_mult_prev,
          ndf * nf_mult,
          kernel_size = kw,
          stride = 2,
          padding = padw,
          bias = use_bias
        ),
        norm_layer(ndf * nf_mult),
        nn.LeakyReLU(0.2, True)
      ]

    nf_mult_prev = nf_mult
    nf_mult = min(2 ** n_layers, 8)
    sequence += [
      nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size = kw, stride = 1, padding = padw, bias = use_bias),
      norm_layer(ndf * nf_mult),
      nn.LeakyReLU(0.2, True)
    ]

    sequence += [
      nn.Conv2d(ndf * nf_mult, 1, kernel_size = kw, stride = 1, padding = padw)
    ]
    self.main = nn.Sequential(*sequence)

  def forward(self, input):
    return self.main(input)

def hinge_d_loss(logits_real, logits_fake):
  loss_real = torch.mean(F.relu(1. - logits_real))
  loss_fake = torch.mean(F.relu(1. + logits_fake))
  d_loss = 0.5 * (loss_real + loss_fake)
  return d_loss

class VQLPIPSWithDiscriminator(nn.Module):
  def __init__(
      self,
      disc_start = 0,
      codebook_weight = 1.0,
      pixelloss_weight = 1.0,
      disc_num_layers = 2,
      disc_in_channels = 3,
      disc_factor = 1.0,
      disc_weight = 1.0,
      perceptual_weight = 1.0,
      disc_conditional = False,
      disc_ndf = 64,
      disc_loss = "hinge"
  ):
    super().__init__()
    self.codebook_weight = codebook_weight
    self.pixel_weight = pixelloss_weight
    self.perceptual_loss = LPIPS().eval()
    self.perceptual_weight = perceptual_weight

    self.discriminator = NLayerDiscriminator(
      input_nc = disc_in_channels,
      n_layers = disc_num_layers,
      ndf = disc_ndf
    ).apply(weights_init)

    self.discriminator_iter_start = disc_start

    self.disc_loss = hinge_d_loss

    self.disc_factor = disc_factor
    self.discriminator_weight = disc_weight
    self.disc_conditional = disc_conditional

  def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
    if last_layer is not None:
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
    else:
        nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

    d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
    d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
    d_weight = d_weight * self.discriminator_weight
    return d_weight

  def forward(
        self,
        codebook_loss,
        inputs,
        reconstructions,
        optimizer_idx,
        global_step,
        last_layer = None,
        cond = None,
        split = "train"
  ):
     pass

class VQModel(pl.LightningModule):
  def __init__(
      self,
      ddconfig,
      lossconfig,
      n_embed = 16384,
      embed_dim = 256,
      ckpt_path = None,
      ignore_keys = [],
      image_key = "image",
      colorize_nlabels = None,
      monitor = "val/rec_loss",
      remap = None,
      sane_index_shape = False
  ):
    super().__init__()
    self.image_key = image_key
    self.encoder = Encoder(**ddconfig)
    self.decoder = Decoder(**ddconfig)
    self.loss = VQLPIPSWithDiscriminator(
      disc_conditional = False,
      disc_in_channels = 3,
      disc_start = 0,
      disc_weight = 0.75,
      disc_num_layers = 2,
      codebook_weight = 1.0
    )
    self.quantize = VectorQuantizer2(
      n_embed,
      embed_dim,
      beta=0.25,
      remap = None,
      sane_index_shape = False
    )
    self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
    self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

  def init_from_ckpt(
      self,
      path,
      ignore_keys = list()
  ):
      sd = torch.load(path, map_location = "cpu")["state_dict"]
      keys = list(sd.keys())
      for k in keys:
        for ik in ignore_keys:
          if k.startswith(ik):
            print("Deleting key {} from state_dict.".format(k))
            del sd[k]
      self.load_state_dict(sd, strict = False)
      print(f"Restored from  {path}")

  def encode(self, x):
    # h is (batch_size, 256, 16, 16)
    h = self.encoder(x)
    # h is still (batch_size, 256, 16, 16)
    h = self.quant_conv(h)
    quant, emb_loss, indices = self.quantize(h)
    return quant, emb_loss, indices.reshape(x.shape[0], -1)
  
  def decode(self, quant):
    quant = self.post_quant_conv(quant)
    dec = self.decoder(quant)
    return dec

  def decode_from_ids(self, indices, batch_size, fmap_size):
    quant_b = rearrange(self.quantize.embedding(indices).view((batch_size, fmap_size, fmap_size, 256)), "b h w c -> b c h w")
    dec = self.decode(quant_b)
    return dec
0.3923, -0.4934, -0.7899
