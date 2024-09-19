from copy import deepcopy
from functools import partial

import torch
from torch import nn, Tensor
from torch.nn import Module

def inplace_copy(target_tensor, source_tensor):
  target_tensor.copy_(source_tensor)

def inplace_lerp(target_tensor, source_tensor, weight):
  target_tensor.lerp_(source_tensor, weight)

class EMA(Module):
  def __init__(
      self,
      model,
      beta = 0.999,
      update_after_step = 100,
      update_every = 10,
      inv_gamma = 1.0,
      power = 2 / 3,
      min_value = 0.0,
      param_or_buffer_names_no_ema = set(),
      ignore_names = set(),
      ignore_startswith_names = set(),
      include_online_model = True,
      allow_different_devices = False,
  ):
    super().__init__()
    self.beta = beta
    self.is_frozen = beta == 1

    self.include_online_model = include_online_model

    self.online_model = model
    
    try:
      self.ema_model = deepcopy(model)
    except Exception as e:
      print(f'Error while trying to deepcopy model for EMA: {e}')
      exit()

    self.inplace_copy = inplace_copy
    self.inplace_lerp = inplace_lerp

    self.ema_model.requires_grad_(False)

    self.parameter_names = {name for name, param in self.ema_model.named_parameters() if torch.is_floating_point(param) or torch.is_complex(param)}
    self.buffer_names = {name for name, buffer in self.ema_model.named_buffers() if torch.is_floating_point(buffer) or torch.is_complex(buffer)}
    
    self.update_every = update_every
    self.update_after_step = update_after_step

    self.inv_gamma = inv_gamma
    self.power = power
    self.min_value = min_value

    self.param_or_buffer_names_no_ema = param_or_buffer_names_no_ema

    self.ignore_names = ignore_names
    self.ignore_startswith_names = ignore_startswith_names

    self.allow_different_devices = allow_different_devices

    self.register_buffer('initted', torch.tensor(False))
    self.register_buffer('step', torch.tensor(0))

  @property
  def model(self):
    return self.online_model
  
  def eval(self):
    return self.ema_model.eval()

  def restore_ema_model_device(self):
    device = self.initted.device
    self.ema_model.to(device)

  def get_params_iter(self, model):
    for name, param in model.named_parameters():
      if name not in self.parameter_names:
        continue
      yield name, param

  def get_buffers_iter(self, model):
    for name, buffer in model.named_buffers():
      if name not in self.buffer_names:
        continue
      yield name, buffer

  def copy_params_from_model_to_ema(self):
    copy = self.inplace_copy
    for (_, ma_params), (_, current_params) in zip(self.get_params_iter(self.ema_model), self.get_params_iter(self.model)):
      copy(ma_params.data, current_params.data)

    for (_, ma_buffers), (_, current_buffers) in zip(self.get_buffers_iter(self.ema_model), self.get_buffers_iter(self.model)):
      copy(ma_buffers.data, current_buffers.data)

  def get_current_decay(self):
    """
    decay = (-epoch)^(-2/3)
    starts at 1 at quickly decreases
    """
    epoch = (self.step - self.update_after_step - 1).clamp(min = 0.)
    
    value = 1 - (1 + epoch / self.inv_gamma) ** (-self.power)

    if epoch.item() <= 0:
      return 0.
    
    return value.clamp(min = self.min_value, max = self.beta).item()

  def update(self):
    step = self.step.item()
    self.step += 1

    if (step % self.update_every) != 0:
      return
    
    # if we haven't yet updated EMA params, initialized them now
    if step <= self.update_after_step:
      self.copy_params_from_model_to_ema()

    # same as above
    # perhaps the use case is, loading from a saved model? 
    # but that should save step too
    if not self.initted.item():
      self.copy_params_from_model_to_ema()
      self.initted.data.copy_(torch.tensor(True))

    self.update_moving_average(self.ema_model, self.model)


  @torch.no_grad()
  def update_moving_average(self, ma_model, current_model):
    if self.is_frozen:
      return
    
    copy, lerp = self.inplace_copy, self.inplace_lerp
    current_decay = self.get_current_decay()

    for (name, current_params), (_, ma_params) in zip(self.get_params_iter(current_model), self.get_params_iter(ma_model)):
      lerp(ma_params.data, current_params.data, 1. - current_decay)

    for (name, current_buffer), (_, ma_buffer) in zip(self.get_buffers_iter(current_model), self.get_buffers_iter(ma_model)):
      lerp(ma_buffer.data, current_buffer.data, 1. - current_decay)

  def __call__(self, *args, **kwargs):
    return self.ema_model(*args, **kwargs)
