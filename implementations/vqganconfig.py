config = {'model': {'base_learning_rate': 4.5e-06, 'target': 'taming.models.vqgan.VQModel', 'params': {'embed_dim': 256, 'n_embed': 16384, 'monitor': 'val/rec_loss', 'ddconfig': {'double_z': False, 'z_channels': 256, 'resolution': 256, 'in_channels': 3, 'out_ch': 3, 'ch': 128, 'ch_mult': [1, 1, 2, 2, 4], 'num_res_blocks': 2, 'attn_resolutions': [16], 'dropout': 0.0}, 'lossconfig': {'target': 'taming.modules.losses.vqperceptual.VQLPIPSWithDiscriminator', 'params': {'disc_conditional': False, 'disc_in_channels': 3, 'disc_start': 0, 'disc_weight': 0.75, 'disc_num_layers': 2, 'codebook_weight': 1.0}}}}}

ddconfig = config["model"]["params"]["ddconfig"]
lossconfig = config["model"]["params"]["lossconfig"]

vqgan_config = dict(
  ddconfig = ddconfig,
  lossconfig = lossconfig,
  embed_dim = 256,
  n_embed = 16384
)