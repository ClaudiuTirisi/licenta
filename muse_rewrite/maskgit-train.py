from vae import VQGanVAE
from trainers import VQGanVAETrainer

from maskgit_lowres_trainer import MaskGitTrainer
from masked_transformer import MaskGitTransformer, MaskGit, TokenCritic

train_path = "../COCO-Captions/train2017"
valid_path = "../COCO-Captions/val2017"

vae = VQGanVAE(
    dim = 128,
    codebook_size = 8192
)

# this checkpoint was trained with image size upto 256
vae.load("results-backup/vae.300000.ema.pt", is_ema=True)

transformer = MaskGitTransformer(
    num_tokens = 8192,       # must be same as codebook size above
    seq_len = 16*16,            # must be equivalent to fmap_size ** 2 in vae
    dim = 2048,                # model dimension
    depth = 24,                # depth
    dim_head = 64,            # attention head dimension
    heads = 8,                # attention heads,
    ff_mult = 2,              # feedforward expansion factor
)

base_maskgit = MaskGit(
    vae = vae,                 # vqgan vae
    transformer = transformer, # transformer
    image_size = 256,          # image size
    cond_drop_prob = 0.25,     # conditional dropout, for classifier free guidance
    self_token_critic = True,
    no_mask_token_prob = 0.25,
).cuda()

base_maskgit.load("results-maskgit/maskgit.299999.pt")

trainer = MaskGitTrainer(
    base_maskgit,
    image_folder = "../COCO-Captions/train2017",
    token_folder = "../COCO-Captions/train_tokens",
    caption_file = "../COCO-Captions/annotations_trainval2017/captions_train2017.json",
    num_train_steps = 300000,
    batch_size = 4,
    image_size = 256,
    lr = 1e-4,
    weight_decay = 0.045,
    grad_accum_every = 1,
    save_results_every = 100,
    save_model_every = 1000,
    results_folder = "./results-maskgit",
    random_split_seed = 42, 
    valid_frac = 0.05,
)

trainer.train()