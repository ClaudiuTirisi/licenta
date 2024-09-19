from vae import VQGanVAE
from trainers import VQGanVAETrainer

train_path = "../COCO-Captions/train2017"
valid_path = "../COCO-Captions/val2017"

vae = VQGanVAE(
    dim = 128,
    codebook_size = 8192
)

vae.load("results-backup/vae.376000.ema.pt", is_ema=True)

# train on folder of images, as many images as possible

# one training steps sees batch_size * (2*grad_accum_every) images
# so 4*2*8=64 images
# about 2000 training steps = 1 epoch

trainer = VQGanVAETrainer(
    vae = vae,
    # image_size ignored if is_web_dataset is True
    image_size = 512,
    folder = train_path,
    valid_frac = 0.05,
    batch_size = 3,
    grad_accum_every = 6,
    num_train_steps = 100000-23000-53000,
    lr=1e-4,
).cuda()

# 2*(100000 * 128 + 100000 * 32 * 4 + 100000 * 8 * 4 + 100000 * 3 * 6)

trainer.train()