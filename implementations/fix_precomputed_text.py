import torch
from pathlib import Path

from tqdm import tqdm

folder = "../cc3m-precomputed"

for path in tqdm(Path(folder).glob("**/*.txt")):
    try:
        text = torch.load(path)
    except:
        continue
    with open(path, "w") as f:
        f.write(text)