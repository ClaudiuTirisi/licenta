import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from pathlib import Path
from PIL import Image

"""

"""
class ImageTextDataset(Dataset):
  def __init__(
      self,
      folder,
      image_size,
      exts = ['jpg', 'jpeg', 'png']
  ):
    super().__init__()
    self.image_size = image_size
    
    self.transform = T.Compose([
      T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
      T.Resize((image_size, image_size)),
      # image might not fit the text after flip and crop
      # T.RandomHorizontalFlip(),
      # T.CenterCrop(image_size),
      T.ToTensor()
    ])

    self.image_paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}') if not "ipynb" in str(p)]

  def __len__(self):
    return len(self.image_paths)

  def __getitem__(self, index):
    image_path = self.image_paths[index]
    
    if "jpg" not in str(image_path):
      print("non jpg image detected")

    text_path = Path(str(image_path).replace("jpg", "txt"))

    img = Image.open(image_path)
    return self.transform(img), open(text_path).read()

class ImageTextNameDataset(ImageTextDataset):
  def __init__(
    self,
    folder,
    image_size,
    exts = ['jpg', 'jpeg', 'png']
  ):
    super().__init__(folder, image_size, exts)

  def __getitem__(self, index):
    img, text = super().__getitem__(index)
    return img, text, str(self.image_paths[index]).split("/")[-1].split(".")[0]

class ImageTokensTextDataset(Dataset):
  def __init__(
      self,
      folder,
      images_only = False
  ):
    super().__init__()
    self.folder = folder
    self.indices_paths = list(Path(f'{folder}').glob(f'**/*-indices.txt'))
    self.images_only = images_only

  def __len__(self):
    return len(self.indices_paths)
  
  def __getitem__(self, index):
    indices_path = self.indices_paths[index]
    indices_tensor = torch.Tensor(list(map(int, open(indices_path).read()[1:-1].split(", ")))).long()
    if self.images_only:
        return indices_tensor
    text_path = Path(str(indices_path).replace("-indices", ""))
    text = open(text_path).read()
    return indices_tensor, text