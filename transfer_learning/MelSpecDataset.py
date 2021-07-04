# -*- coding: utf-8 -*-
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import os

class MelSpecDataset(Dataset):
  def __init__(self , df, data_dir):
    
    self.df = df
    self.data_dir = data_dir

  def __getitem__(self, idx):
    d = self.df.iloc[idx]
    arr = np.load(os.path.join(self.data_dir, str(d.image) + '.npy'))
    arr = arr/np.linalg.norm(arr)
    trans = transforms.ToTensor()
    arr = trans(arr).double()
    label = torch.tensor(d.integer)
    return arr, label
  
  def __len__(self):
    return len(self.df)















