import torchvision.transforms as T
import torch
import numpy as np
from skimage.color import rgb2lab, rgb2gray
from torchvision import datasets

class ColorizeData(datasets.ImageFolder):

  def __getitem__(self, index):
      path,_ = self.imgs[index]
      input = self.loader(path) 
      input = self.transform(input) 
      input = np.asarray(input) 
      img_lab = rgb2lab(input) 
      img_lab = (img_lab + 128) / 255 
      img_ab = img_lab[:, :, 1:3] 
      img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float() 
      input = rgb2gray(input) 
      input = torch.from_numpy(input).unsqueeze(0).float() 
      return input, img_ab 
