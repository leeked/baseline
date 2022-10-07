import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from utils import DEVICE

from timm import create_model

"""

Models

"""
class VGGNetwork(nn.Module):
  """
  VGG16 model using pretrained weights

  Parameters:
  -----------

  Attributes:
  -----------
  model_ver : 
    VGG16 pretrained model
  layers : list
    Selection of layers to use from VGG16 (feature extraction)
  feat_extractor : nn.Sequential
    Feature extraction layers to use
  head : nn.Sequential
    Pyramidal classification head
  flatten : nn.Flatten
    Flatten layer
  """

  def __init__(self):
    super(VGGNetwork, self).__init__()

    self.model_ver = models.vgg16(weights="DEFAULT")
    self.layers = list(self.model_ver.children())[:-1]
    self.feat_extractor = nn.Sequential(*self.layers)
    
    self.head = nn.Sequential(
      nn.Linear(50176, 8192),
      nn.BatchNorm1d(8192),
      nn.ReLU(inplace=True),
      
      nn.Linear(8192, 1024),
      nn.BatchNorm1d(1024),
      nn.ReLU(inplace=True),

      nn.Linear(1024, 128),
      nn.BatchNorm1d(128),
      nn.ReLU(inplace=True),

      nn.Linear(128, 1),
      nn.Sigmoid()
    )

    self.flatten = nn.Flatten()

  def forward(self, input1, input2):
    """
    Runs Forward Pass

    Parameters:
    -----------
    input1 : torch.Tensor
      (batch_size, in_chans, img_size, img_size)
    input2 : torch.Tensor
      (batch_size, in_chans, img_size, img_size)
    
    Returns:
    --------
    output : torch.Tensor
      (batch_size, prediction)
    """
    output1 = self.feat_extractor(input1)
    output2 = self.feat_extractor(input2)
    
    """Concatenate the image pair"""
    output = torch.cat((output1, output2), 1)
    
    """Flatten to send to FC layer"""
    output = self.flatten(output)
    
    output = self.head(output)

    return output

class ViTNetwork(nn.Module):
  """
  ViT model using pretrained weights

  Parameters:
  -----------

  Attributes:
  -----------
  model_name : string
    Name of pretrained model to use
  encoder : 
    Pretrained ViT model
  head : nn.Sequential
    Regression prediction head.
  """

  def __init__(self):
    super(ViTNetwork, self).__init__()

    self.model_name = 'vit_base_patch16_224' # vit_base_patch16_224
    self.encoder = create_model(self.model_name, pretrained=True)
    
    self.head = nn.Sequential(
        nn.Linear(1536, 384),
        nn.BatchNorm1d(384),
        nn.ReLU(inplace=True),
        
        nn.Linear(384, 96),
        nn.BatchNorm1d(96),
        nn.ReLU(inplace=True),

        nn.Linear(96, 1),
        nn.Sigmoid()
    )

  def forward(self, input1, input2):
    """
    Runs Forward Pass

    Parameters:
    -----------
    input1 : torch.Tensor
      (batch_size, in_chans, img_size, img_size)
    input2 : torch.Tensor
      (batch_size, in_chans, img_size, img_size)
    
    Returns:
    --------
    output : torch.Tensor
      (batch_size, prediction)
    """
    batch_size = input1.shape[0]

    cls_token = self.encoder.cls_token.expand(batch_size, -1, -1)

    """First Image"""
    img1 = self.encoder.patch_embed(input1)

    img1 = torch.cat((cls_token, img1), dim=1) + self.encoder.pos_embed
    img1 = self.encoder.pos_drop(img1)
    
    for block in self.encoder.blocks:
      img1 = block(img1)

    img1 = self.encoder.norm(img1)
    output1 = img1[:,0] # Just the cls token

    """Second Image"""
    img2 = self.encoder.patch_embed(input2)

    img2 = torch.cat((cls_token, img2), dim=1) + self.encoder.pos_embed
    img2 = self.encoder.pos_drop(img2)

    for block in self.encoder.blocks:
      img2 = block(img2)

    img2 = self.encoder.norm(img2)
    output2 = img2[:,0] # Just the cls token

    
    """Concatenate the image pair"""
    output = torch.cat((output1, output2), dim=1)
    
    output = self.head(output)

    return output