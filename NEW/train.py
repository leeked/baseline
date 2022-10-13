import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm
import os
import argparse
from utils import DEVICE, vis, cnnDataset
from base_models import VGGNetwork, ViTNetwork, VGGFeat, ViTFeat

parser = argparse.ArgumentParser()
parser.add_argument('epochs', type=int, help="number of epochs to train")
parser.add_argument('--vgg', action="store_true")
parser.add_argument('--vit', action="store_true")
parser.add_argument('--path', type=str)
args = parser.parse_args()

"""

Train

"""
def train(model, feat, pos: pd.DataFrame, neg: pd.DataFrame, training_dir: str, vdl: cnnDataset, num_epochs: int) -> dict:
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
  loss_fn = nn.BCELoss()

  last_best = np.inf
  best_so_far = None

  start_ind = 0
  end_ind = len(pos)

  t_train_loss, t_val_loss = [], []

  for epoch in range(num_epochs):
    """Determine Balanced Training Set"""
    idx = np.arange(start_ind, end_ind, 1) % len(neg)

    curr_train_set = pd.concat([pos, neg.iloc[idx]])

    training_dataset = cnnDataset(curr_train_set, 
                                    training_dir,
                                    transform=transforms.Compose([transforms.Resize((224,224)),
                                                                transforms.ToTensor()]))
    tdl = DataLoader(training_dataset,
                                    shuffle=True,
                                    num_workers=2,
                                    batch_size=16)

    """Training Loop"""
    model.train()
    feat.eval()
    print(f"EPOCH: {epoch}")
    val_losses, train_losses = [], []
    for img0, img1,  label in tqdm(tdl, total=len(tdl)):
      img0, img1, label = img0.to(DEVICE), img1.to(DEVICE), label.to(DEVICE)

      optimizer.zero_grad()

      extracted = feat(img0, img1)

      preds = model(extracted)
      loss = loss_fn(preds, label)
      loss.backward()
      train_losses.append(loss.detach().cpu().numpy())
      optimizer.step()

    """Validation Loop"""
    with torch.no_grad():
      model.eval()
      feat.eval()
      for (img0, img1, label) in tqdm(vdl, total=len(vdl)):
        img0, img1, label = img0.to(DEVICE), img1.to(DEVICE), label.to(DEVICE)

        extracted = feat(img0, img1)

        preds = model(extracted)
        loss = loss_fn(preds, label)
        val_losses.append(loss.detach().cpu().numpy())

    print(f'\n Training Loss: {np.mean(train_losses)}\n Validation Loss: {np.mean(val_losses)}\n')
    t_train_loss.append(np.mean(train_losses))
    t_val_loss.append(np.mean(val_losses))

    """Save the best model so far"""
    if np.mean(val_losses) < last_best:
      best_so_far = model.state_dict()
      last_best = np.mean(val_losses)
    
    start_ind += len(pos)
    end_ind += len(pos)

  vis(t_train_loss, t_val_loss)
  return best_so_far

def main():
  print("Initializing dataset...")
  full_train_df = pd.read_csv('data/training_csv.csv')
  full_train_df.columns = ["image1","image2","label"]

  training_dir = 'data/training'

  """Determining Train/Val split"""

  validation_df = full_train_df.groupby('label', group_keys=False).apply(lambda x: x.sample(frac=0.2))
  training_df = pd.concat([full_train_df, validation_df]).drop_duplicates(keep=False)

  class_groups = training_df.groupby('label')
  neg, pos = [class_groups.get_group(x) for x in class_groups.groups]

  validation_df.reset_index(drop=True, inplace=True)
  pos.reset_index(drop=True, inplace=True)
  neg.reset_index(drop=True, inplace=True)

  validation_dataset = cnnDataset(validation_df,
                          training_dir,
                          transform=transforms.Compose([transforms.Resize((224,224)),
                                                        transforms.ToTensor()]))

  vdl = DataLoader(validation_dataset,
                          shuffle=True,
                          num_workers=2,
                          batch_size=16)
  
  print("Initializing model(s)...")
  file_name = ""
  if args.vgg:
    print("Model: VGG")
    model = VGGNetwork().to(DEVICE)
    feat = VGGFeat().to(DEVICE)
    file_name = "vgg_model.pth"
  if args.vit:
    print("Model: ViT")
    model = ViTNetwork().to(DEVICE)
    feat = ViTFeat().to(DEVICE)
    file_name = "vit_model.pth"
  if not args.vgg and not args.vit:
    raise Exception("Need to specify model (vgg, vit)")

  print("Entering train...")
  print(f"Training for {args.epochs} epochs.")
  res_model = train(
      model,
      feat,
      pos,
      neg,
      training_dir,
      vdl,
      num_epochs=args.epochs,
  )

  print("Saving...")
  torch.save(res_model, args.path + file_name)


if __name__ == "__main__":
  main()
