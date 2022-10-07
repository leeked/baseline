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
from base_models import VGGNetwork, ViTNetwork

parser = argparse.ArgumentParser()
parser.add_argument('--vgg', action="store_true")
parser.add_argument('--vit', action="store_true")
parser.add_argument('--path', type=str)
args = parser.parse_args()

def test(model: VGGNetwork, test_dataloader: cnnDataset):
  loss_fn = nn.BCELoss()

  acc = []
  test_loss = []

  with torch.no_grad():
    model.eval()
    for img0, img1, label in tqdm(test_dataloader, total=len(test_dataloader)):
      img0, img1, label = img0.to(DEVICE), img1.to(DEVICE), label.to(DEVICE)

      output = model(img0, img1)

      loss = loss_fn(output, label)
      test_loss.append(loss.item())

      cut = np.array([int(x >= 0.5) for x in output.detach().cpu().numpy()]).reshape((len(label), 1))

      batch_acc = np.sum(np.equal(cut, label.detach().cpu().numpy())) / len(cut)

      acc.append(batch_acc)

  acc = np.array(acc)
  test_loss = np.array(test_loss)
  print(f"\nAverage Test Accuracy across batches: {np.mean(acc)}\nAverage Test Loss across batches: {np.mean(test_loss)}")
  return np.mean(acc), np.mean(test_loss)

def main():
  print("Initializing dataset...")
  testing_df = pd.read_csv('data/testing_csv.csv')
  testing_df.columns = ["image1","image2","label"]

  testing_dir = 'data/testing'

  testing_dataset = cnnDataset(testing_df,
                            testing_dir,
                            transform=transforms.Compose([transforms.Resize((224,224)),
                                                          transforms.ToTensor()]))

  test_dataloader = DataLoader(testing_dataset,
                      shuffle=True,
                      num_workers=2,
                      batch_size=16)

  print("Initializing model...")
  if args.vgg:
    model = VGGNetwork().to(DEVICE)
    model.load_state_dict(torch.load(args.path + "vgg_model.pth"))
  if args.vit:
    model = ViTNetwork().to(DEVICE)
    model.load_state_dict(torch.load(args.path + "vit_model.pth"))
  if not args.vgg and not args.vit:
    raise Exception("Need to specify model (vgg, vit)")

  print("Entering test...")
  acc, loss = test(
      model,
      test_dataloader,
  )

  print("Saving...")
  with open("log/test_log.txt", "w") as f:
    f.write("Acc: " + str(acc) + "\nLoss: " + str(loss))


if __name__ == "__main__":
  main()