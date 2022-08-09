from pathlib import Path
import string
import numpy as np
import pandas as pd
import os
from PIL import Image
import PIL.ImageOps

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.utils

from tqdm import tqdm
from utils import DEVICE


"""

Model

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

        self.model_ver = models.vgg16(pretrained=True)
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


"""

Dataset

"""
class cnnDataset():
    def __init__(self,input_df=None,input_dir=None,transform=None):
        # used to prepare the labels and images path
        self.input_df=input_df
        #self.input_df.columns =["image1","image2","label"]
        self.input_dir = input_dir    
        self.transform = transform

    def __getitem__(self,index):
        # getting the image path
        image1_path=os.path.join(self.input_dir,self.input_df.iat[index,0])
        image2_path=os.path.join(self.input_dir,self.input_df.iat[index,1])
        # Loading the image
        img0 = Image.open(image1_path)
        img1 = Image.open(image2_path)
        # Apply image transformations
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        return img0, img1 , torch.from_numpy(np.array([int(self.input_df.iat[index,2])],dtype=np.float32))
        
    def __len__(self):
        return len(self.input_df)

"""

Train

"""
def train(model: VGGNetwork, pos: pd.DataFrame, neg: pd.DataFrame, training_dir: string, vdl: cnnDataset, num_epochs: int) -> dict:

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.BCELoss()

    last_best = np.inf
    best_so_far = None

    start_ind = 0
    end_ind = len(pos)

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
        print(f"EPOCH: {epoch}")
        val_losses, train_losses = [], []
        for img0, img1,  label in tqdm(tdl, total=len(tdl)):
            img0, img1, label = img0.to(DEVICE), img1.to(DEVICE), label.to(DEVICE)

            optimizer.zero_grad()

            preds = model(img0, img1)
            loss = loss_fn(preds, label)
            loss.backward()
            train_losses.append(loss.detach().cpu().numpy())
            optimizer.step()

        """Validation Loop"""
        with torch.no_grad():
            model.eval()
            for (img0, img1, label) in tqdm(vdl, total=len(vdl)):
                img0, img1, label = img0.to(DEVICE), img1.to(DEVICE), label.to(DEVICE)

                preds = model(img0, img1)
                loss = loss_fn(preds, label)
                val_losses.append(loss.detach().cpu().numpy())

        print(f'\n Training Loss: {np.mean(train_losses)}\n Validation Loss: {np.mean(val_losses)}\n')

        """Save the best model so far"""
        if np.mean(val_losses) < last_best:
            best_so_far = model.state_dict()
            last_best = np.mean(val_losses)
        
        start_ind += len(pos)
        end_ind += len(pos)

    return best_so_far


def main():
    full_train_df = pd.read_csv('/data/training_csv.csv')
    full_train_df.columns = ["image1","image2","label"]

    training_dir = '/data/training'

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

    model = VGGNetwork().to(DEVICE)

    vgg_model = train(
        model,
        pos,
        neg,
        training_dir,
        vdl,
        num_epochs=30,
    )
    torch.save(vgg_model, "/model/vgg_model.pth")


if __name__ == "__main__":
    main()
