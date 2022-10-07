import numpy as np
import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def imshow(img, text=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
        
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    

def vis(train_loss, val_loss):
  """
  Visualization of loss over epochs
  """

  plt.plot(np.arange(len(train_loss)), train_loss, c='r')
  plt.plot(np.arange(len(val_loss)), val_loss, c='y')
  plt.legend(['Training','Validation'])
  plt.grid(True)
  plt.title('Loss over Epochs')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.savefig("log/loss.png")

  print(f"Average Training Loss   : {np.mean(train_loss)}\n")

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