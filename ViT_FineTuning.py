"""
/*******************************************************************************************
 _____
|     |             Authors: Diogo Araújo (ist193906)
| IST | TECNICO     
 \   /  LISBOA      Description: Pre-trained ViT Fine-Tuning implementation
  \ /               
  
*******************************************************************************************/
"""

############################ Import Libraries ##############################

from PIL import Image
import torch
import numpy as np
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
import seaborn as sn
from sklearn.metrics import confusion_matrix, mean_squared_error, classification_report
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
from torchinfo import summary
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision

######################## Classes and Functions #############################

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str, 
    test_dir: str, 
    transform: transforms.Compose, 
    batch_size: int, 
    num_workers: int=NUM_WORKERS
):
  """Creates training and testing DataLoaders.
  Takes in a training directory and testing directory path and turns
  them into PyTorch Datasets and then into PyTorch DataLoaders.
  Args:
    train_dir: Path to training directory.
    test_dir: Path to testing directory.
    transform: torchvision transforms to perform on training and testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.
  Returns:
    A tuple of (train_dataloader, test_dataloader, class_names).
    Where class_names is a list of the target classes.
    Example usage:
      train_dataloader, test_dataloader, class_names = \
        = create_dataloaders(train_dir=path/to/train_dir,
                             test_dir=path/to/test_dir,
                             transform=some_transform,
                             batch_size=32,
                             num_workers=4)
  """
  # Use ImageFolder to create dataset(s)
  train_data = datasets.ImageFolder(train_dir, transform=transform)
  test_data = datasets.ImageFolder(test_dir, transform=transform)

  # Get class names
  class_names = train_data.classes

  # Turn images into data loaders
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True,
  )

  return train_dataloader, test_dataloader, class_names

######################## Initializations ############################

# Set the random seed
np.random.seed(42)

# Defining the directories
train_dir = ['ISIC_2020/train/melanoma', 'ISIC_2020/train/benign']

# Labels
labels_map = { 0: "benign", 1: "melanoma" }
labels_map_inv = { "benign": 0, "melanoma": 1}

############################# main() ##################################

# (1)
# initializing feature extractor:
#feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT 
vit_transforms = vit_weights.transforms()

dataset = create_dataloaders('ISIC_2020/train/melanoma','ISIC_2020/train/melanoma', vit_transforms,batch_size=1024)
