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
import numpy as np
import torchvision
import torch.nn as nn
from torchinfo import summary
import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
import matplotlib.pyplot as plt

######################## Classes and Functions #############################

class MyImageDataset(Dataset):
    '''
        Define the Dataset class that will be used to load the data.
    '''
    
    def __init__(self, root_dir, transform = None):
        '''
            Get Images filenames and their corresponding labels.
        '''
        
        self.root_dir = root_dir
        self.transform = transform
    
        # Load the list of image filenames and their corresponding labels
        self.img_filenames=[]
        self.img_labels=[]
        
        for dir in root_dir:
            for filename in os.listdir(dir):      
                # Extract the label from root_dir name:
                if dir.split("/")[-1] == 'melanoma':
                    label = 1
                elif dir.split("/")[-1] == 'benign':
                    label = 0
                    
                # Add the filename and label to the list
                self.img_filenames.append(os.path.join(dir, filename))
                self.img_labels.append(label)
                
        # Generate a random permutation of the data's length
        permut = np.random.permutation(len(self.img_filenames))
        # Convert list into numpy array
        self.img_filenames = np.array(self.img_filenames)
        self.img_labels = np.array(self.img_labels)
        # Rearrange the indexes given the permutation
        self.img_filenames = self.img_filenames[permut]
        self.img_labels = self.img_labels[permut]
        # Convert numpy array into list
        self.img_filenames = self.img_filenames.tolist()
        self.img_labels = self.img_labels.tolist()

    def __len__(self):
        '''
            Get the number of images.
        '''
        return len(self.img_filenames)
        
    def __getitem__(self, idx):
        '''
            Get the images and respective labels
        '''
        # Load image and label
        img = Image.open(self.img_filenames[idx])
        label = self.img_labels[idx]
        # Apply any transformations (if provided)
        if self.transform:
            img = self.transform(img)

        return img, label
    

######################## Initializations #############################

# Set the random seed
np.random.seed(42)

# Defining the directories
train_dir = ['ISIC_2020/train/melanoma', 'ISIC_2020/train/benign']

# Labels
labels_map = { 0: "benign", 1: "melanoma" }
labels_map_inv = { "benign": 0, "melanoma": 1}

############################# main() ##################################

# (1)
# Download pretrained ViT weights and model
vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT 
pretrained_vit = torchvision.models.vision_transformer.vit_b_16(vit_weights)

# (2)
# Freeze all layers in the pretrained model
for param in pretrained_vit.parameters():
  param.requires_grad = False

# (3)
# Update the pretrained ViT head
embedding_dim = 768 # ViT_Base
num_classes = 2
#set_seeds()
pretrained_vit.heads = nn.Sequential(
  nn.LayerNorm(normalized_shape=embedding_dim),
  nn.Linear(in_features=embedding_dim, out_features=num_classes)
)

# (4)
# Print a summary using torchinfo (uncomment for actual output)
summary(model=pretrained_vit, 
         input_size=(1, 3, 224, 224), # (batch_size, color_channels, height, width)
         # col_names=["input_size"], # uncomment for smaller output
         col_names=["input_size", "output_size", "num_params", "trainable"],
         col_width=20,
         row_settings=["var_names"]
)

# (5)
# Preprocess the data
vit_transforms = vit_weights.transforms()

# (6)
# Get preprocess data
# Create the Dataset object that will be used to load the data
dataset = MyImageDataset(train_dir, transform=vit_transforms)

# Split the data into training, validation, and test sets
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders for the datasets
train_loader = DataLoader(train_dataset, batch_size=120, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=120, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=120, shuffle=False)
#print(len(train_loader),len(val_loader),len(test_loader))

# (7)
# Fine-tune a pretrained ViT feature extractor  