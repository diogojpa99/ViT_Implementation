import numpy as np
import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
       

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
            

# Set the random seed
np.random.seed(42)

# Defining the directories
train_dir = ['ISIC_2020/train/melanoma', 'ISIC_2020/train/benign']

# Create the Dataset object that will be used to load the data
dataset = MyImageDataset(train_dir, transform=Compose([
    Resize((256, 256)),
    ToTensor()
]))

# Split the data into training, validation, and test sets
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders for the datasets
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
