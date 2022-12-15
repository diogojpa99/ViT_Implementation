import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image

# Define the Dataset class that will be used to load the data
class MyImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Load the list of image filenames and their corresponding labels
        self.img_filenames = []
        self.img_labels = []
        for filename in os.listdir(root_dir):
            # Extract the label from the filename
            #label = int(filename.split("_")[0])
            
            # Extract the label from root_dir name:
            if root_dir.split("/")[-1] == 'melanoma':
                label = 1
            elif root_dir.split("/")[-1] == 'benign':
                label = 0

            # Add the filename and label to the list
            self.img_filenames.append(os.path.join(root_dir, filename))
            self.img_labels.append(label)

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        # Load the image and label
        img = Image.open(self.img_filenames[idx])
        label = self.img_labels[idx]

        # Apply any transformations (if provided)
        if self.transform:
            img = self.transform(img)

        return img, label

# Create the Dataset object that will be used to load the data
dataset = MyImageDataset('ISIC_2020/train/benign', transform=Compose([
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

# Load the ViT model from PyTorch
model = torchvision.models.vision.vit.vit_base()

# Set the model to training mode
model.train()

# Use the Adam optimizer to optimize the model's parameters
optimizer = torch.optim.Adam(model.parameters())

# Define the loss function
loss_fn = torch.nn.CrossEntropyLoss()

# Train the model for 10 epochs
for epoch in range(10):
    # Loop over the training data
    for images, labels in train_loader:
        # Forward pass
        outputs = model(images)

       
