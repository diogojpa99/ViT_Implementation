"""
/*******************************************************************************************
 _____
|     |             Authors: Diogo Araújo (ist193906)
| IST | TECNICO     
 \   /  LISBOA      Description: Pre-trained ViT Fine-Tuning implementation
  \ /               
  
*******************************************************************************************/
"""

# (1)
# Import the necessary libraries
from PIL import Image
import numpy as np
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from torchinfo import summary
from PIL import Image
import numpy as np
import torch
from glob import glob

# (2)
# Download pretrained ViT weights and model
vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT 
pretrained_vit = torchvision.models.vision_transformer.vit_b_16(vit_weights)

# (3)
# Freeze all layers in the pretrained model
for param in pretrained_vit.parameters():
  param.requires_grad = False

# (4)
# Update the pretrained ViT head
embedding_dim = 768 # ViT_Base
num_classes = 2
#set_seeds()
pretrained_vit.heads = nn.Sequential(
  nn.LayerNorm(normalized_shape=embedding_dim),
  nn.Linear(in_features=embedding_dim, out_features=num_classes)
)

# (5)
# Print a summary using torchinfo (uncomment for actual output)
summary(model=pretrained_vit, 
         input_size=(1, 3, 224, 224), # (batch_size, color_channels, height, width)
         # col_names=["input_size"], # uncomment for smaller output
         col_names=["input_size", "output_size", "num_params", "trainable"],
         col_width=20,
         row_settings=["var_names"]
)

# (6)
# Import the data(Images)
def import_img(img_path, label_flag):
  
  images = []
  for path in img_path:
    img = plt.imread(path)
    img_array = np.array(img)
    images.append(img_array)
    
  images = np.array(images)
  if label_flag == 0: 
    labels = np.zeros(images.shape[0])
  elif label_flag == 1:
    labels = np.ones(images.shape[0])
      
  return images, labels
  
# Get the list of image paths in the directory
benign_train = glob('ISIC_2020_light/train/benign_light/*.jpg')
melanoma_train = glob('ISIC_2020_light/train/melanoma_light/*.jpg')
benign_test = glob('ISIC_2020_light/val/benign_light/*.jpg')
melanoma_test = glob('ISIC_2020_light/val/melanoma_light/*.jpg')

x_train_benign, y_train_benign = import_img(benign_train, 0)
x_train_mela, y_train_mela = import_img(melanoma_train, 1)
x_test_benign, y_test_benign = import_img(benign_test, 0)
x_test_mela, y_test_mela = import_img(melanoma_test, 1)

# (7)
# Preprocess the data
vit_transforms = vit_weights.transforms()

