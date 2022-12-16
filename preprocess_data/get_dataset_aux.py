import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image


# Defining the directories
dic_train_mela = 'ISIC_2020/train/melanoma'
dic_train_benign = 'ISIC_2020/train/benign'
dic_test_mela = 'ISIC_2020/validation/melanoma'
dic_test_benign = 'ISIC_2020/validation/benign'

dic = dic_train_benign

for filename in os.listdir(dic):
        
    # Define the label
    if dic.split("/")[-1] == 'melanoma':
        label = 1
    elif dic.split("/")[-1] == 'benign':
        label = 0
    