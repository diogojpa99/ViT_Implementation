import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image

dic = 'ISIC_2020_light/train/melanoma_light'

for filename in os.listdir(dic):
    print(dic.split("/"))
    print('-------- File_name --------')
    print(filename)
    
    #print(filename.split("_"))
    #label = int(filename.split("_")[0])
    
    #print(os.path.join(dic, filename))
    #print(label)