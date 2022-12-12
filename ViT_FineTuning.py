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
import torchvision

# (2)
# Download pretrained ViT weights and model
vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT 
pretrained_vit = torchvision.models.vit_b_16(weigths = vit_weights)

# (3)
# Freeze all layers in the pretrained model



