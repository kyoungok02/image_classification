from lzma import PRESET_DEFAULT
import torch
import torch.nn as nn
import torch.nn.functional as F

# dataset and transformation
from ops import dataset, models
import os
import torchvision.transforms as transforms

# display images
from torchvision import utils
import matplotlib.pyplot as plt

# load images
from PIL import Image

# utils
import numpy as np
import time
import copy

# model
# from networks import vgg,inception

# for parser
import argparse

torch.manual_seed(0)

# train device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# for apple m1
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
# device = 'cpu'

def main():
    
    # load image
    with Image.open("./test_image/1.jpg") as im:
        im.resize([244,244])
    # trans = transforms.ToPILImage()
    transformer = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Resize((224,224)),
    ])                            
    im = transformer(im)
    im = im.to(device)
    # load_model
    model = models.load_model("VGG11",3,10,True,device,False)
    # load_weight
    model.load_state_dict(torch.load("./checkpoint/VGG_STL10_e50_mps_02d17:28"))
    model.eval()
    
    pred = model(im.unsqueeze(0))
    print(pred.argmax(1).item())
    
if __name__ == '__main__':
    main()
    
