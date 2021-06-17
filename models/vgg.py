#from config import VGG_types
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from torchsummary import summary

VGG_types = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, 'M', 128, 128, 'M', 256, 256,256, "M", 512, 512, 512, 'M', 512, 512, 512, "M"],
    "VGG19": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, "M", 512, 512, 512, 512, 'M', 512, 512, 512, 512, "M"]
        
    }


class Vgg(nn.Module):
    def __init__(self, in_channels = 3, n_cls = 10, architecture = None):
        super(Vgg, self).__init__()
        
        self.in_channels = in_channels
        self.architecture = architecture
        self.conv_layer = self.create_conv_layer(self.architecture)

        self.fc1 = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(4096, n_cls)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x

    def create_conv_layer(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x

                layers += [nn.Conv2d(in_channels = in_channels, 
                            out_channels = out_channels,
                            kernel_size= (3,3), 
                            stride = (1,1), 
                            padding = (1,1)), 
                            nn.BatchNorm2d(x),
                            nn.ReLU()]

                in_channels = x
            
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size = (2,2), stride = (2,2))]


        return nn.Sequential(*layers)   



