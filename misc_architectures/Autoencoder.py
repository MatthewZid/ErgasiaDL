# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F

class ConvAutoencoder(nn.Module):
    def __init__(self, extractor = False):
        super(ConvAutoencoder, self).__init__()
        self.extractor = extractor

        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 16, 
                               kernel_size = (2, 2), 
                               stride = (1, 1),
                               padding = (1, 1),
                               dilation = (1, 1))
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, 
                               kernel_size = (3, 3), 
                               stride = (2, 1),
                               padding = (1, 1),
                               dilation = (1, 1))
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 32, 
                               kernel_size = (3, 3), 
                               stride = (3, 1),
                               padding = (1, 1),
                               dilation = (1, 1)) 
        self.pool1 = nn.MaxPool2d(kernel_size = (4, 2), 
                                 stride = 2)      
        self.pool2 = nn.MaxPool2d(kernel_size = 4, 
                                 stride = 2)
        self.act = nn.LeakyReLU()
        
        self.tconv1 = nn.ConvTranspose2d(in_channels = 32, out_channels = 32, 
                               kernel_size = (7, 1), 
                               stride = (2, 2),
                               padding = (0, 0),
                               dilation = (1, 1))
        self.tconv2 = nn.ConvTranspose2d(in_channels = 32, out_channels = 16, 
                               kernel_size = (5, 1), 
                               stride = (3, 1),
                               padding = (1, 0),
                               dilation = (1, 1))
        self.tconv3 = nn.ConvTranspose2d(in_channels = 16, out_channels = 8, 
                               kernel_size = (5, 1), 
                               stride = (3, 1),
                               padding = (0, 0),
                               dilation = (1, 1))        
        self.tconv4 = nn.ConvTranspose2d(in_channels = 8, out_channels = 1, 
                               kernel_size = (2, 1), 
                               stride = (2, 1),
                               padding = (1, 0),
                               dilation = (1, 1))        
        

    def forward(self, x):
        ## encode ##
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.pool1(x)  
        x = self.act(self.conv3(x))
        x = F.max_pool2d(x, kernel_size = (11, 1))
        # compressed representation
        if self.extractor:
            return x
        ## decode ##
        x = self.act(self.tconv1(x))
        x = self.act(self.tconv2(x))
        x = self.act(self.tconv3(x))
        x = self.act(self.tconv4(x))
        return x