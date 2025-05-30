import torch.nn as nn
import torch
import torch.nn.functional as F
from math import prod
from utils.logger import LOGGER_SINGLETON, DEBUG


## WEIGHTS INITIAlIZER, M=0, STD=0.02
## INPUT_SIZE = Bx100
## OUTPUT_SIZE = Bx3x64x64
## Z IS SAMPLED FROM A UNIFORM DISTRIBUTION
class Generator(nn.Module):
    def __init__(self, latent_space_size, initial_fmaps_size ,lrelu_slope):
        super(Generator, self).__init__()
        LATENT_SPACE_SIZE = latent_space_size
        INITIAL_FMAPS_SIZE = initial_fmaps_size
        FMAPS_SIZES = [int(INITIAL_FMAPS_SIZE/2**i) for i in range(4)]
        FMAPS_SIZES.append(3)
        UP_PROJECTION_SIZE = (FMAPS_SIZES[0], 4,4)
        LEAKY_RELU_SLOPE = lrelu_slope

        
        #layer 1
        self.layer1 = nn.Sequential(
            nn.Linear(LATENT_SPACE_SIZE, prod(UP_PROJECTION_SIZE)),
            nn.BatchNorm1d(prod(UP_PROJECTION_SIZE)),
            nn.LeakyReLU(LEAKY_RELU_SLOPE)
        )
        

        self.reshape = nn.Unflatten(1, UP_PROJECTION_SIZE)
        
        #layer 2
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(FMAPS_SIZES[0], FMAPS_SIZES[1], kernel_size= (5,5), bias=False),
            nn.BatchNorm2d(FMAPS_SIZES[1]),
            nn.LeakyReLU(LEAKY_RELU_SLOPE)
        )

        #layer 3
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(FMAPS_SIZES[1], FMAPS_SIZES[2], kernel_size= (4,4), bias=False, stride=2, padding=1),
            nn.BatchNorm2d(FMAPS_SIZES[2]),
            nn.LeakyReLU(LEAKY_RELU_SLOPE)
        )
        

        #layer 4
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(FMAPS_SIZES[2], FMAPS_SIZES[3], kernel_size= (4,4), padding=1,bias=False, stride=2),
            nn.BatchNorm2d(FMAPS_SIZES[3]),
            nn.LeakyReLU(LEAKY_RELU_SLOPE)
        )

        #layer 5
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(FMAPS_SIZES[3], FMAPS_SIZES[4], kernel_size= (4,4), padding=1,  bias=True, stride=2),
            nn.Tanh()
        )

    def forward(self, x:torch.tensor):
        # upprojection
        DEBUG("Gen input SIZE", x.size())
        x = self.layer1(x)
        #output size: B,FMAPS[0],4,4

        #tconv1
        DEBUG("SIZE after tconv1", x.size())
        x = self.reshape(x)
        x = self.layer2(x)
        #output size: B,FMAPS[1],8,8

        #tconv2
        DEBUG("SIZE after tconv2", x.size())
        x = self.layer3(x)
        #output size: B,FMAPS[2],16,16

        #tconv3
        DEBUG("SIZE after tconv3", x.size())
        x = self.layer4(x)
        #output size: B,FMAPS[3],32,32

        #tconv4
        DEBUG("SIZE after tconv4", x.size())
        x = self.layer5(x)
        #output size: B,FMAPS[4],64,64
        DEBUG("Gen output SIZE", x.size())
        return x
        

    def _initialize_weights(self, w_mean, w_std, bn_mean, bn_std) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, w_mean, w_std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, bn_mean, bn_std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
