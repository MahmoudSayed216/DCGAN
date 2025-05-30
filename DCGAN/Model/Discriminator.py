import torch
import torch.nn as nn

from utils.logger import LOGGER_SINGLETON, DEBUG

class Discriminator(nn.Module):
    def __init__(self, initial_fmaps_size ,lrelu_slope):
        super(Discriminator, self).__init__()
        INITIAL_FMAPS_SIZE = initial_fmaps_size
        FMAPS_SIZES = [INITIAL_FMAPS_SIZE*(2**i) for i in range(4)]
        FMAPS_SIZES.append(1)
        LEAKY_RELU_SLOPE = lrelu_slope

        #layer 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, FMAPS_SIZES[0], kernel_size=(4,4), stride=(2,2), padding=(1,1), bias=False),
            nn.LeakyReLU(LEAKY_RELU_SLOPE)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(FMAPS_SIZES[0], FMAPS_SIZES[1], kernel_size=(4,4), stride=(2,2), padding=(1,1), bias=False),
            nn.BatchNorm2d(FMAPS_SIZES[1]),
            nn.LeakyReLU(LEAKY_RELU_SLOPE)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(FMAPS_SIZES[1], FMAPS_SIZES[2], kernel_size=(4,4), stride=(2,2), padding=(1,1), bias=False),
            nn.BatchNorm2d(FMAPS_SIZES[2]),
            nn.LeakyReLU(LEAKY_RELU_SLOPE)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(FMAPS_SIZES[2], FMAPS_SIZES[3], kernel_size=(4,4), stride=(2,2), padding=(1,1), bias=False),
            nn.BatchNorm2d(FMAPS_SIZES[3]),
            nn.LeakyReLU(LEAKY_RELU_SLOPE)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(FMAPS_SIZES[3], FMAPS_SIZES[4], kernel_size=(4,4), stride=(1,1), bias=False),
            nn.Sigmoid()
        )

        # self.flatten = nn.Flatten(FMAPS_SIZES)

    def forward(self, x):
        DEBUG("Disc input SIZE", x.size())
        
        #conv1
        x = self.layer1(x)
        #output size: BxFMAPS[0]x32x32

        #conv2
        DEBUG("SIZE after conv2", x.size())
        x = self.layer2(x)
        #output size: BxFMAPS[1]x16x16

        
        #conv3
        DEBUG("SIZE after conv3", x.size())
        x = self.layer3(x)
        #output size: BxFMAPS[2]x8x8

        #conv4
        DEBUG("SIZE after conv4", x.size())
        x = self.layer4(x)
        #output size: BxFMAPS[3]x4x4

        #conv5
        DEBUG("SIZE after conv5", x.size())
        x = self.layer5(x)
        #output size: BxFMAPS[4]x1x1

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
