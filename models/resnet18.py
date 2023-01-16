import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv,self).__init__()
        padding = kernel_size // 2
        layers = list()
        
        if transpose: # 逆捲積
            padding = kernel_size // 2-1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride,bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)







class ResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        self.module = list(self.resnet18.children())[:-2]
        self.resnet = nn.Sequential(*self.module)


        # for i in range(len(list(self.resnet18.children()))):
        #     print(list(self.resnet18.children())[i])
        #     print("====================")
        
        self.down1 = nn.Sequential(*list(self.resnet18.children())[0:3])

        self.down2 = nn.Sequential(*list(self.resnet18.children())[3:5])

        self.down3 = nn.Sequential(*list(self.resnet18.children())[5])

        self.down4 = nn.Sequential(*list(self.resnet18.children())[6])

        self.down5 = nn.Sequential(*list(self.resnet18.children())[7:-2])
        
        self.up1 = BasicConv(512, 256, kernel_size=4, relu=True, stride=2, transpose=True)
        self.up2 = BasicConv(256, 128, kernel_size=4, relu=True, stride=2, transpose=True)
        self.up3 = BasicConv(128, 64, kernel_size=4, relu=True, stride=2, transpose=True)
        self.up4 = BasicConv(64, 64, kernel_size=4, relu=True, stride=2, transpose=True)
        self.up5 = BasicConv(64, 3, kernel_size=4, relu=False, stride=2, transpose=True)
        

    def forward(self, input):
        x1 = self.down1(input)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        # print(x4.shape)
        # print(x5.shape)
        out = x4+self.up1(x5)
        # print(out.shape)
        out = x3+self.up2(out)
        # print(out.shape)
        out = x2+self.up3(out)
        # print(out.shape)
        out = x1+self.up4(out)
        
        return self.up5(out)