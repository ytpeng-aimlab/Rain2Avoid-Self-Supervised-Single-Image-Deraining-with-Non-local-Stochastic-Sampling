import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun


class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]


class PCBActiv(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, sample='none-3', activ='relu',
                 conv_bias=False):
        super().__init__()
        if sample == 'down-5':
            self.conv = nn.Conv2d(in_ch, out_ch, 5, 2, 2, dilation=1, groups=1, bias=conv_bias)
        elif sample == 'down-7':
            self.conv = nn.Conv2d(in_ch, out_ch, 7, 2, 3, dilation=1, groups=1, bias=conv_bias)
        elif sample == 'down-3':
            self.conv = nn.Conv2d(in_ch, out_ch, 3, 2, 1, dilation=1, groups=1, bias=conv_bias)
        else:
            self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1, dilation=1, groups=1, bias=conv_bias)

        if bn:
            self.bn = nn.BatchNorm2d(out_ch)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input):
        h = self.conv(input)
        if hasattr(self, 'bn'):
            h = self.bn(h)
        if hasattr(self, 'activation'):
            h = self.activation(h)
        return h


class UNet(nn.Module):
    def __init__(self, layer_size=4, input_channels=3, upsampling_mode='nearest'):
        super().__init__()
        self.freeze_enc_bn = False
        self.upsampling_mode = upsampling_mode
        self.layer_size = layer_size
        self.enc_1 = PCBActiv(input_channels, 64, bn=False, sample='down-7')
        self.enc_2 = PCBActiv(64, 128, sample='down-5')
        self.enc_3 = PCBActiv(128, 256, sample='down-5')
        self.enc_4 = PCBActiv(256, 512, sample='down-3')
        
        self.dec_4 = PCBActiv(512 + 256, 256, activ='leaky')
        self.dec_3 = PCBActiv(256 + 128, 128, activ='leaky')
        self.dec_2 = PCBActiv(128 + 64, 64, activ='leaky')
        self.dec_1 = PCBActiv(64 + input_channels, input_channels,
                              bn=False, activ=None, conv_bias=True)

    def forward(self, input):
        h_dict = {}  # for the output of enc_N
        h_dict['h_0']= input
        h_key_prev = 'h_0'
        for i in range(1, self.layer_size + 1):
            l_key = 'enc_{:d}'.format(i)
            h_key = 'h_{:d}'.format(i)
            h_dict[h_key] = getattr(self, l_key)(
                h_dict[h_key_prev])
            h_key_prev = h_key

        h_key = 'h_{:d}'.format(self.layer_size)
        h = h_dict[h_key]

        

        for i in range(self.layer_size, 0, -1):
            enc_h_key = 'h_{:d}'.format(i - 1)
            dec_l_key = 'dec_{:d}'.format(i)
            h = F.interpolate(h, scale_factor=2, mode=self.upsampling_mode)
            h = torch.cat([h, h_dict[enc_h_key]], dim=1)
            h= getattr(self, dec_l_key)(h)

        return h

    def train(self, mode=True):
        
        super().train(mode)
        if self.freeze_enc_bn:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d) and 'enc' in name:
                    module.eval()

class DnCNN(nn.Module):
    def __init__(self, channels=3, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out

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
        
        self.resnet18 = models.resnet18(pretrained=False)
        self.module = list(self.resnet18.children())[:-2]
        self.resnet = nn.Sequential(*self.module)
        
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
        out = x4+self.up1(x5)
        out = x3+self.up2(out)
        out = x2+self.up3(out)
        out = x1+self.up4(out)
        return self.up5(out)