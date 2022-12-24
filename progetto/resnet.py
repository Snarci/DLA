# write a resnet 18 model from scratch using pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import *

class ResNet18(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.2)
        self.maxpool = nn.MaxPool2d(kernel_size=7, stride=2, padding=3)
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_residual_blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, downsample=True))
        for _ in range(num_residual_blocks - 1):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=7, stride=1, padding=3)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.2)
        self.downsample = downsample
        if self.downsample:
            self.downsample_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
            self.downsample_bn = nn.BatchNorm2d(out_channels)
        self.attention = CBAMBlock(out_channels)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample:
            identity = self.downsample_conv(identity)
            identity = self.downsample_bn(identity)
        x += identity
        x = self.activation(x)
        x = self.attention(x)
        x = self.dropout(x)
        return x

#if __name__ == '__main__':
    #model = ResNet18()
    #x = torch.randn(2, 3, 224, 224)
    #print(model(x).shape)