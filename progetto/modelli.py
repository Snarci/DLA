import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_Luca_Massi(nn.Module):
    def __init__(self, num_classes=10, in_channels=3, img_size=256,pooling_method='max', kernel_size=3,stride=1, padding=1, activation='gelu', dropout=0.2):
        super().__init__()
        self.pooling_method = pooling_method
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.img_size = img_size
        self.depth = 0

        # define pooling layers
        if pooling_method == 'max':
            self.pool = nn.MaxPool2d(2, 2)
        elif pooling_method == 'avg':
            self.pool = nn.AvgPool2d(2, 2)
        #define activation function
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()   

        # define convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size, stride, padding)
        self.conv1_repeat = nn.Conv2d(32, 32, kernel_size, stride, padding)

        self.conv2 = nn.Conv2d(32, 64, kernel_size, stride, padding)
        self.conv2_repeat = nn.Conv2d(64, 64, kernel_size, stride, padding)

        self.conv3 = nn.Conv2d(64, 128, kernel_size, stride, padding)
        self.conv3_repeat = nn.Conv2d(128, 128, kernel_size, stride, padding)

        self.conv4 = nn.Conv2d(128, 256, kernel_size, stride, padding)
        self.conv4_repeat = nn.Conv2d(256, 256, kernel_size, stride, padding)

        self.conv5 = nn.Conv2d(256, 512, kernel_size, stride, padding)
        self.conv5_repeat = nn.Conv2d(512, 512, kernel_size, stride, padding)

        self.conv6 = nn.Conv2d(512, 1024, kernel_size, stride, padding)
        self.conv6_repeat = nn.Conv2d(1024, 1024, kernel_size, stride, padding)


        # define dropout layer
        self.dropout = nn.Dropout(dropout)
        # regularization
        self.layer_norm = nn.LayerNorm([3, img_size, img_size])
        self.layer_norm1 = nn.LayerNorm([32, img_size, img_size])
        self.layer_norm2 = nn.LayerNorm([64, img_size//2, img_size//2])
        self.layer_norm3 = nn.LayerNorm([128, img_size//4, img_size//4])
        self.layer_norm4 = nn.LayerNorm([256, img_size//8, img_size//8])
        self.layer_norm5 = nn.LayerNorm([512, img_size//16, img_size//16])
        self.layer_norm6 = nn.LayerNorm([1024, img_size//32, img_size//32])
        # global average pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # fully connected layer
        self.fc = nn.Linear(1024, num_classes)
        


    def forward(self, x):
        #N, C, H, W = x.shape
        # first block of convolutions N, 3, 256, 256
        x = self.layer_norm(x)
        x = self.dropout(self.activation(self.conv1(x)))
        residual = x
        for i in range(self.depth):
            x = self.dropout(self.activation(self.conv1_repeat(x)))+x
        if self.depth >0:
            x = self.layer_norm1(x+residual)
        x = self.pool(x)
        # second block of convolutions N, 32, 128, 128
        x = self.dropout(self.activation(self.conv2(x)))
        residual = x
        for i in range(self.depth):
            x = self.dropout(self.activation(self.conv2_repeat(x)))+x
        if self.depth >0:
            x = self.layer_norm2(x+residual)
        x = self.pool(x)
        # third block of convolutions N, 64, 64, 64
        x = self.dropout(self.activation(self.conv3(x)))
        residual = x
        for i in range(self.depth):
            x = self.dropout(self.activation(self.conv3_repeat(x)))+x
        if self.depth >0:
            x = self.layer_norm3(x+residual)
        x = self.pool(x)
        # fourth block of convolutions N, 128, 32, 32
        x = self.dropout(self.activation(self.conv4(x)))
        residual = x
        for i in range(self.depth):
            x = self.dropout(self.activation(self.conv4_repeat(x)))+x
        if self.depth >0:
            x = self.layer_norm4(x+residual)
        x = self.pool(x)
        # fifth block of convolutions N, 256, 16, 16
        x = self.dropout(self.activation(self.conv5(x)))
        residual = x
        for i in range(self.depth):
            x = self.dropout(self.activation(self.conv5_repeat(x)))+x
        if self.depth >0:
            x = self.layer_norm5(x+residual)
        x = self.pool(x)
        # 512, 8, 8
        x = self.dropout(self.activation(self.conv6(x)))
        residual = x
        for i in range(self.depth):
            x = self.dropout(self.activation(self.conv6_repeat(x)))+x
        if self.depth >0:
            x = self.layer_norm6(x+residual)
        x = self.pool(x)
        # 1024, 4, 4
        # global average pooling
        x = self.gap(x)
        # 1024, 1, 1
        # flatten
        x = x.view(x.size(0), -1) # N, 1024
        # fully connected layer
        x = self.fc(x)
        # N, Classes
        return x

