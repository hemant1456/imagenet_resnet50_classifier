import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_channel, out_channel, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channel, out_channels=out_channel,kernel_size=kernel_size, padding=padding, stride = stride,bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU()
    )


class simple_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = conv_block(3,16, 3, 1, 1) # 32x32x16
        self.conv2 = conv_block(16,16, 3, 1, 1) # 32x32x16

        self.pool1 = nn.MaxPool2d(2,2) # 16x16x16

        self.conv3 = conv_block(16,32, 3, 1, 1) # 16x16x32
        self.conv4 = conv_block(32,32, 3, 1, 1) # 16x16x32

        self.pool2 = nn.MaxPool2d(2,2) # 8x8x32

        self.conv5 = conv_block(32,64, 3, 1, 1) # 8x8x64
        self.conv6 = conv_block(64,64, 3, 1, 1) # 8x8x64

        self.pool3 = nn.MaxPool2d(2,2) # 4x4x64

        self.conv7 = conv_block(64,128, 3, 1, 1) # 4x4x128
        self.conv8 = conv_block(128,128, 3, 1, 1) # 4x4x128

        self.gap = nn.AdaptiveAvgPool2d((1,1)) # 

        self.conv9 = nn.Conv2d(128, 10, 1,1,0,bias=False) # 1x1x10

    
    def forward(self,x):
        x = self.pool1(self.conv2(self.conv1(x)))
        x = self.pool2(self.conv4(self.conv3(x)))
        x = self.pool3(self.conv6(self.conv5(x)))
        x = self.conv8(self.conv7(x))
        x = self.conv9(self.gap(x))
        x = x.view((x.size(0),-1))
        return x
