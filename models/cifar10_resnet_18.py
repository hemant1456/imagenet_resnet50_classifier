import torch.nn as nn
import torch.nn.functional as F


class basic_block1(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels*2,kernel_size=3, padding=1, stride = 2,bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels*2)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=in_channels*2, out_channels=in_channels*2,kernel_size=3, padding=1, stride = 1,bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels*2)
        self.relu2 = nn.ReLU()
        self.drop = nn.Dropout2d(0.1)

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels*2,kernel_size=1, padding=0, stride = 2,bias=False),
            nn.BatchNorm2d(in_channels*2))
    def forward(self,x):
        identity = self.downsample(x)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.drop(self.relu2(identity + self.bn2(self.conv2(x))))
        return x

class basic_block2(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,kernel_size=3, padding=1, stride = 1,bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,kernel_size=3, padding=1, stride = 1,bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu2 = nn.ReLU()
        self.drop = nn.Dropout2d(0.1)

    def forward(self,x):
        identity = x
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.drop(self.relu2(identity + self.bn2(self.conv2(x))))
        return x




class resnet_block(nn.Module):
    def __init__(self,in_channel):
        super().__init__()
        self.bb1 = basic_block1(in_channel)
        self.bb2 = basic_block2(in_channel*2)
    
    def forward(self,x):
        x = self.bb2(self.bb1(x))
        return x




class resnet18(nn.Module):
    def __init__(self,n_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3,32,3,1,1) # 32x32x32
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()

        #resblock1
        self.res_block1 = resnet_block(32)
        self.res_block2 = resnet_block(64)
        self.res_block3 = resnet_block(128)
        self.res_block4 = resnet_block(256)

        self.gap = nn.AdaptiveAvgPool2d((1,1))

        self.conv2 = nn.Conv2d(512,n_classes,1,1,0)

    def forward(self,x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.conv2(self.gap(x))
        x = x.view((x.size(0),-1))
        return x
