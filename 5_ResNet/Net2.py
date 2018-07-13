import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['ResNet50']


class BottleNeck(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, typ):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, 1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU()
        )
        self.conv2_nosample = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU()
        )
        self.conv2_sample = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, 2, 1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1, 1)
        self.typ = typ

    def forward(self, x):
        if self.typ is 'in':
            # different dimension, no sampling
            y = self.conv1(x)
            y = self.conv2_nosample(y)
            y = self.conv3(y)
            x = self.shortcut(x)
            y += x
            return y
        elif self.typ is 'mid':
            # same dimension, no sampling
            y = self.conv1(x)
            y = self.conv2_nosample(y)
            y = self.conv3(y)
            y += x
            return y
        elif self.typ is 'out':
            # same dimension, sampling
            y = self.conv1(x)
            y = self.conv2_sample(y)
            y = self.conv3(y)
            x = F.max_pool2d(x, 2, 2)
            y += x
            return y
        else:
            raise Exception


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.op1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.MaxPool2d(3, 2, 1)
        )
        self.block56 = nn.Sequential(
            BottleNeck(64, 64, 256, 'in'),
            BottleNeck(256, 64, 256, 'mid'),
            BottleNeck(256, 64, 256, 'out')
        )
        self.block28 = nn.Sequential(
            BottleNeck(256, 128, 512, 'in'),
            BottleNeck(512, 128, 512, 'mid'),
            BottleNeck(512, 128, 512, 'mid'),
            BottleNeck(512, 128, 512, 'out')
        )
        self.block14 = nn.Sequential(
            BottleNeck(512, 256, 1024, 'in'),
            BottleNeck(1024, 256, 1024, 'mid'),
            BottleNeck(1024, 256, 1024, 'mid'),
            BottleNeck(1024, 256, 1024, 'mid'),
            BottleNeck(1024, 256, 1024, 'mid'),
            BottleNeck(1024, 256, 1024, 'out')
        )
        self.block7 = nn.Sequential(
            BottleNeck(1024, 512, 2048, 'in'),
            BottleNeck(2048, 512, 2048, 'mid'),
            BottleNeck(2048, 512, 2048, 'mid')
        )
        self.fc = nn.Sequential(
            nn.Linear(3 * 3 * 2048, 1000),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.op1(x)
        x = self.block56(x)
        x = self.block28(x)
        x = self.block14(x)
        x = self.block7(x)
        x = F.avg_pool2d(x, 3, 2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
