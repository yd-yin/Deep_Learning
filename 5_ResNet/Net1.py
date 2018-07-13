import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['ResNet34']


class BasicBlock(nn.Module):
    # building BasicBlock when the input and output are of the same dimensions
    def __init__(self, channels):
        super(BasicBlock, self).__init__()
        self.keepsize = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

    def forward(self, x):
        y = self.keepsize(x)
        y = self.keepsize(y)
        y += x
        return y


class BasicBlock_d(nn.Module):
    """
    building BasicBlock when the input and output are of different dimensions
    type A: Performs identity mapping, with extra zero entries padded for increasing dimensions
    type B: The projection shortcut is used to match dimensions
    """
    def __init__(self, in_channels, out_channels, typ=None):
        super(BasicBlock_d, self).__init__()
        self.sampling = nn.Sequential(
            # sampling
            nn.Conv2d(in_channels, out_channels, 3, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.keepsize = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.shortcut = nn.Conv2d(in_channels, out_channels, 1, 1)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.typ = typ

    def forward(self, x):
        y = self.sampling(x)
        y = self.keepsize(y)
        x = F.max_pool2d(x, 2, 2)   # downsample
        if self.typ is 'A':
            x = torch.cat((x, torch.zeros(x.size(0), self.out_channels - self.in_channels, x.size(2), x.size(3))), dim=1)
        elif self.typ is 'B':
            x = self.shortcut(x)
        else:
            raise Exception
        y += x
        return y


class ResNet34(nn.Module):
    def __init__(self, typ):
        super(ResNet34, self).__init__()
        self.op1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.MaxPool2d(3, 2, 1)
        )
        self.block64 = nn.Sequential(
            BasicBlock(64),
            BasicBlock(64),
            BasicBlock(64)
        )
        self.block128 = nn.Sequential(
            BasicBlock_d(64, 128, typ),
            BasicBlock(128),
            BasicBlock(128),
            BasicBlock(128)
        )
        self.block256 = nn.Sequential(
            BasicBlock_d(128, 256, typ),
            BasicBlock(256),
            BasicBlock(256),
            BasicBlock(256),
            BasicBlock(256),
            BasicBlock(256)
        )
        self.block512 = nn.Sequential(
            BasicBlock_d(256, 512, typ),
            BasicBlock(512),
            BasicBlock(512),
        )
        self.fc = nn.Sequential(
            nn.Linear(3 * 3 * 512, 1000),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.op1(x)
        x = self.block64(x)
        x = self.block128(x)
        x = self.block256(x)
        x = self.block512(x)
        x = F.avg_pool2d(x, 3, 2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
