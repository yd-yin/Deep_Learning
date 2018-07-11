import torch.nn as nn
import torch.nn.functional as F

__all__ = ['VGGNetA']


class ConvReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        return x


class VGGNetA(nn.Module):
    def __init__(self):
        super(VGGNetA, self).__init__()
        self.op1 = nn.Sequential(
            ConvReLU(3, 64),
            nn.MaxPool2d(2, 2)
        )
        self.op2 = nn.Sequential(
            ConvReLU(64, 128),
            nn.MaxPool2d(2, 2)
        )
        self.op3 = nn.Sequential(
            ConvReLU(128, 256),
            ConvReLU(256, 256),
            nn.MaxPool2d(2, 2)
        )
        self.op4 = nn.Sequential(
            ConvReLU(256, 512),
            ConvReLU(512, 512),
            nn.MaxPool2d(2, 2)
        )
        self.op5 = nn.Sequential(
            ConvReLU(512, 512),
            ConvReLU(512, 512),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 1000),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.op1(x)
        x = self.op2(x)
        x = self.op3(x)
        x = self.op4(x)
        x = self.op5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
