import torch.nn as nn
import torch.nn.functional as F

__all__ = ['AlexNet', 'AlexNet1']

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            # part 1
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(96),
            # part 2
            nn.Conv2d(96,256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            # part 3
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            # part 4
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            # part 5
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.fc = nn.Sequential(
            # part 6
            nn.Linear(6 * 6 * 256, 4096),
            nn.ReLU(),
            nn.Dropout(),
            # part 7
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            # part 8
            nn.Linear(4096, 1000),
            nn.Softmax(dim=1),
        )

    def forward(self, input):
        output = self.conv(input)
        output = output.view(input.size(0), -1)
        output = self.fc(output)
        return output


# 下面这种写法更容易调试跟踪每个输出的size，容易debug
class AlexNet1(nn.Module):
    def __init__(self):
        super(AlexNet1, self).__init__()
        self.op1 = nn.Sequential(
            ConvReLU(3, 96, kernel_size=11, stride=4, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(96)
        )
        self.op2 = nn.Sequential(
            ConvReLU(96, 256, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(256)
        )
        self.op3 = ConvReLU(256, 384, kernel_size=3, padding=1)
        self.op4 = ConvReLU(384, 384, kernel_size=3, padding=1)
        self.op5 = nn.Sequential(
            ConvReLU(384, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.fc = nn.Sequential(
            # part 1
            nn.Linear(6 * 6 * 256, 4096),
            nn.ReLU(),
            nn.Dropout(),
            # part 2
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            # part 3
            nn.Linear(4096, 1000),
            nn.Softmax(dim=1),
        )

    def forward(self, input):
        output = self.op1(input)
        output = self.op2(output)
        output = self.op3(output)
        output = self.op4(output)
        output = self.op5(output)
        output = output.view(input.size(0), -1)
        output = self.fc(output)
        return output


class ConvReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        return x
