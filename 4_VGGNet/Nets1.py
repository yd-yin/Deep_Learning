# Inspired by the source code of torchvision.models.vgg
# VGGNet is regular in arguments of Conv2d/Maxpooling and the output size,
# which can be used to establish the nets conveniently

import torch.nn as nn

__all__ = ['VGGNet']

# 'M' stands for Maxpooling
# 'S' stands for kernel_size = 1 (special case)
dic = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'A_bn': [64, 'N', 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'C': [64, 64, 'M', 128, 128, 'M', 256, 256, 'S', 256, 'M', 512, 512, 'S', 512, 'M', 512, 512, 'S', 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


def make_layer(typ):
    layer = []
    in_channels = 3
    single = False
    for i in dic[typ]:
        if i == 'S':
            single = True
        elif i == 'M':
            layer += [nn.MaxPool2d(2, 2)]
        elif i == 'N':
            layer += [nn.BatchNorm2d(in_channels)]
        else:
            if single:
                layer += [nn.Conv2d(in_channels, i, 1, 1, 0), nn.ReLU()]
                single = False
            else:
                layer += [nn.Conv2d(in_channels, i, 3, 1, 1), nn.ReLU()]
            in_channels = i
    return nn.Sequential(*layer)


class VGGNet(nn.Module):
    def __init__(self, typ):
        super(VGGNet, self).__init__()
        self.op1 = make_layer(typ)
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
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
