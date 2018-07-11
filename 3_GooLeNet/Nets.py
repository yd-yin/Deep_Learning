import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvReLu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvReLu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        return x


class Layer1and2(nn.Module):
    def __init__(self):
        super(Layer1and2, self).__init__()
        self.op1 = nn.Sequential(
            ConvReLu(3, 64, 7, 2, 3),
            nn.MaxPool2d(3, 2, 1),
            nn.BatchNorm2d(64)
        )
        self.op2 = ConvReLu(64, 64, 1, 1)
        self.op3 = nn.Sequential(
            ConvReLu(64, 192, 3, 1, 1),
            nn.BatchNorm2d(192),
            nn.MaxPool2d(3, 2, 1)
        )

    def forward(self, x):
        x = self.op1(x)
        x = self.op2(x)
        x = self.op3(x)
        return x


class Layer3a(nn.Module):
    def __init__(self):
        super(Layer3a, self).__init__()
        self.a = ConvReLu(192, 64, 1, 1)
        self.b = nn.Sequential(
            ConvReLu(192, 96, 1, 1),
            ConvReLu(96, 128, 3, 1, 1)
        )
        self.c = nn.Sequential(
            ConvReLu(192, 16, 1, 1),
            ConvReLu(16, 32, 5, 1, 2)
        )
        self.d = nn.Sequential(
            nn.MaxPool2d(3, 1),
            ConvReLu(192, 32, 1, 1, 1)
        )

    def forward(self, x):
        out1 = self.a(x)
        out2 = self.b(x)
        out3 = self.c(x)
        out4 = self.d(x)
        # out.size = batch_size, channels, size_len, size_width，要连接的是channel维度, dim=1
        out = torch.cat((out1, out2, out3, out4), dim=1)
        return out


class Layer3b(nn.Module):
    def __init__(self):
        super(Layer3b, self).__init__()
        self.a = ConvReLu(256, 128, 1, 1)
        self.b = nn.Sequential(
            ConvReLu(256, 128, 1, 1),
            ConvReLu(128, 192, 3, 1, 1)
        )
        self.c = nn.Sequential(
            ConvReLu(256, 32, 1, 1),
            ConvReLu(32, 96, 5, 1, 2)
        )
        self.d = nn.Sequential(
            nn.MaxPool2d(3, 1),
            ConvReLu(256, 64, 1, 1, 1)
        )

    def forward(self, x):
        out1 = self.a(x)
        out2 = self.b(x)
        out3 = self.c(x)
        out4 = self.d(x)
        out = torch.cat((out1, out2, out3, out4), dim=1)
        out = F.max_pool2d(out, 3, 2, 1)
        return out


class Layer4a(nn.Module):
    def __init__(self):
        super(Layer4a, self).__init__()
        self.a = ConvReLu(480, 192, 1, 1)
        self.b = nn.Sequential(
            ConvReLu(480, 96, 1, 1),
            ConvReLu(96, 208, 3, 1, 1)
        )
        self.c = nn.Sequential(
            ConvReLu(480, 16, 1, 1),
            ConvReLu(16, 48, 5, 1, 2)
        )
        self.d = nn.Sequential(
            nn.MaxPool2d(3, 1),
            ConvReLu(480, 64, 1, 1, 1)
        )

    def forward(self, x):
        out1 = self.a(x)
        out2 = self.b(x)
        out3 = self.c(x)
        out4 = self.d(x)
        out = torch.cat((out1, out2, out3, out4), dim=1)
        return out


class Layer4b(nn.Module):
    def __init__(self):
        super(Layer4b, self).__init__()
        self.a = ConvReLu(512, 160, 1, 1)
        self.b = nn.Sequential(
            ConvReLu(512, 112, 1, 1),
            ConvReLu(112, 224, 3, 1, 1)
        )
        self.c = nn.Sequential(
            ConvReLu(512, 24, 1, 1),
            ConvReLu(24, 64, 5, 1, 2)
        )
        self.d = nn.Sequential(
            nn.MaxPool2d(3, 1),
            ConvReLu(512, 64, 1, 1, 1)
        )
        self.e = nn.Sequential(
            nn.AvgPool2d(5, 3),
            ConvReLu(512, 128, 1, 1)
        )
        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.Dropout(0.7),
            nn.Linear(1024, 1000),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        out1 = self.a(x)
        out2 = self.b(x)
        out3 = self.c(x)
        out4 = self.d(x)
        out = torch.cat((out1, out2, out3, out4), dim=1)
        out_aux_temp = self.e(x)
        # 拉平，size = batch_size, nodes
        out_aux_temp = out_aux_temp.view(out_aux_temp.size(0), -1)
        out_aux = self.fc(out_aux_temp)
        return out, out_aux


class Layer4c(nn.Module):
    def __init__(self):
        super(Layer4c, self).__init__()
        self.a = ConvReLu(512, 128, 1, 1)
        self.b = nn.Sequential(
            ConvReLu(512, 128, 1, 1),
            ConvReLu(128, 256, 3, 1, 1)
        )
        self.c = nn.Sequential(
            ConvReLu(512, 24, 1, 1),
            ConvReLu(24, 64, 5, 1, 2)
        )
        self.d = nn.Sequential(
            nn.MaxPool2d(3, 1),
            ConvReLu(512, 64, 1, 1, 1)
        )

    def forward(self, x):
        out1 = self.a(x)
        out2 = self.b(x)
        out3 = self.c(x)
        out4 = self.d(x)
        out = torch.cat((out1, out2, out3, out4), dim=1)
        return out


class Layer4d(nn.Module):
    def __init__(self):
        super(Layer4d, self).__init__()
        self.a = ConvReLu(512, 112, 1, 1)
        self.b = nn.Sequential(
            ConvReLu(512, 144, 1, 1),
            ConvReLu(144, 288, 3, 1, 1)
        )
        self.c = nn.Sequential(
            ConvReLu(512, 32, 1, 1),
            ConvReLu(32, 64, 5, 1, 2)
        )
        self.d = nn.Sequential(
            nn.MaxPool2d(3, 1),
            ConvReLu(512, 64, 1, 1, 1)
        )

    def forward(self, x):
        out1 = self.a(x)
        out2 = self.b(x)
        out3 = self.c(x)
        out4 = self.d(x)
        out = torch.cat((out1, out2, out3, out4), dim=1)
        return out


class Layer4e(nn.Module):
    def __init__(self):
        super(Layer4e, self).__init__()
        self.a = ConvReLu(528, 256, 1, 1)
        self.b = nn.Sequential(
            ConvReLu(528, 160, 1, 1),
            ConvReLu(160, 320, 3, 1, 1)
        )
        self.c = nn.Sequential(
            ConvReLu(528, 32, 1, 1),
            ConvReLu(32, 128, 5, 1, 2)
        )
        self.d = nn.Sequential(
            nn.MaxPool2d(3, 1),
            ConvReLu(528, 128, 1, 1, 1)
        )
        self.e = nn.Sequential(
            nn.AvgPool2d(5, 3),
            ConvReLu(528, 128, 1, 1)
        )
        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.Dropout(0.7),
            nn.Linear(1024, 1000),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        out1 = self.a(x)
        out2 = self.b(x)
        out3 = self.c(x)
        out4 = self.d(x)
        out = torch.cat((out1, out2, out3, out4), dim=1)
        out = F.max_pool2d(out, 3, 2, 1)
        out_aux_temp = self.e(x)
        # 拉平，size = batch_size, nodes
        out_aux_temp = out_aux_temp.view(out_aux_temp.size(0), -1)
        out_aux = self.fc(out_aux_temp)
        return out, out_aux


class Layer5a(nn.Module):
    def __init__(self):
        super(Layer5a, self).__init__()
        self.a = ConvReLu(832, 256, 1, 1)
        self.b = nn.Sequential(
            ConvReLu(832, 160, 1, 1),
            ConvReLu(160, 320, 3, 1, 1)
        )
        self.c = nn.Sequential(
            ConvReLu(832, 32, 1, 1),
            ConvReLu(32, 128, 5, 1, 2)
        )
        self.d = nn.Sequential(
            nn.MaxPool2d(3, 1),
            ConvReLu(832, 128, 1, 1, 1)
        )

    def forward(self, x):
        out1 = self.a(x)
        out2 = self.b(x)
        out3 = self.c(x)
        out4 = self.d(x)
        out = torch.cat((out1, out2, out3, out4), dim=1)
        return out


class Layer5b(nn.Module):
    def __init__(self):
        super(Layer5b, self).__init__()
        self.a = ConvReLu(832, 384, 1, 1)
        self.b = nn.Sequential(
            ConvReLu(832, 192, 1, 1),
            ConvReLu(192, 384, 3, 1, 1)
        )
        self.c = nn.Sequential(
            ConvReLu(832, 48, 1, 1),
            ConvReLu(48, 128, 5, 1, 2)
        )
        self.d = nn.Sequential(
            nn.MaxPool2d(3, 1),
            ConvReLu(832, 128, 1, 1, 1)
        )
        self.e = nn.AvgPool2d(7, 1)
        self.fc = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.Dropout(0.4),
            nn.Linear(1024, 1000),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        out1 = self.a(x)
        out2 = self.b(x)
        out3 = self.c(x)
        out4 = self.d(x)
        out = torch.cat((out1, out2, out3, out4), dim=1)
        out = self.e(out)
        # 拉平，size = batch_size, nodes
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# class GooLeNet(nn.Module):
#     def __init__(self):
#         super(GooLeNet, self).__init__()
#         self.op1 = nn.Sequential(
#             Layer1and2(),
#             Layer3a(),
#             Layer3b(),
#             Layer4a(),
#             Layer4b()
#         )
#         self.op2 = nn.Sequential(
#             Layer4c(),
#             Layer4d(),
#             Layer4e()
#         )
#         self.op3 = nn.Sequential(
#             Layer5a(),
#             Layer5b()
#         )
#
#     def forward(self, x):
#         x, out_aux1 = self.op1(x)
#         x, out_aux2 = self.op2(x)
#         x = self.op3(x)
#         return x, out_aux1, out_aux2


# The way easy to debug
class GooLeNet(nn.Module):
    def __init__(self):
        super(GooLeNet, self).__init__()
        self.Layer1and2 = Layer1and2()
        self.Layer3a = Layer3a()
        self.Layer3b = Layer3b()
        self.Layer4a = Layer4a()
        self.Layer4b = Layer4b()
        self.Layer4c = Layer4c()
        self.Layer4d = Layer4d()
        self.Layer4e = Layer4e()
        self.Layer5a = Layer5a()
        self.Layer5b = Layer5b()

    def forward(self, x):
        x = self.Layer1and2(x)
        x = self.Layer3a(x)
        x = self.Layer3b(x)
        x = self.Layer4a(x)
        x, out_aux1 = self.Layer4b(x)
        x = self.Layer4c(x)
        x = self.Layer4d(x)
        x, out_aux2 = self.Layer4e(x)
        x = self.Layer5a(x)
        x = self.Layer5b(x)
        return x, out_aux1, out_aux2
