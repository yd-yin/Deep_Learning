import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=(5, 5)),
            nn.Tanh(),
            # nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2,2), stride=2),
            # nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(6, 16, kernel_size=(5, 5)),
            nn.Tanh(),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.AvgPool2d(kernel_size=(2,2), stride=2),
            nn.Conv2d(16, 120, kernel_size=(5, 5)),
            # nn.ReLU()
            nn.Tanh()
        )
        self.fc = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, img):
        output = self.conv(img)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output
