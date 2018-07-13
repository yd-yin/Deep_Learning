import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from Net2 import *
from Net1 import *

# 1. Load the data
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

testset = ImageFolder('C:/Users/Cooper/PycharmProjects/Imagenet', transform=transform)
testloader = DataLoader(testset, batch_size=10)

# 2. Define a net
net = ResNet50()

# 3. Test the net
for img, label in testloader:
    out = net(img)
    _, predicted = out.max(1)
    print(predicted)
