# The network is developed by Visual Geometry Group of Univ. of Oxford, so named 'VGGNet'
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms.transforms as transforms
from torchvision.datasets.folder import ImageFolder
from Nets1 import VGGNet

# 1. Load the data
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

testset = ImageFolder('C:/Users/Cooper/PycharmProjects/Imagenet', transform=transform)
testloader = DataLoader(testset, batch_size=10)

# 2. Define a network
net = VGGNet('C')

# 4. Test the network
for img, label in testloader:
    output = net(img)
    _, predicted = output.max(1)
print(predicted)
