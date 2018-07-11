import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from Nets import GooLeNet

# 1. Load the data
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
trainset = ImageFolder('C:/Users/Cooper/PycharmProjects/Imagenet', transform=transform)
trainloader = DataLoader(trainset, batch_size=10)

# 2. Define a network
net = GooLeNet()

# 3. Train the network
# The network is randomly initialized now

# 4. Test the network
for img, label in trainloader:
    out, out_aux1, out_aux2 = net(img)
    final_out = out + 0.3 * out_aux1 + 0.3 * out_aux2
    _, predicted = final_out.max(1)
    print(predicted)
