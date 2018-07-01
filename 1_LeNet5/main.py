from Nets import LeNet5
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# 1. Loading and Transforming the data
trainset = MNIST(root='./pytorch_data/mnist', download=False, 
                   transform=transforms.Compose([
                       transforms.Resize((32, 32)),
                       transforms.ToTensor()]))
testset = MNIST(root='./pytorch_data/mnist', train=False, download=False,
                  transform=transforms.Compose([
                      transforms.Resize((32,32)),
                      transforms.ToTensor()]))
trainloader = DataLoader(trainset, batch_size=256, shuffle=True)
testloader = DataLoader(testset, batch_size=1024, shuffle=False)


# 2. Define a Network
net = LeNet5()


# 3. Define a Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3)


# 4. Train the network
def train(epoch):
    train_loss = 0.0

    # get the data
    for i, (images, labels) in enumerate(trainloader):

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = net(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        train_loss += loss.item()
        if i % 20 == 0 and i != 0:
            print('Train - Epoch %d, Batch: %d, Losss: %f' % (epoch, i, train_loss / 20))
            train_loss = 0.0


# 5. Test the Network
def test():
    correct = 0
    for images, labels in testloader:
        output = net(images)
        predicted = output.max(1)[1]
        correct += (predicted == labels).sum().item()

    acc = correct / len(testset)
    print('Accuracy of Test Set: %f' % (acc))


def main():
    for epoch in range(1, 2):
        train(epoch)
        test()


if __name__ == '__main__':
    main()
