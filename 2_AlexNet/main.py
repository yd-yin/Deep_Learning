import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import Nets

# 1.Load the data
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


testset = ImageFolder('C:/Users/Cooper/PycharmProjects/Imagenet', transform=transform)
testloader = DataLoader(testset, batch_size=10, shuffle=True)

# 2. Define a Network
net = Nets.AlexNet1()

# 3. Use the pre-trained parameters
# pretrained_dict = alexnet(pretrained=True).state_dict()
# net_dict = net.state_dict()
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict}
# net_dict.update(pretrained_dict)
# net.load_state_dict(net_dict)

# The network is randomly initialized now

# 4. Test the network
doc = open('./predict.txt','w')
pred_out = []
for img, label in testloader:
    output = net(img)
    _, predicted = output.max(1)
    predict_list = predicted.tolist()
    pred_out.extend(predict_list)
print(predict_list, file=doc)
