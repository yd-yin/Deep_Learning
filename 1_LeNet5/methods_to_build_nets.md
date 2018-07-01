假设构建网络如下：  
卷积层 -> Relu层 -> 池化层 -> 全连接层 -> Relu层 -> 全连接层  
```python
import torch
import torch.nn.functional as F
from collections import OrderedDict
```

## Method 1
把卷积层和全连接层当作layer，池化和非线性函数当作操作放在forward里
```python
class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.fc1 = nn.Linear(32 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        output = F.max_pool2d(F.relu(self.conv(x)), 2)  # x通过卷积层，relu操作，池化操作给output
        output = output.view(output.size(0), -1)  # output改变大小准备进入全连接层
        output = F.relu(self.fc1(output)) # 进入全连接层，relu操作，更新output
        output = self.fc2(output) # 进入全连接层
        return output
```

### x = x.view(x.size(0),-1)
通过卷积层以后x有四维大小：batchsize, channels, x, y  
本语句将四维整成两维，第一维是x.size(0), 即batchsize；第二维不指定，由程序计算  
通过这样的处理使得数据能进入全连接层

## Method 2
通过Sequantial将各层顺序添加到容器中，缺点是每层的编号是默认的阿拉伯数字，不易区分。  
把relu和池化都当作层来处理
```python
class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Sequential(
            nn.Linear(32 * 3 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        output = self.conv1(x)  # 通过卷积层、relu层、池化层
        output = output.view(output.size(0), -1)  # 改变大小准备进入全连接层
        output = self.fc(output)  # 进入全连接层、relu层、全连接层
        return output
```

## Method 3
通过字典的形式添加每一层，并且设置单独的层名称。  
OrderedDict有序字典
```python
class Net4(nn.Module):
    def __init__(self):
        super(Net4, self).__init__()
        self.conv = nn.Sequential(
            OrderedDict([
                    ("conv1", nn.Conv2d(3, 32, 3, 1, 1)),
                    ("relu1", nn.ReLU()),
                    ("pool", nn.MaxPool2d(2))
            ])
        )

        self.fc = nn.Sequential(
            OrderedDict([
                ("fc1", nn.Linear(32 * 3 * 3, 128)),
                ("relu2", nn.ReLU()),
                ("fc2", nn.Linear(128, 10))
            ])
        )

    def forward(self, x):
        output = self.conv1(x)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output
```

# 疑惑
把relu和池化当作操作时用的是`torch.nn.Functional.relu/max_pool2d`  
把他们当作层加入sequential时用的是`torch.nn.relu/max_pool2d`  
