# 引包
```python
import torchvision.transforms as transforms
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader
```
### import 和from... import
import 模块：导入一个文件，是个相对路径。  
from…import：导入了文件中的某个类/函数，是个绝对路径，在本程序中可以直接使用，而不用在前面加点。  
import as 和import一样，只是给文件起了个新的名字  
from... import *：导入文件中所有的类  

# 准备数据
```python
data_train = MNIST(root='./pytorch_data/mnist', download=False,
                   transform=transforms.Compose([
                       transforms.Resize((32, 32)),
                       transforms.ToTensor()]
                   ))
data_test = MNIST(root='./pytorch_data/mnist', train=False, download=False,
                  transform=transforms.Compose([
                      transforms.Resize((32, 32)),
                      transforms.ToTensor()
                  ]))
```
### pytorch的图像预处理包
[pytorch中transform函数](https://www.jianshu.com/p/13e31d619c15)

# 迭代器
```python
data_train_loader = DataLoader(data_train, batch_size=256)
data_test_loader = DataLoader(data_test, batch_size=256)
```
[Dataloader官方文档](http://pytorch-cn.readthedocs.io/zh/latest/package_references/data/)
