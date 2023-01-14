from PRASAD import DataLoader_train, torch_train, torch_eval
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST

train_MNIST = MNIST('./data/MNIST', train=True, download=True, transform=transforms.ToTensor())
test_MNIST = MNIST('./data/MNIST', train=False, download=True, transform=transforms.ToTensor())

layers_MNIST = list()
layers_MNIST.append(nn.Linear(784, 256))
layers_MNIST.append(nn.ReLU())
layers_MNIST.append(nn.Linear(256, 10))
net_MNIST = nn.Sequential(*layers_MNIST)

criterion_MNIST = nn.CrossEntropyLoss(); lr = 0.001

net_MNIST = torch_train(net_MNIST, criterion_MNIST, lr, train_MNIST, number_of_batches=50000)
torch_eval(net_MNIST, test_MNIST)
