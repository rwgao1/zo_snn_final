import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import SubsetRandomSampler, Subset, DataLoader

import snntorch as snn
from snntorch import surrogate
import snntorch.functional as SF

import optimizee

import os
import numpy as np
import random
import time

# Wrapper for LocalZO
def local_zo(delta=0.05, q=1):
    delta = delta
    q = q
    def inner(input_):
        return LocalZO.apply(input_, delta, q)
    return inner


class LocalZO(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, delta=0.05, q=1):
        grad = torch.zeros_like(input_)
        for _ in range(q):
            z = torch.randn_like(input_)
            grad += (torch.abs(input_) < delta * torch.abs(z)) * torch.abs(z) / (2 * delta)
        torch.div(grad, q)

        ctx.save_for_backward(grad)
        out = (input_ > 0)
        
        return out.float()
    

    @staticmethod
    def backward(ctx, grad_output):
        (grad,) = ctx.saved_tensors
        grad_input = grad_output.clone()

        return grad * grad_input, None, None


class CifarModel(optimizee.Optimizee):
    def __init__(self):
        super(CifarModel, self).__init__()
        self.loss = SF.ce_rate_loss()

    
    @staticmethod
    def dataset_loader(data_dir='./data', batch_size=128, test_batch_size=128):
        cifar_mean = (0.4914, 0.4822, 0.4465)
        cifar_std = (0.2470, 0.2435, 0.2616)

        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(cifar_mean, cifar_std)])
        
        cifar_train = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
        cifar_test = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)

        train_loader = DataLoader(cifar_train, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(cifar_test, batch_size=test_batch_size, shuffle=True, drop_last=True)

        return train_loader, test_loader


    def loss(self, fx, tgt):
        loss = self.loss(fx, tgt)
        return loss
    

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    

class CifarSpikingResNet18(CifarModel):
    
    # Temporal Dynamics
    num_steps = 25
    beta = 0.95

    def __init__(self, block=BasicBlock, num_blocks=[1,1,1,1], num_classes=10):
        super(CifarSpikingResNet18, self).__init__()
        spike_grad = local_zo(delta=0.6, q=1)
        # spike_grad = surrogate.fast_sigmoid()
        

        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        # self.lif = [snn.Leaky(beta=CifarSpikingResNet18.beta, spike_grad=spike_grad, init_hidden=True).to('cuda') for _ in range(6)]
        self.lif = snn.Leaky(beta=CifarSpikingResNet18.beta, spike_grad=spike_grad, init_hidden=True).to('cuda')

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    def forward(self, x):
        self.lif.init_leaky()
        spk_rec = []

        for _ in range(CifarSpikingResNet18.num_steps):
            out = F.relu(self.bn1(self.conv1(x)))
            # out = self.lif[0](out)

            out = self.layer1(out)
            # out = self.lif[1](out)

            out = self.layer2(out)
            # out = self.lif[2](out)

            out = self.layer3(out)
            # out = self.lif[3](out)

            out = self.layer4(out)
            # out = self.lif[4](out)

            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            # out = self.lif[5](out)
            out = self.lif(out)

            spk_rec.append(out)

        return torch.stack(spk_rec)

    # def __init__(self, block=BasicBlock, num_blocks=[1,1,1,1], num_classes=10):
    #     super(CifarSpikingResNet18, self).__init__()
    #     self.in_planes = 64

    #     self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
    #                            stride=1, padding=1, bias=False)
    #     self.bn1 = nn.BatchNorm2d(64)
    #     self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
    #     self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
    #     self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
    #     self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
    #     self.linear = nn.Linear(512*block.expansion, num_classes)
    #     self.flatten = nn.Flatten()
    #     self.relu = nn.ReLU()
    #     self.lif = [snn.Synaptic(alpha=0.9, beta=0.5, threshold=1,
    #                             reset_mechanism='subtract', init_hidden=True).to('cuda') for _ in range(6)]

    # def _make_layer(self, block, planes, num_blocks, stride):
    #     strides = [stride] + [1]*(num_blocks-1)
    #     layers = []
    #     for stride in strides:
    #         layers.append(block(self.in_planes, planes, stride))
    #         self.in_planes = planes * block.expansion
    #     return nn.Sequential(*layers)

    # def forward(self, x):
    #     spk_rec = []
    #     for i in range(CifarSpikingResNet18.num_steps):
    #         out = self.relu(self.bn1(self.conv1(x)))
    #         out = self.lif[0](out)
    #         out = self.layer1(out)
    #         out = self.lif[1](out)
    #         out = self.layer2(out)
    #         out = self.lif[2](out)
    #         out = self.layer3(out)
    #         out = self.lif[3](out)
    #         out = self.layer4(out)
    #         out = self.lif[4](out)
    #         out = F.avg_pool2d(out, 4)
    #         out = self.flatten(out)
    #         out = self.linear(out)
    #         out = self.lif[5](out)
    #         spk_rec.append(out)

    #     return torch.stack(spk_rec)