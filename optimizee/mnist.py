import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import SubsetRandomSampler, Subset, DataLoader

import snntorch as snn
import snntorch.functional as SF

import optimizee

import os
import numpy as np
import random
import time

def local_zo(delta=0.05, q=1):
    
    def inner(input_):
        return LocalZO.apply(input_, delta, q)
    return inner


class LocalZO(torch.autograd.Function):
    total_forward = 0
    @staticmethod
    def forward(ctx, input_, delta=0.05, q=1):
        # print("LocalZO forward")
        grad = torch.zeros_like(input_)
        for _ in range(q):
            z = torch.randn_like(input_)
            grad += (torch.abs(input_) < delta * torch.abs(z)) * torch.abs(z) / (2 * delta)
        torch.div(grad, q)

        ctx.save_for_backward(grad)
        out = (input_ > 0).float()
        
        LocalZO.total_forward += 1

        return out
    

    @staticmethod
    def backward(ctx, grad_output):
        # print("LocalZO backward")
        (grad,) = ctx.saved_tensors
        grad_input = grad_output.clone()

        return grad * grad_input, None, None


# class MnistModel(optimizee.Optimizee):
#     def __init__(self):
#         super(MnistModel, self).__init__()

#     @staticmethod
#     def dataset_loader(data_dir, batch_size, test_batch_size):
#         train_loader = torch.utils.data.DataLoader(
#             datasets.MNIST(data_dir, train=True, download=True,
#                            transform=transforms.Compose([
#                                transforms.ToTensor(),
#                                transforms.Normalize((0.1307,), (0.3081,))
#                            ])),
#             batch_size=batch_size, shuffle=True)

#         test_loader = torch.utils.data.DataLoader(
#             datasets.MNIST(data_dir, train=False, transform=transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.1307,), (0.3081,))
#             ])),
#             batch_size=test_batch_size, shuffle=False)

#         return train_loader, test_loader

#     def loss(self, fx, tgt):
#         loss = F.nll_loss(fx, tgt)
#         return loss


# class MnistLinearModel(MnistModel):
#     '''A MLP on dataset MNIST.'''

#     def __init__(self):
#         super(MnistLinearModel, self).__init__()
#         self.linear1 = nn.Linear(28 * 28, 32)
#         self.linear2 = nn.Linear(32, 10)

#     def forward(self, inputs):
#         x = inputs.view(-1, 28 * 28)
#         x = F.relu(self.linear1(x))
#         x = self.linear2(x)
#         return F.log_softmax(x)


# class MnistConvModel(MnistModel):
#     def __init__(self):
#         super(MnistConvModel, self).__init__()
#         self.conv1 = nn.Conv2d(1, 20, 5, 1)
#         self.conv2 = nn.Conv2d(20, 50, 5, 1)
#         self.fc1 = nn.Linear(4 * 4 * 50, 500)
#         self.fc2 = nn.Linear(500, 10)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = x.view(-1, 4 * 4 * 50)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)

class SpikingMnistModel(optimizee.Optimizee):
    def __init__(self):
        super(SpikingMnistModel, self).__init__()
        self.loss = SF.ce_rate_loss()


    @staticmethod
    def dataset_loader(data_dir='/tmp/data/mnist', batch_size=128, test_batch_size=128):
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])
        
        mnist_train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        mnist_test = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

        train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=False, drop_last=True)
        test_loader = DataLoader(mnist_test, batch_size=test_batch_size, shuffle=False, drop_last=True)

        return train_loader, test_loader

    def loss(self, spk_rec, tgt):
        loss = self.loss(spk_rec, tgt)
        return loss
    

class MnistLinearModel20(SpikingMnistModel):
    '''A MLP on dataset MNIST.'''

    def __init__(self):
        super(MnistLinearModel20, self).__init__()
        self.spike_grad = local_zo(delta=0.6, q=1)

        self.linear1 = nn.Linear(28 * 28, 20)
        self.linear2 = nn.Linear(20, 10)

        self.lif = [snn.Leaky(beta=0.95, spike_grad=self.spike_grad, init_hidden=True).to('cuda') for _ in range(2)]


    def forward(self, inputs):
        for lif in self.lif:
            lif.init_leaky()
        spk_rec = []

        for i in range(25):
            out = inputs.view(-1, 28 * 28)
            out = F.relu(self.linear1(out))
            out = self.lif[0](out)
            out = self.linear2(out)
            out = self.lif[1](out)
            spk_rec.append(out)
        
        print(LocalZO.total_forward)
        return torch.stack(spk_rec)


class MnistLinearModel40(SpikingMnistModel):
    '''A MLP on dataset MNIST.'''

    def __init__(self):
        super(MnistLinearModel40, self).__init__()
        spike_grad = local_zo(delta=0.6, q=1)

        self.linear1 = nn.Linear(28 * 28, 40)
        self.linear2 = nn.Linear(40, 10)

        self.lif = [snn.Leaky(beta=0.95, spike_grad=spike_grad, init_hidden=True).to('cuda') for _ in range(2)]


    def forward(self, inputs):
        for lif in self.lif:
            lif.init_leaky()
        spk_rec = []

        for i in range(25):
            out = inputs.view(-1, 28 * 28)
            out = F.relu(self.linear1(out))
            out = self.lif[0](out)
            out = self.linear2(out)
            out = self.lif[1](out)
            spk_rec.append(out)
        
        return torch.stack(spk_rec)
    

class MnistLinearModel2(SpikingMnistModel):
    '''A MLP on dataset MNIST.'''

    def __init__(self):
        super(MnistLinearModel2, self).__init__()
        spike_grad = local_zo(delta=0.6, q=1)

        self.linear1 = nn.Linear(28 * 28, 20)
        self.linear2 = nn.Linear(20, 20)
        self.linear3 = nn.Linear(20, 10)

        self.lif = [snn.Leaky(beta=0.95, spike_grad=spike_grad, init_hidden=True).to('cuda') for _ in range(3)]


    def forward(self, inputs):
        for lif in self.lif:
            lif.init_leaky()
        spk_rec = []

        for i in range(25):
            out = inputs.view(-1, 28 * 28)
            out = F.relu(self.linear1(out))
            out = self.lif[0](out)
            out = F.relu(self.linear2(out))
            out = self.lif[1](out)
            out = self.linear3(out)
            out = self.lif[2](out)
            spk_rec.append(out)
        # print(LocalZO.total_forward)

        return torch.stack(spk_rec)



class SpikingMnistConvModel(SpikingMnistModel):
    def __init__(self):
        super(SpikingMnistConvModel, self).__init__()
        spike_grad = local_zo(delta=0.6, q=1)

        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)
        self.lif = [snn.Leaky(beta=0.95, spike_grad=spike_grad, init_hidden=True).to('cuda') for _ in range(4)]


    def forward(self, x):
        for lif in self.lif:
            lif.init_leaky()
        spk_rec = []
        for i in range(25):
            out = F.relu(self.conv1(x))
            out = F.max_pool2d(out, 2, 2)
            out = self.lif[0](out)
            out = F.relu(self.conv2(out))
            out = F.max_pool2d(out, 2, 2)
            out = self.lif[1](out)
            out = out.view(-1, 4 * 4 * 50)
            out = F.relu(self.fc1(out))
            out = self.lif[2](out)
            out = self.fc2(out)
            out = self.lif[3](out)
            spk_rec.append(out)
        # print(LocalZO.total_forward)

        return torch.stack(spk_rec)

    

# class MnistSpikingConvModel(SpikingMnistModel):

#     # Network Architecture
#     num_inputs = 28*28
#     num_hidden = 1000
#     num_outputs = 10

#     # Temporal Dynamics
#     num_steps = 25
#     beta = 0.95

#     def __init__(self):
#         super(MnistSpikingConvModel, self).__init__()
#         spike_grad = local_zo(delta=0.6, q=1)
        
#         # Initialize layers
#         self.fc1 = nn.Linear(MnistSpikingConvModel.num_inputs, MnistSpikingConvModel.num_hidden)
#         self.lif1 = snn.Leaky(beta=MnistSpikingConvModel.beta, spike_grad=spike_grad)

#         self.fc2 = nn.Linear(MnistSpikingConvModel.num_hidden, MnistSpikingConvModel.num_outputs)
#         self.lif2 = snn.Leaky(beta=MnistSpikingConvModel.beta, spike_grad=spike_grad)


#     def forward(self, x):
#         x = x.view(16, -1)
#         mem1 = self.lif1.init_leaky()
#         mem2 = self.lif2.init_leaky()

#         spk2_rec = []
#         mem2_rec = []

#         for _ in range(MnistSpikingConvModel.num_steps):
#             cur1 = self.fc1(x)
#             spk1, mem1 = self.lif1(cur1, mem1)
#             cur2 = self.fc2(spk1)
#             spk2, mem2 = self.lif2(cur2, mem2)
#             spk2_rec.append(spk2)
#             mem2_rec.append(mem2)

#         # return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)
#         return torch.stack(spk2_rec)


# class MnistAttack(optimizee.Optimizee):
#     def __init__(self, attack_model, batch_size=1, channel=1, width=28, height=28, c=0.1, gap=0.0,
#                  loss_type="l1", initial_noise=True):
#         super(MnistAttack, self).__init__()

#         if not initial_noise:
#             self.weight = torch.nn.Parameter(torch.zeros((batch_size, channel, width, height)))
#         else:
#             torch.random.manual_seed(1234)
#             self.weight = torch.nn.Parameter(1e-4 * torch.normal(torch.zeros((batch_size, channel, width, height)),
#                                                                  torch.ones((batch_size, channel, width, height))))
#             torch.random.manual_seed(time.time())
#         self.c = c  # regularization parameter c trades off adversarial success and L2 distortion
#         self.gap = gap  # confidence parameter that guarantees a constant gap

#         self.attack_model = attack_model

#         self.bs = batch_size

#         self.loss_type = loss_type

#     @staticmethod
#     def dataset_loader(data_dir, batch_size, test_batch_size, train_num=100, test_num=100):
#         path = os.path.join(data_dir, "mnist_correct/label_correct_index.npy")
#         label_correct_indices = list(np.load(path))
#         random.seed(1234)
#         random.shuffle(label_correct_indices)
#         train_indices = label_correct_indices[:train_num]
#         test_indices = label_correct_indices[5000:5000 + test_num]

#         train_loader = torch.utils.data.DataLoader(
#             datasets.MNIST(data_dir, train=False, download=True,
#                            transform=transforms.Compose([
#                                transforms.ToTensor(),
#                                transforms.Normalize((0.1307,), (0.3081,))
#                            ])),
#             batch_size=batch_size, shuffle=False, sampler=SubsetRandomSampler(train_indices), drop_last=True)

#         test_loader = torch.utils.data.DataLoader(
#             Subset(datasets.MNIST(data_dir, train=False, transform=transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.1307,), (0.3081,))
#             ])), test_indices),
#             batch_size=test_batch_size, shuffle=False)

#         return train_loader, test_loader

#     @staticmethod
#     def data_denormalize(data):
#         return data * 0.3081 + 0.1307

#     def forward(self, x):
#         x_ = x
#         x = x * 0.3081 + 0.1307  # [0, 1]

#         perturb = self.weight

#         fx = x + perturb
#         fx.clamp_(0.0, 1.0)

#         fx = (fx - 0.1307) / 0.3081
#         return fx, x_

#     def loss(self, fx, tgt, return_tuple=False):
#         assert isinstance(fx, tuple)
#         x_attack, x = fx

#         if self.loss_type == "l1":
#             loss_distort = ((self.data_denormalize(x_attack) - self.data_denormalize(x)).abs()).sum()
#         elif self.loss_type == "l2":
#             loss_distort = ((self.data_denormalize(x_attack) - self.data_denormalize(x)) ** 2).sum()

#         pred_scores = self.attack_model.model(x_attack)  # log likelyhood: (B, 10)
#         tgt_onehot = F.one_hot(tgt, num_classes=10).double()  # (B, 10)

#         correct_log_prob, correct_indices = torch.max(pred_scores - 1e9 * (1 - tgt_onehot), dim=1)
#         assert torch.equal(correct_indices, tgt), (
#             correct_indices, tgt, pred_scores, x_attack, x, self.attack_model.model.state_dict())
#         max_wrong_log_prob, max_wrong_indices = torch.max(pred_scores - 1e9 * tgt_onehot, dim=1)
#         loss_attack = torch.max(correct_log_prob - max_wrong_log_prob, max_wrong_log_prob.new_ones(()) * -self.gap)
#         loss_attack = loss_attack.mean()

#         if not return_tuple:
#             return (loss_attack + self.c * loss_distort) * self.bs
#         else:
#             return (loss_attack, self.c * loss_distort)

#     def nondiff_loss(self, weight, x, tgt, batch_weight=False):
#         if not batch_weight:
#             x_ = x
#             x = x * 0.3081 + 0.1307  # [0, 1]

#             fx = x + weight
#             fx.clamp_(0.0, 1.0)

#             fx = (fx - 0.1307) / 0.3081

#             x_attack, x = fx, x_

#             if self.loss_type == "l1":
#                 loss_distort = ((self.data_denormalize(x_attack) - self.data_denormalize(x)).abs()).sum()
#             elif self.loss_type == "l2":
#                 loss_distort = ((self.data_denormalize(x_attack) - self.data_denormalize(x)) ** 2).sum()

#             pred_scores = self.attack_model.model(x_attack)  # log likelyhood: (B, 10)
#             tgt_onehot = F.one_hot(tgt, num_classes=10).double()  # (B, 10)

#             correct_log_prob, correct_indices = torch.max(pred_scores - 1e9 * (1 - tgt_onehot), dim=1)
#             assert torch.equal(correct_indices, tgt), (
#                 correct_indices, tgt, pred_scores, x_attack, x, self.attack_model.model.state_dict())
#             max_wrong_log_prob, max_wrong_indices = torch.max(pred_scores - 1e9 * tgt_onehot, dim=1)
#             loss_attack = torch.max(correct_log_prob - max_wrong_log_prob, max_wrong_log_prob.new_ones(()) * -self.gap)
#             loss_attack = loss_attack.mean()

#             return (loss_attack + self.c * loss_distort) * self.bs
#         else:
#             x_ = x.unsqueeze(0)  # (1, B, *)
#             x = x * 0.3081 + 0.1307  # [0, 1]

#             fx = x + weight  # (B_weight, B, *)
#             fx.clamp_(0.0, 1.0)

#             fx = (fx - 0.1307) / 0.3081

#             x_attack, x = fx, x_
#             x_attack_shape = x_attack.size()

#             if self.loss_type == "l1":
#                 loss_distort = ((self.data_denormalize(x_attack) - self.data_denormalize(x)).abs()).sum(
#                     dim=[1, 2, 3, 4])
#             elif self.loss_type == "l2":
#                 loss_distort = ((self.data_denormalize(x_attack) - self.data_denormalize(x)) ** 2).sum(dim=[1, 2, 3, 4])

#             pred_scores = self.attack_model.model(
#                 x_attack.view(-1, x_attack_shape[2], x_attack_shape[3], x_attack_shape[4])).view(x_attack_shape[0],
#                                                                                                  x_attack_shape[1],
#                                                                                                  10)  # log likelyhood: (B_weight, B, 10)
#             tgt_onehot = F.one_hot(tgt, num_classes=10).double().unsqueeze(1)  # (1, B, 10)

#             correct_log_prob, correct_indices = torch.max(pred_scores - 1e9 * (1 - tgt_onehot), dim=2)  # (B_weight, B)
#             assert torch.equal(correct_indices, tgt.expand_as(correct_indices)), (
#                 correct_indices, tgt, pred_scores, x_attack, x, self.attack_model.model.state_dict())
#             max_wrong_log_prob, max_wrong_indices = torch.max(pred_scores - 1e9 * tgt_onehot, dim=2)  # (B_weight, B)
#             loss_attack = torch.max(correct_log_prob - max_wrong_log_prob, max_wrong_log_prob.new_ones(()) * -self.gap)
#             loss_attack = loss_attack.mean(dim=1)

#             return (loss_attack + self.c * loss_distort) * self.bs
