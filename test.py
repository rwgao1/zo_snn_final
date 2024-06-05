import torch
import optimizee
import optimizee.cifar
import nn_optimizer
from train import train_update_rnn, train_model_with_optimizer, train_benchmark
import numpy as np
import matplotlib.pyplot as plt

import random

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available()  else "cpu"
    torch.manual_seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True


    BATCH_SIZE = 32
    NUM_EPOCH = 2

    optimizer_train_args = {
        "num_epoch": NUM_EPOCH,
        "updates_per_epoch": 10,
        "optimizer_steps": 200,
        "truncated_bptt_step": 20,
        "batch_size": BATCH_SIZE,
        # "optimizee": optimizee.cifar.CifarSpikingResNet18
        "optimizee": optimizee.mnist.MnistLinearModel
    }

    optimizee_train_args = {
        "num_epoch": NUM_EPOCH,
        "batch_size": BATCH_SIZE,
        # "optimizee": optimizee.cifar.CifarSpikingResNet18
        "optimizee": optimizee.mnist.SpikingMnistConvModel
    }

    benchmark_train_args = {
        "name": "adam",
        "num_epoch": NUM_EPOCH,
        "batch_size": BATCH_SIZE,
        # "optimizee": optimizee.cifar.CifarSpikingResNet18
        "optimizee": optimizee.mnist.SpikingMnistConvModel,
        "optimizer": torch.optim.Adam
    }


    # meta_optimizer, optim_loss_hist = train_update_rnn(optimizer_train_args, device)

    # np.save("optim_loss_hist.npy", optim_loss_hist)
    # meta_model = optimizer_train_args["optimizee"]()
    # meta_model.to('cuda')
    # meta_optimizer = nn_optimizer.zoopt.ZOOptimizer(optimizee.MetaModel(meta_model))
    # meta_optimizer.load_state_dict(torch.load("meta_optimizer.pt"))
    # loss_hist, test_loss_hist = train_model_with_optimizer(optimizee_train_args, meta_optimizer, device)
    # np.save("loss_hist.npy", loss_hist)
    # np.save("test_loss_hist.npy", test_loss_hist)

    meta_model = optimizee_train_args["optimizee"]()
    meta_model.to('cuda')
    meta_optimizer = nn_optimizer.zoopt.ZOOptimizer(optimizee.MetaModel(meta_model))
    meta_optimizer.load_state_dict(torch.load("meta_optimizer_MnistLinearModel.pt"))
    loss_hist, test_loss_hist = train_model_with_optimizer(optimizee_train_args, meta_optimizer, device)
    np.save("loss_hist.npy", loss_hist)

    # loss_hist = np.load("loss_hist.npy")
    benchmark_loss_hist, _ = np.load("benchmark_loss_hist.npy")
    # print(benchmark_loss_hist)

    # benchmark_loss_hist = train_benchmark(benchmark_train_args, optimizer, device)

    # np.save("benchmark_loss_hist.npy", benchmark_loss_hist)

    fig = plt.figure(facecolor="w", figsize=(20, 5))
    plt.plot(loss_hist, label="Train Loss")
    plt.plot(benchmark_loss_hist, label="Benchmark Loss")
    plt.legend()
    plt.show()