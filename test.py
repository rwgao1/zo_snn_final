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

    OPTIMIZER_NUM_EPOCH = 10
    
    BATCH_SIZE = 32
    NUM_EPOCH = 1

    optimizer_train_args = {
        "num_epoch": OPTIMIZER_NUM_EPOCH,
        "updates_per_epoch": 10,
        "optimizer_steps": 200,
        "truncated_bptt_step": 20,
        "batch_size": 32,
        "finite_diff": False,
        "validation": False,
        # "optimizee": optimizee.cifar.CifarSpikingResNet18
        "optimizee": optimizee.mnist.MnistLinearModel2
    }

    optimizee_train_args = {
        "num_epoch": NUM_EPOCH,
        "batch_size": BATCH_SIZE,
        "trials": 10,
        # "optimizee": optimizee.cifar.MnistLinearModel20
        "optimizee": optimizee.mnist.SpikingMnistConvModel
    }

    benchmark_train_args = {
        "name": "adam",
        "num_epoch": NUM_EPOCH,
        "batch_size": BATCH_SIZE,
        "trials": 10,
        # "optimizee": optimizee.cifar.CifarSpikingResNet18
        "optimizee": optimizee.mnist.MnistLinearModel40,
        "optimizer": torch.optim.Adagrad
    }


    # meta_optimizer, optim_loss_hist, optimizee_loss_path = train_update_rnn(optimizer_train_args, device)
    # if optimizer_train_args["validation"]:
    #     np.save(f"optimizer_loss_V{optimizer_train_args['optimizee'].__name__}", optim_loss_hist)
    #     np.save(f"optimizee_loss_path_V{optimizer_train_args['optimizee'].__name__}.npy", optimizee_loss_path)
    # else:
    #     np.save(f"optimizer_loss_{optimizer_train_args['optimizee'].__name__}", optim_loss_hist)
    #     np.save(f"optimizee_loss_path_{optimizer_train_args['optimizee'].__name__}.npy", optimizee_loss_path)

    # torch.save(meta_optimizer.state_dict(), f"meta_optimizer_{optimizer_train_args['optimizee'].__name__}.pt")


    # meta_model = optimizer_train_args["optimizee"]()
    # meta_model.to('cuda')
    # meta_optimizer = nn_optimizer.zoopt.ZOOptimizer(optimizee.MetaModel(meta_model))

    # meta_optimizer.load_state_dict(torch.load("meta_optimizer_MnistLinearModel20.pt"))
    # optimizee_meta_model = optimizee.MetaModel(optimizee_train_args["optimizee"]().to(device))
    # meta_optimizer.meta_model = optimizee_meta_model

    # loss_hist = train_model_with_optimizer(optimizee_train_args, meta_optimizer, device)
    # np.save("loss_hist.npy", loss_hist)
    # np.save("linear2_average_meta_loss.npy", np.average(loss_hist, axis=0))
    # np.save("test_loss_hist.npy", test_loss_hist)

    meta_model = optimizee_train_args["optimizee"]()
    meta_model.to('cuda')
    meta_optimizer = nn_optimizer.zoopt.ZOOptimizer(optimizee.MetaModel(meta_model))
    meta_optimizer.load_state_dict(torch.load("meta_optimizer_MnistLinearModel2.pt"))
    loss_hist = train_model_with_optimizer(optimizee_train_args, meta_optimizer, device)
    loss_hist = np.average(loss_hist, axis=0)
    np.save("conv_from_linear2_average_meta_loss.npy", loss_hist)

    # loss_hist = np.load("loss_hist.npy")
    # benchmark_loss_hist = np.load("benchmark_loss_hist.npy")
    # benchmark_loss_hist = np.average(benchmark_loss_hist, axis=0)

    # np.save("linear20_average_adam_loss.npy", benchmark_loss_hist)
    # print(benchmark_loss_hist)

    # benchmark_loss_hist = train_benchmark(benchmark_train_args, device)

    # np.save("linear40_average_adagrad_loss.npy", benchmark_loss_hist)

    # fig = plt.figure(facecolor="w", figsize=(20, 5))
    # plt.plot(loss_hist, label="Train Loss")
    # plt.plot(benchmark_loss_hist, label="Benchmark Loss")
    # plt.legend()
    # plt.show()