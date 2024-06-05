import torch
from torch.autograd import Variable
import snntorch.functional as SF


import nn_optimizer
from torch import optim
import optimizee

import matplotlib.pyplot as plt
import numpy as np



def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def print_batch_accuracy(model, data, targets, batch_size, train=False):
    output = model(data)
    _, idx = output.sum(dim=0).max(1)
    acc = np.mean((targets.clone() == idx).detach().cpu().numpy())

    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")

def train_printer(model, batch_size, data, targets, test_data, test_targets, epoch, iter_counter, counter, loss_hist, test_loss_hist):
    print(f"Epoch {epoch}, Iteration {iter_counter}")
    print(f"Train Set Loss: {loss_hist[counter]:.2f}")
    print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
    print_batch_accuracy(model, data, targets, batch_size, train=True)
    print_batch_accuracy(model, test_data, test_targets, batch_size, train=False)
    print("\n")


def train_update_rnn(train_args, device):
    # num_epoch = 2
    # updates_per_epoch = 5
    # optimizer_steps = 200
    # truncated_bptt_step = 10
    # batch_size = 32

    num_epoch = train_args["num_epoch"]
    updates_per_epoch = train_args["updates_per_epoch"]
    optimizer_steps = train_args["optimizer_steps"]
    truncated_bptt_step = train_args["truncated_bptt_step"]
    batch_size = train_args["batch_size"]

    meta_model = train_args["optimizee"]()
    meta_model.to(device)

    train_loader, _ = meta_model.dataset_loader(batch_size=batch_size)
    train_loader = iter(cycle(train_loader))


    meta_optimizer = nn_optimizer.zoopt.ZOOptimizer(optimizee.MetaModel(meta_model))
    meta_optimizer.to(device)

    optimizer = optim.Adam(meta_optimizer.parameters(), lr=1e-3)
    optimizer_loss_hist = []

    for epoch in range(num_epoch):
        meta_optimizer.train()
        decrease_in_loss = 0
        final_loss = 0
        for _ in range(updates_per_epoch):
            
            # sample new model
            model = train_args["optimizee"]().to(device)
            model.train()
            data, target = next(train_loader)
            data, target = Variable(data).to(device), Variable(target).to(device)

            # initials
            spk_rec = model(data)

            initial_loss = model.loss(spk_rec, target) # SF.ce_rate_loss()

            for k in range(optimizer_steps // truncated_bptt_step):
                meta_optimizer.reset_state(keep_states=k>0, model=model)

                loss_sum = 0
                prev_loss = torch.zeros(1).to(device)

                for j in range(truncated_bptt_step):
                    # Perfom a meta update using gradients from model
                    # and return the current meta model saved in the nn_optimizer

                    data, target = next(train_loader)
                    # data, target = Variable(data).to(device), Variable(target).to(device)
                    data, target = data.to(device), target.to(device)
                    
                    # compute gradient (LocalZO)
                    spk_rec = model(data)
                    loss = model.loss(spk_rec, target)
                    model.zero_grad()
                    loss.backward(retain_graph=True)


                    meta_model = meta_optimizer.local_zo_meta_update(model)

                    # Compute a loss for a step the meta nn_optimizer using LocalZO
                    spk_rec = meta_model(data)
                    loss = meta_model.loss(spk_rec, target)

                    # loss_sum += (k * truncated_bptt_step + j) * (loss - Variable(prev_loss))
                    loss_sum += loss - Variable(prev_loss)
                    print(loss_sum.item())
                    prev_loss = loss.data

                    if hasattr(meta_optimizer, "reg_loss"):
                        loss_sum += meta_optimizer.reg_loss
                    if hasattr(meta_optimizer, "grad_reg_loss"):
                        loss_sum += meta_optimizer.grad_reg_loss

                # Update the parameters of the meta nn_optimizer
                meta_optimizer.zero_grad()
                loss_sum.backward()
                optimizer_loss_hist.append(loss_sum.item())

                for name, param in meta_optimizer.named_parameters():
                    if param.requires_grad:
                        param.grad.data.clamp_(-1, 1)
                optimizer.step()
            
            decrease_in_loss += loss.item() / initial_loss.item()
            final_loss += loss.item()
            

        print("Epoch: {}, final loss {}, average final/initial loss ratio: {}".format(epoch, final_loss / updates_per_epoch,
                                                                       decrease_in_loss / updates_per_epoch))
    torch.save(meta_optimizer.state_dict(), f"meta_optimizer_{train_args['optimizee'].__name__}.pt")

    return meta_optimizer, optimizer_loss_hist


def train_model_with_optimizer(train_args, meta_optimizer, device):
    

    num_epoch = train_args["num_epoch"]
    batch_size = train_args["batch_size"]

    meta_model = train_args["optimizee"]()
    meta_model.to(device)
    train_loader, test_loader = meta_model.dataset_loader(batch_size=batch_size, test_batch_size=batch_size)

    meta_optimizer.eval().to(device)
    model = train_args["optimizee"]().to(device)
    meta_optimizer.reset_state(keep_states=False, model=model)
    loss_hist = []
    test_loss_hist = []
    counter = 0
    for epoch in range(num_epoch):
        iter_counter = 0
        train_batch = iter(train_loader)
        for train_data, train_target in train_batch:
            train_data, train_target = Variable(train_data).to(device), Variable(train_target).to(device)
            
            model.train()
            spk_rec = model(train_data)
            loss = model.loss(spk_rec, train_target)
            model.zero_grad()
            loss.backward()

            with torch.no_grad():
                _ = meta_optimizer.local_zo_meta_update(model)
            
            loss_hist.append(loss.item())

            with torch.no_grad():
                model.eval()
                test_data, test_targets = next(iter(test_loader))
                test_data, test_targets = Variable(test_data).to(device), Variable(test_targets).to(device)

                # Test set forward pass
                test_spk = model(test_data)
                test_loss = model.loss(test_spk, test_targets)
                test_loss_hist.append(test_loss.item())

                # Print train/test loss/accuracy
                if counter % 50 == 0:
                    train_printer(model, batch_size, train_data, train_target, test_data, test_targets, epoch, iter_counter, counter, loss_hist, test_loss_hist)
                counter += 1
                iter_counter +=1
    

    fig = plt.figure(facecolor="w", figsize=(10, 5))
    plt.plot(loss_hist)
    plt.plot(test_loss_hist)
    plt.title("Loss Curves")
    plt.legend(["Train Loss", "Test Loss"])
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig("loss_curves.png")
    plt.show()

    return loss_hist, test_loss_hist


def train_benchmark(train_args, optimizer, device):
    num_epoch = train_args["num_epoch"]
    batch_size = train_args["batch_size"]
    model = train_args["optimizee"]().to(device)
    optimizer = optimizer(model.parameters(), lr=1e-3)

    train_loader, test_loader = model.dataset_loader(batch_size=batch_size, test_batch_size=batch_size)


    loss_hist = []
    test_loss_hist = []
    counter = 0
    for epoch in range(num_epoch):
        iter_counter = 0
        train_batch = iter(train_loader)
        for train_data, train_target in train_batch:
            train_data, train_target = Variable(train_data).to(device), Variable(train_target).to(device)
            
            model.train()
            spk_rec = model(train_data)
            loss = model.loss(spk_rec, train_target)
            model.zero_grad()
            loss.backward()
            optimizer.step()

            # with torch.no_grad():
            #     _ = meta_optimizer.local_zo_meta_update(model)
            
            loss_hist.append(loss.item())

            with torch.no_grad():
                model.eval()
                test_data, test_targets = next(iter(test_loader))
                test_data, test_targets = Variable(test_data).to(device), Variable(test_targets).to(device)

                # Test set forward pass
                test_spk = model(test_data)
                test_loss = model.loss(test_spk, test_targets)
                test_loss_hist.append(test_loss.item())

                # Print train/test loss/accuracy
                if counter % 50 == 0:
                    train_printer(model, batch_size, train_data, train_target, test_data, test_targets, epoch, iter_counter, counter, loss_hist, test_loss_hist)
                counter += 1
                iter_counter +=1


    fig = plt.figure(facecolor="w", figsize=(10, 5))
    plt.plot(loss_hist)
    plt.plot(test_loss_hist)
    plt.title("Loss Curves")
    plt.legend(["Train Loss", "Test Loss"])
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()
    plt.savefig(f"{train_args['name']}_loss_curves.png")

    return loss_hist, test_loss_hist