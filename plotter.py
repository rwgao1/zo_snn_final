import numpy as np
import matplotlib.pyplot as plt

# x = [np.load(f"benchmark_mlp_adam_{i+1}.npy") for i in range(7)]
# print(x[0].shape)
# for i in range(7):
#     plt.plot(x[i], label=f"adam_{i+1}")
# x = np.average(x, axis=0)
# x = x[0:1850]
# np.save("linear40_average_adam2_loss.npy", x)
# plt.plot(x, label="SGD")
# plt.show()

# x = np.load("losses/loss_hist.npy")
# x = np.average(x, axis=0)
# plt.plot(x, "hi")

# np.save("linear2_average_adam_loss.npy", x)

def plot_losses(model_name, benchmark=False):
    line_width = 1
    line_styles = ["-", "--", "-.", ":"]
    average_meta_loss = np.load(f"losses/{model_name}_average_meta_loss.npy")
    average_adam_loss = np.load(f"losses/{model_name}_average_adam_loss.npy")
    average_sgd_loss = np.load(f"losses/{model_name}_average_sgd_loss.npy")
    average_adagrad_loss = np.load(f"losses/{model_name}_average_adagrad_loss.npy")
    average_vmeta_loss = np.load(f"losses/V{model_name}_average_meta_loss.npy")
    if model_name != "linear20":
        average_20_transfer_loss = np.load(f"losses/{model_name}_from_linear20_average_meta_loss.npy")
    if model_name == "conv":
        conv_from_linear2_loss = np.load("losses/conv_from_linear2_average_meta_loss.npy")
    
    plt.plot(average_adam_loss, label="Adam", linewidth=line_width)
    plt.plot(average_sgd_loss, label="SGD", linewidth=line_width)
    plt.plot(average_adagrad_loss, label="Adagrad", linewidth=0.5)
    if not benchmark:
        if model_name != "linear20":
            plt.plot(average_20_transfer_loss, label="LocalZO + LSTM (from Linear 20)", linewidth=line_width)
        if model_name == "conv":
            plt.plot(conv_from_linear2_loss, label="LocalZO + LSTM (from Linear 2)", linewidth=line_width)
        plt.plot(average_vmeta_loss, label="LocalZO + LSTM (V)", linewidth=line_width)
        plt.plot(average_meta_loss, label="LocalZO + LSTM", linewidth=line_width)
    

    plt.legend()
    plt.title(f"Average Loss over 10 Runs on {model_name}")
    plt.xlabel("iter")
    plt.ylabel("Cross-Entropy Rate Loss")
    plt.show()

plot_losses("conv", benchmark=True)
plot_losses("conv", benchmark=False)

# optim_loss_hist = np.load("optim_loss_hist.npy")
# hists = []
# for i in range(10):
#     hists.append(optim_loss_hist[i * 100:(i + 1) * 100])
# print(len(optim_loss_hist))
# plt.plot(hists[9][0:10])
# plt.show()
