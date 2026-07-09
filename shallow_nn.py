import numpy as np
import matplotlib.pyplot as plt

def ReLU(preactivation):

    # activaion = np.zeros_like(preactivation)
    activation = preactivation.clip(min = 0)

    return activation


# make an array of inputs
z = np.arange(-5, 5, 0.1)
relu = ReLU(z)

# print(relu)

# plot the relu function
fig, ax = plt.subplots()
ax.plot(z, relu, 'r-')
# remember min is -5 in z and max is 5
# so limits are acceptable
ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])
ax.set_xlabel('z')
ax.set_ylabel('Relu[z]')
ax.set_aspect('equal')
plt.savefig("relu_plot.png")
plt.show()

# relu paper -> https://arxiv.org/pdf/1803.08375

# define a shallow neural network with one input, one output and three hidden units
def shallow_1_1_3(x, activation_fn, phi_0, phi_1, phi_2, phi_3, theta_10, theta_11, theta_20, theta_21, theta_30, theta_31):

    # this network is 1 - 3 - 1
    # full math walkthrough should be able to picture and inject the variables into the network

    # the initial lines
    # these are the preactivations
    pre_1 = theta_10 + theta_11 * x
    pre_2 = theta_20 + theta_21 * x
    pre_3 = theta_30 + theta_31 * x

    # pass these through the relu functions
    # to compute the activations
    act_1 = activation_fn(pre_1)
    act_2 = activation_fn(pre_2)
    act_3 = activation_fn(pre_3)

    # weight the activations using phi1, phi2, phi3
    w_act_1 = phi_1 * act_1
    w_act_2 = phi_2 * act_2
    w_act_3 = phi_3 * act_3

    # combine the weighted activations and add phi_0 to create the output
    y = phi_0 + w_act_1 + w_act_2 + w_act_3

    return y, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_1, w_act_2, w_act_3


# plot the shallow neural network
# assume => input in the range [0, 1]
# assume => output in the range [-1, 1]
def plot_neural(x, y, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_1, w_act_2, w_act_3, plot_all=False, x_data=None, y_data=None):

    # plot all intermediate points
    if plot_all:
        fig, ax = plt.subplots(3, 3)
        fig.set_size_inches(8.5, 8.5)
        fig.tight_layouts(pad=3.0)
        ax[0, 0].plot(x, pre_1, 'r-')
        ax[0, 0].set_ylabel("Preactivation")

        ax[0, 1].plot(x, pre_2, 'b-')
        ax[0, 1].set_ylabel("Preactivation")

        ax[0, 2].plot(x, pre_3, 'g-')
        ax[0, 2].set_ylabel("Preactivation")

        ax[1, 0].plot(x, act_1, "Activation")
        ax[1, 0].set_ylabel("Activation")

        ax[1, 1].plot(x, act_2, "Activation")
        ax[1, 1].set_ylabel("Activation")

    for plot_y in range(3):
        for plot_x in range(3):
            ax[plot_y, plot_x].set_xlim([0, 1])
            ax[plot_x, plot_y].set_ylim([-1, 1])
        
        ax[2, plot_y].set_xlabel("Input x")
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel([0, 1])
    ax.set_ylim([-1, 1])
    ax.set_aspect(0.5)

    if x_data is not None:
        ax.plot(x_data, y_data, "mo")



    
    return 0