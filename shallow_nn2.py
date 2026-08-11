import numpy as np

import matplotlib.pyplot as plt

# draw 2d function
def draw_2d_function(ax, x1_mesh, x2_mesh, y):

    pos = ax.contour(x1_mesh, x2_mesh, y, levels=256, cmap="hot", vmin=-10, vmax=10.0)

    ax.set_xlabel('x1')
    ax.set_ylabel("x2")

    levels = np.arange(-10, 10, 1.0)

    ax.contour(x1_mesh, x2_mesh, levels, cmap="winter")
    # return 0


def plot_neural_2_inputs(x1, x2, y, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_1, w_act_2, w_act_3):

    fig, ax = plt.subplots(3, 3)
    fig.set_size_inches(8.5, 8.5)

    fig.tight_layout(pad=3.0)

    fig, ax = plt.subplots()
    draw_2d_function(ax, x1, x2, y)
    ax.set_title("Network output, y")
    ax.set_aspect(1.0)
    plt.show()


    # return 0


def relu(preactivation):

    activation = preactivation.clip(0.0)
    return activation

# define a shallow neural network with two inputs, one output and three hidden units
def shallow_2_1_3(x1, x2, activation_fn, phi_0, phi_1, phi_3, theta_10, theta_11, theta_12, theta_20, theta_21, theta_22, theta_30, theta_31, theta_32):

    pre_1 = np.zeros_like(x1)
    pre_2 = np.zeros_like(x1)
    pre_3 = np.zeros_like(x1)

    return 0


# defining some parameters and running the nn
theta_10 = -4.0
theta_11 = 0.9
theta_12 = 0.0

theta_20 = 5.0
theta_21 = -0.9
theta_22 = -0.5

theta_30 = -7
theta_31 = 0.5
theta_32 = 0.9

phi_0 = 0.0
phi_1 = -2.0
phi_2 = 2.0
phi_3 = 1.5

x1 = np.arange(0.0, 10.0, 0.1)
# print(x1)
x2 = np.arange(0.0, 10.0, 0.1)
x1 = np.meshgrid(x1)
x2 = np.meshgrid(x2)

