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


