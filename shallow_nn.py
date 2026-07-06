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
def shallow_1_1_3():
    pass