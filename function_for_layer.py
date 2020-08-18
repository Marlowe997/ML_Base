import numpy as np
from scipy import stats


# 各激活函数及其导数

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def der_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def der_tanh(x):
    return 1 - tanh(x) * tanh(x)


def relu(x):
    temp = np.zeros_like(x)
    if_bigger_zero = (x > temp)
    return x * if_bigger_zero


def der_relu(x):
    temp = np.zeros_like(x)
    if_bigger_equal_zero = (x >= temp)
    return if_bigger_equal_zero * np.ones_like(x)


def softmaxwithloss(inputs, label):
    temp1 = np.exp(inputs)
    probability = temp1 / (np.tile(np.sum(temp1, 1), (inputs.shape[1], 1))).T
    temp3 = np.argmax(label, 1)
    temp4 = [probability[i, j] for (i, j) in zip(np.arange(label.shape[0]), temp3)]
    loss = -1 * np.mean(np.log(temp4))
    return loss


def der_softmaxwithloss(inputs, label):
    temp1 = np.exp(inputs)
    temp2 = np.sum(temp1, 1)
    probability = temp1 / (np.tile(temp2, (inputs.shape[1], 1))).T
    gradient = probability - label
    return gradient

# xavier 初始化方法
def xavier(num_neuron_inputs, num_neuron_outputs):
    temp1 = np.sqrt(6) / np.sqrt(num_neuron_inputs + num_neuron_outputs + 1)
    weights = stats.uniform.rvs(-temp1, 2 * temp1, (num_neuron_inputs, num_neuron_outputs))
    return weights
