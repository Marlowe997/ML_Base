import numpy as np

momentum = 0.9
base_lr = 0
iteration = -1


def inv(gamma=0.0005, power=0.75):
    if iteration == -1:
        assert False, '需要在训练过程中,改变update_method 模块里的 iteration 的值'
    return base_lr * np.power((1 + gamma * iteration), -power)


def fixed():
    return base_lr


def batch_gradient_descent(weights, grad_weights, previous_direction):
    lr = inv()
    direction = momentum * previous_direction + lr * grad_weights
    weights_now = weights - direction
    return weights_now, direction
