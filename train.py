import scipy.io
import random
import net
import numpy as np
import matplotlib.pyplot as plt

# 导入数据;
data = scipy.io.loadmat('data.mat')
train_label = data['train_label']
train_data = data['train_data']
test_label = data['test_label']
test_data = data['test_data']

num_train = 2000
lr = 0.1
weight_decay = 0.001
train_batch_size = 100
test_batch_size = 10000

solver = net.net(train_batch_size, lr, weight_decay)
solver.load_sample_and_label_train(train_data, train_label)
solver.load_sample_and_label_test(test_data, test_label)
solver.initial()


train_error = np.zeros(num_train)

for i in range(num_train):
	net.layer.update_method.iteration = i
	solver.forward_train()
	solver.backward_train()
	solver.update()
	train_error[i] = solver.loss.loss

plt.plot(train_error)
plt.show()
# 测试
solver.turn_to_test(test_batch_size)
solver.forward_test()
print('测试样本的识别率为:', solver.loss.accuracy)
