import layer


class net:
    def __init__(self, batch_size, lr, weights_decay):
        layer.batch_size = batch_size
        layer.update_method.base_lr = lr
        layer.weights_decay = weights_decay

        # 搭建一个四层的神经网络;
        self.inputs_train = layer.data()  # 训练样本的输入层
        self.inputs_test = layer.data()  # 测试样本的输入层

        self.fc1 = layer.fully_connected_layer(784, 50)
        self.bn1 = layer.batch_normalization_layer(50, 50)
        self.ac1 = layer.activation_layer('relu')
        self.fc2 = layer.fully_connected_layer(50, 50)
        self.bn2 = layer.batch_normalization_layer(50, 50)
        self.ac2 = layer.activation_layer('relu')
        self.fc3 = layer.fully_connected_layer(50, 10)
        self.loss = layer.loss_layer('SoftmaxWithLoss')

    def load_sample_and_label_train(self, sample, label):
        self.inputs_train.get_data(sample, label)

    def load_sample_and_label_test(self, sample, label):
        self.inputs_test.get_data(sample, label)

    def initial(self):
        self.fc1.initialize_weights()
        self.fc2.initialize_weights()
        self.fc3.initialize_weights()

    def forward_train(self):
        self.inputs_train.pull_data()

        self.fc1.get_inputs_for_forward(self.inputs_train.output_sample)
        self.fc1.forward()
        self.bn1.get_inputs_for_forward(self.fc1.outputs)
        self.bn1.forward()
        self.ac1.get_inputs_for_forward(self.bn1.inputs_hat)
        self.ac1.forward()

        self.fc2.get_inputs_for_forward(self.ac1.outputs)
        self.fc2.forward()
        self.bn2.get_inputs_for_forward(self.fc2.outputs)
        self.bn2.forward()
        self.ac2.get_inputs_for_forward(self.bn2.inputs_hat)
        self.ac2.forward()

        self.fc3.get_inputs_for_forward(self.ac2.outputs)
        self.fc3.forward()

        self.loss.get_inputs_for_loss(self.fc3.outputs)
        self.loss.get_label_for_loss(self.inputs_train.output_label)
        self.loss.compute_loss_and_accuracy()

    # 可能训练时的bathsize与测试时的batchsize不一样,所以加了下面四条代码
    def turn_to_test(self, batch_size_test):
        layer.batch_size = batch_size_test

    def turn_to_train(self, batch_size_train):
        layer.batch_size = batch_size_train

    def forward_test(self):
        self.inputs_test.pull_data()

        self.fc1.get_inputs_for_forward(self.inputs_test.output_sample)
        self.fc1.forward()
        self.bn1.get_inputs_for_forward(self.fc1.outputs)
        self.bn1.forward()
        self.ac1.get_inputs_for_forward(self.bn1.test)
        self.ac1.forward()

        self.fc2.get_inputs_for_forward(self.ac1.outputs)
        self.fc2.forward()
        self.bn2.get_inputs_for_forward(self.fc2.outputs)
        self.bn2.forward()
        self.ac2.get_inputs_for_forward(self.bn2.test)
        self.ac2.forward()

        self.fc3.get_inputs_for_forward(self.ac2.outputs)
        self.fc3.forward()

        self.loss.get_inputs_for_loss(self.fc3.outputs)
        self.loss.get_label_for_loss(self.inputs_test.output_label)
        self.loss.compute_loss_and_accuracy()

    def backward_train(self):
        self.loss.compute_gradient()
        self.fc3.get_inputs_for_backward(self.loss.grad_inputs)
        self.fc3.backward()
        self.ac2.get_inputs_for_backward(self.fc3.grad_inputs)
        self.ac2.backward()
        self.bn2.get_inputs_for_backward(self.ac2.grad_inputs)
        self.bn2.backward()
        self.fc2.get_inputs_for_backward(self.bn2.dx)
        self.fc2.backward()
        self.ac1.get_inputs_for_backward(self.fc2.grad_inputs)
        self.ac1.backward()
        self.bn1.get_inputs_for_backward(self.ac1.grad_inputs)
        self.bn1.backward()
        self.fc1.get_inputs_for_backward(self.bn1.dx)
        self.fc1.backward()

    def update(self):
        self.fc1.update()
        self.fc2.update()
        self.fc3.update()
