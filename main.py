import mnist_loader as ml
import my_network


training_data, validation_data, test_data = ml.load_data_wrapper()
net = my_network.NeuralNetwork([784, 30, 10])
net.train(training_data, test_data, 15)
