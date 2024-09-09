import mnist_loader as ml
import my_network


training_data, validation_data, test_data = ml.load_data_wrapper()
net = my_network.NeuralNetwork(
    [784, 30, 10], training_data, test_data, 15, use_saved_model=True
)

print(f"Accuracy: {net.test(test_data)/len(test_data)*100}%")
