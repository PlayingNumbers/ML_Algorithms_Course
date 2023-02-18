import numpy as np

class NeuralNetwork:
    def __init__(self, num_inputs, num_hidden, num_outputs):
        
        # initialize weights and biases for the hidden layer and output layer
        self.hidden_weights = np.random.randn(num_inputs, num_hidden)
        self.hidden_bias = np.zeros((1, num_hidden))
        self.output_weights = np.random.randn(num_hidden, num_outputs)
        self.output_bias = np.zeros((1, num_outputs))
        
    def forward(self, inputs):
        # pass inputs through the hidden layer
        hidden_layer = np.dot(inputs, self.hidden_weights) + self.hidden_bias
        self.hidden_layer_activation = self.sigmoid(hidden_layer)
        
        # pass hidden layer output through the output layer
        output_layer = np.dot(self.hidden_layer_activation, self.output_weights) + self.output_bias
        output_layer_activation = self.sigmoid(output_layer)
        
        return output_layer_activation
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def backward(self, inputs, targets, output):
        # calculate the error in the output layer
        output_error = targets - output
        output_delta = output_error * self.sigmoid_derivative(output)
        
        # calculate the error in the hidden layer
        hidden_error = np.dot(output_delta, self.output_weights.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_layer_activation)
        
        # update the weights and biases for the output layer and hidden layer
        self.output_weights += np.dot(self.hidden_layer_activation.T, output_delta)
        self.output_bias += np.sum(output_delta, axis=0, keepdims=True)
        self.hidden_weights += np.dot(inputs.T, hidden_delta)
        self.hidden_bias += np.sum(hidden_delta, axis=0, keepdims=True)
        
    def train(self, inputs, targets, num_epochs, learning_rate):
        for epoch in range(num_epochs):
            # forward pass
            output = self.forward(inputs)
            
            # backward pass
            self.backward(inputs, targets, output)
            
            # print the mean squared error at each epoch
            mse = np.mean(np.square(targets - output))
            print("Epoch:", epoch, "MSE:", mse)
            
            # update the learning rate at each epoch
            learning_rate *= 0.99

    def predict(self, inputs):
        return self.forward(inputs)

# Create a neural network with 2 inputs, 3 hidden neurons, and 1 output
nn = NeuralNetwork(2, 3, 1)

# Create a set of training data
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])

# Train the neural network for 100 epochs
nn.train(inputs, targets, 100, 0.1)

# Test the neural network on a new set of inputs
new_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
output = nn.forward(new_inputs)

# Print the output
print(output)