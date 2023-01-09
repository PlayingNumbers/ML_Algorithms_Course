import pandas as pd
import numpy as np

# Define Abstract Class
class Layer:
	def __init__(self):
		self.input = None
		self.output = None

	def forward_propagation(self,
						    input):
		raise NotImplementedError

	def backward_propagation(self,
							 output_error,
							 learning_rate):

		raise NotImplementedError

def FCLayer(Layer):
	def __init__(self,
				 input_size,
				 output_size):

		self.weights = np.random.rand(input_size, output_size) - 0.5
		self.bias = np.random.rand(1, output_size) - 0.5

	def forward_propagation(self, input_data):
		self.input = input_data
		self.output = np.dot(self.input,self.weights) + self.bias
		return self.output

	def backward_propagation(self,
							  output_error, 
							  learning_rate):

		## Need to better understand these lines
		input_error = np.dot(output_error, self.weights.T)
		weights_error = np.dot(self.input.T, output_error)

		#update parameters
		self.weights -= learning_rate * weights_error
		self.bias -= learning_rate * output_error # Don't get this part

		return input_error

class ActivationLayer(Layer):
	def __init__(self,
				 activation, 
				 activation_prime):
		self.activation = activation
		self.activation_prime = activation_prime

	def forward_propagation(self, input_data):
		self.input = input_data
		self.output = self.activation(self.input)
		return self.output

	def backward_propagation(self,
							 output_error, 
							 learning_rate):
		return self.activation_prime(self.input) * output_error


class NeuralNetwork:
	def __init__(self,
				 epochs,
				 hidden_layer_size):

		self.epochs = epochs
		self.hidden_layer_size = hidden_layer_size
		self.weights 


	def fit(self, X, y):
		self.X = X
		self.y = y

		for i in range(self.epochs):
			self._feed_forward

		pass 

	def _feed_forward(self):
		pass

	def _back_propogation(self):
		pass

	def predict(self, X):
