# ANN
import numpy as np

def getError():
    return True


#number of neurons in layers
hidden_neurons = 3
input_neurons = 3
output_neurons = 3

inputs = np.random.rand(3,1) #image matrix 
training_outputs = [[1,0,0]] #matrix of desired output

#Random starting weights and biases
np.random.seed(1)
W1 = np.random.randn(hidden_neurons, input_neurons)
W2 = np.random.randn(output_neurons, hidden_neurons)
B1 = np.zeros((hidden_neurons, 1))
B2 = np.zeros((output_neurons, 1))

def sigmoid(x): #sigmoid activation function
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x): 
    return x * (1-x)

def forwardstep(X): #from input to output forward
    Z1 = np.dot(W1, X**2) + B1 #function 1 ax2 + c
    A1 = sigmoid(Z1)        #np.maximize?
    Z2 = np.dot(W2, A1**3) + B2 #function 2 ax3 + c
    A2 = sigmoid(Z2)
    return A2

def backward(X):
    return True

for i in range(1):
    output = forwardstep(inputs)
    print("Input:\n", inputs)
    print("Output:\n", output)
    print("")