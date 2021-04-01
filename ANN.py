# ANN
import numpy as np

#number of neurons in layers
input_neurons, hidden_neurons, output_neurons = 3, 3, 3 # To be removed
lr = learning_rate = 0.05
inputs = np.random.rand(3,1) #image matrix 
training_outputs = np.array([[0,0,0]]).T #matrix of desired output

#Random starting weights and biases
np.random.seed(1)
W1 = np.random.randn(hidden_neurons, input_neurons)
W2 = np.random.randn(output_neurons, hidden_neurons)
B1 = np.zeros((hidden_neurons, 1))
B2 = np.zeros((output_neurons, 1))

def sigmoid(x): return 1 / (1 + np.exp(-x)) # sigmoid activation function
    

def sigmoid_derivative(x): return x * (1-x) # derivative of sigmoid function
    
def forwardstep(X): #from input to output forward
    # X is a column vector of all the pixel values
    Z1 = np.dot(W1, X**2) + B1 #function 1 ax2 + c
    A1 = sigmoid(Z1)        #np.maximize?
    Z2 = np.dot(W2, A1**3) + B2 #function 2 ax3 + c 
    A2 = sigmoid(Z2)
    return A2

# ForwardOutput - A2
# Hidden layer output - A1
# Weighted Sum Output Last Layer Z2
# Weighted Sum Output Hidden Layer Z1


def C_0(A2, training_outputs):
    # Find cost of the function using MSE
    # A2 = A2; E = A2 - training_outputs; E = E**2;cost = np.sum(E)
    cost = -(training_outputs*np.log(A2)+(1-training_outputs)*np.log(1-A2))
    return cost
# A2 = np.array([[0.9,0.5,0.1]]).T; print("Cost: ", C_0(A2, training_outputs))
    
def backwardPropagation(A1, A2, training_outputs, Z1, Z2, inputs, W1, W2, B1, B2, lr):
    # Backward propagation for optimisation of the steps
    dloss_yh =  - (np.divide(training_outputs, A2 ) - np.divide(training_outputs, 1 - A2))    #(A2 - training_outputs)**2  # partial derivative of loss wrt to yh    Loss = -(YLog(yh)+(1-Y)Log(1-yh))   A2 = yh / dloss_yh = A2 - yh
    dloss_Z2 = dloss_yh * sigmoid_derivative(Z2)
    dloss_W2 = 
    dloss_A1 = dloss_Z2 * W2 * 3 * A1**2   
    dloss_Z1 = dloss_A1 * sigmoid_derivative(Z1)
    dloss_W1 = dloss_Z1 * 2 * W1 * inputs
    dloss_B1 = False
    dloss_B2 = False
    W1 -= lr* dloss_W1
    W2 -= lr*dloss_W2
    B1 -= lr*dloss_B1
    B2 -= lr*dloss_B2
    return dloss_W1, dloss_W2, dloss_B1, dloss_B2


for i in range(1):
    output = forwardstep(inputs) # Run through the network forward
    C = C_0(output, training_outputs)
    if i % 100 == 0:
        print(f"After {i} iterations, the cost is {C}")


    print("Input:\n", inputs)
    print("Output:\n", output)
    print("")