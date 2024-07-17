import numpy as np


# Get data from a file
def get_data_from_file(file):
    with open(file, "r") as data:
        lines = data.read().splitlines()    
    size = int(lines[0])  # size of data including bias
    inputs = [float(i) for i in lines[1].split(',')]  # data of input layer (first entry must be 1)
    weights = [float(i) for i in lines[2].split(',')]  # weights (first entry is the bias)
    print("Inputs:", weights[1:]) 
    return size, inputs, weights


# Sum the inputs times the weights
def calc_dotp(size, inputs, whights):
    dotp = 0
    for i in range(size):
        dotp += (whights[i] * inputs[i])
    return dotp


# activation function
def activation_function(dot):
    # return 1 if dot > 0 else 0

    # # ReLU
    # return max(0.0, dot)

    # # sigmoid
    return 1 / (1 + np.exp(-dot))


# Main function of the perceptron
def single_layer_perceptron():
    size, inputs, whights = get_data_from_file("data.txt")
    print("Output:", activation_function(calc_dotp(size, inputs, whights)))



# Usage
slp = single_layer_perceptron()