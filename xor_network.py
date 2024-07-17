import numpy as np


def neuron_output(neuron, input_with_bias):
    print("neuron", neuron)
    dotp = 0
    for i in range(len(input_with_bias)):
        dotp += (neuron[i] * input_with_bias[i])
    return activation_function(dotp)

def activation_function(dot):
    # return 1 if dot > 0 else 0

    # # ReLU
    # return max(0.0, dot)

    # # sigmoid
    return 1 / (1 + np.exp(-dot))

def feed_forward(neural_network, input_vector):
    """takes in a neural network
    (represented as a list of lists of lists of weights)
    and returns the output from forward-propagating the input"""
    outputs = []
    # process one layer at a time
    for layer in neural_network:
        input_with_bias = input_vector + [1] # add a bias input
        output = [neuron_output(neuron, input_with_bias) for neuron in layer] # compute the output # for each neuron
        print("layer", layer)
        outputs.append(output) # and remember it
    # then the input to the next layer is the output of this one
        input_vector = output
    return outputs


xor_network = [# hidden layer
            [[20, 20, -30], # 'and' neuron
            [20, 20, -10]], # 'or' neuron
            # output layer
            [[-60, 60, -30]]] # '2nd input but not 1st input' neuron

# input = [0,0]
# print(input, feed_forward(xor_network, input)[-1])

for x in [0, 1]:
    for y in [0, 1]:
        # feed_forward produces the outputs of every neuron
        # feed_forward[-1] is the outputs of the output-layer neurons
        print(x, y, feed_forward(xor_network,[x, y])[-1])



