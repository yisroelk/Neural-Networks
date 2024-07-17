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

input = [0,0]
print(input, feed_forward(xor_network, input)[-1])

# for x in [0, 1]:
#     for y in [0, 1]:
#         # feed_forward produces the outputs of every neuron
#         # feed_forward[-1] is the outputs of the output-layer neurons
#         print(x, y, feed_forward(xor_network,[x, y])[-1])



def backpropagate(network, input_vector, targets):
    hidden_outputs, outputs = feed_forward(network, input_vector)
    # the output * (1 - output) is from the derivative of sigmoid
    output_deltas = [output * (1 - output) * (output - target) for output, target in zip(outputs, targets)]
        # adjust weights for output layer, one neuron at a time
    for i, output_neuron in enumerate(network[-1]):
        # focus on the ith output layer neuron
        for j, hidden_output in enumerate(hidden_outputs + [1]):
            # adjust the jth weight based on both
            # this neuron's delta and its jth input
            output_neuron[j] -= output_deltas[i] * hidden_output
    # back-propagate errors to hidden layer
    hidden_deltas = [hidden_output * (1 - hidden_output) * dot(output_deltas, [n[i] for n in output_layer]) for i, hidden_output in enumerate(hidden_outputs)]
    # adjust weights for hidden layer, one neuron at a time
    for i, hidden_neuron in enumerate(network[0]):
        for j, input in enumerate(input_vector + [1]):
            hidden_neuron[j] -= hidden_deltas[i] * input