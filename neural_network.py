import matplotlib.pyplot as plt
import numpy as np
import random



random.seed(0) # to get repeatable results
input_size = 25 # each input is a vector of length 25
num_hidden = 5 # we'll have 5 neurons in the hidden layer
output_size = 10 # we need 10 outputs for each input
learning_rate = 1

# each hidden neuron has one weight per input, plus a bias weight
hidden_layer = [[random.random() for __ in range(input_size + 1)] for __ in range(num_hidden)]

# each output neuron has one weight per hidden neuron, plus a bias weight
output_layer = [[random.random() for __ in range(num_hidden + 1)] for __ in range(output_size)]
print([n[i] for n in output_layer])

# the network starts out with random weights
network = [hidden_layer, output_layer]

targets = [[1 if i == j else 0 for i in range(10)] for j in range(10)]

inputs = [[1, 1, 1, 1, 1,
           1, 0, 0, 0, 1,
           1, 0, 0, 0, 1,
           1, 0, 0, 0, 1,
           1, 1, 1, 1, 1],
          [0, 0, 1, 0, 0,
           0, 0, 1, 0, 0,
           0, 0, 1, 0, 0,
           0, 0, 1, 0, 0,
           0, 0, 1, 0, 0],
          [1, 1, 1, 1, 1,
           0, 0, 0, 0, 1,
           1, 1, 1, 1, 1,
           1, 0, 0, 0, 0,
           1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1,
           0, 0, 0, 0, 1,
           1, 1, 1, 1, 1,
           0, 0, 0, 0, 1,
           1, 1, 1, 1, 1],
          [1, 0, 0, 0, 1,
           1, 0, 0, 0, 1,
           1, 1, 1, 1, 1,
           0, 0, 0, 0, 1,
           0, 0, 0, 0, 1],
          [1, 1, 1, 1, 1,
           1, 0, 0, 0, 0,
           1, 1, 1, 1, 1,
           0, 0, 0, 0, 1,
           1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1,
           1, 0, 0, 0, 0,
           1, 1, 1, 1, 1,
           1, 0, 0, 0, 1,
           1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1,
           0, 0, 0, 0, 1,
           0, 0, 0, 0, 1,
           0, 0, 0, 0, 1,
           0, 0, 0, 0, 1],
          [1, 1, 1, 1, 1,
           1, 0, 0, 0, 1,
           1, 1, 1, 1, 1,
           1, 0, 0, 0, 1,
           1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1,
           1, 0, 0, 0, 1,
           1, 1, 1, 1, 1,
           0, 0, 0, 0, 1,
           1, 1, 1, 1, 1]]

def neuron_output(neuron, input_with_bias):
    dotp = 0
    for i in range(len(input_with_bias)):
        dotp += (neuron[i] * input_with_bias[i])
    return activation_function(dotp)

def feed_forward(neural_network, input_vector):
    """takes in a neural network
    (represented as a list of lists of lists of weights)
    and returns the output from forward-propagating the input"""
    outputs = []
    # process one layer at a time
    for layer in neural_network:
        input_with_bias = input_vector + [1] # add a bias input
        output = [neuron_output(neuron, input_with_bias) for neuron in layer] # compute the output # for each neuron
        outputs.append(output) # and remember it
    # then the input to the next layer is the output of this one
        input_vector = output
    return outputs


def activation_function(dot):
    # sigmoid
    return 1 / (1 + np.exp(-dot))

def sigmoid_derivative(input):
    return input * (1 - input) 

def mean_squared_error(outputs, targets):
    return np.mean([(output - target) ** 2 for output, target in zip(outputs, targets)]) 

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
    hidden_deltas = [hidden_output * (1 - hidden_output) * np.dot(output_deltas, [n[i] for n in output_layer]) for i, hidden_output in enumerate(hidden_outputs)]
    # adjust weights for hidden layer, one neuron at a time
    for i, hidden_neuron in enumerate(network[0]):
        for j, input in enumerate(input_vector + [1]):
            hidden_neuron[j] -= learning_rate * hidden_deltas[i] * input

# List to store loss values
loss_values = []

# 10,000 iterations seems enough to converge
for epoch in range(10000):
    for input_vector, target_vector in zip(inputs, targets):
        backpropagate(network, input_vector, target_vector)
    if epoch % 1000 == 0:
            # Calculate and print the loss every 1000 epochs
            total_loss = 0
            for input_vector, target_vector in zip(inputs, targets):
                _, outputs = feed_forward(network, input_vector)
                total_loss += mean_squared_error(outputs, target_vector)
            avg_loss = total_loss / len(inputs)
            loss_values.append(avg_loss)
            print(f'Epoch {epoch}, Loss: {total_loss / len(inputs)}')

def predict(input):
    outputs = feed_forward(network, input)[-1]
    for i in range(len(outputs)):
        outputs[i] = round(outputs[i], 2)
    return outputs

# # Plot the loss values
# plt.plot(loss_values)
# plt.xlabel('Epochs (in hundreds)')
# plt.ylabel('Loss')
# plt.title('Loss over time')
# plt.show()


print(predict(inputs[4]))

# # stylized 3
# print(predict([0,1,1,1,0,
#                0,0,0,1,1,
#                0,0,1,1,0,
#                0,0,0,1,1,
#                0,1,1,1,0]))

# # stylized 8
# print(predict([0,1,1,1,0,
#                1,0,0,1,1,
#                0,1,1,1,0,
#                1,0,0,1,1,
#                0,1,1,1,0]))


weights = network[0][0] # first neuron in hidden layer
abs_weights = list(map(abs, weights)) # darkness only depends on absolute value

grid = [abs_weights[row:(row+5)] for row in range(0,25,5)] # turn the weights into a 5x5 grid # [weights[0:5], ..., weights[20:25]]
ax = plt.gca() # to use hatching, we'll need the axis
ax.imshow(grid, cmap=plt.cm.binary, interpolation='none') # here same as plt.imshow # use white-black color scale # plot blocks as blocks

def patch(x, y, hatch, color):
    """return a matplotlib 'patch' object with the specified
    location, crosshatch pattern, and color"""
    return plt.Rectangle((x - 0.5, y - 0.5), 1, 1, hatch=hatch, fill=False, color=color)

# cross-hatch the negative weights
for i in range(5): # row
    for j in range(5): # column
        if weights[5*i + j] < 0: # row i, column j = weights[5*i + j]
            # add black and white hatches, so visible whether dark or light
            ax.add_patch(patch(j, i, '/', "white"))
            ax.add_patch(patch(j, i, '\\', "black"))
plt.show()