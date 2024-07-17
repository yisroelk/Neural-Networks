import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
 
class SingleLayerPerceptron:
    def __init__(self, input_size):
        # Initialization of weights and bias with random values
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand(1)
 
    def activation_function(self, dotp):
        return 1 / (1 + np.exp(-dotp))
        # return  1 if dotp > 0 else 0
 
    def predict(self, inputs):
        # Calculation of the weighted sum of the inputs and application of the activation function
        dotp = np.dot(inputs, self.weights) + self.bias
        return self.activation_function(dotp)
        
 
    def plot_network(self):
        # Creation of a directed graph
        G = nx.DiGraph()

        # Add nodes
        G.add_nodes_from(['Input {}'.format(i+1) for i in range(len(self.weights))])
        G.add_node('Summation Node')
        G.add_node('Output')

        # Add edges
        for i, weight in enumerate(self.weights):
            G.add_edge('Input {}'.format(i+1), 'Summation Node', weight=weight)
        G.add_edge('Summation Node', 'Output', weight=self.bias[0])

        # Place the nodes
        m = np.mean(range(len(self.weights)))
        pos = {'Summation Node': (1, m)}
        for i in range(len(self.weights)):
            pos['Input {}'.format(i+1)] = (0, i)
        pos['Output'] = (2, m)

        # Draw the graph
        nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=1000, node_color='skyblue', font_size=8, arrowsize=20)

        # Bow labels with weights
        edge_labels = {(edge[0], edge[1]): str(round(edge[2]['weight'], 2)) for edge in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        # Show the plot
        plt.show()


if __name__ == "__main__":
    # Creazione di un perceptron con 3 input
    input_size = 4
    perceptron = SingleLayerPerceptron(input_size)
 
    # Input example
    input_data = np.array([0.5, 0.3, 0.8, 0.9])
 
    # Output prediction
    output = perceptron.predict(input_data)
 
    # Print the results
    print("Input:", input_data)
    print("Output:", output)

    # Network visualization
    perceptron.plot_network()