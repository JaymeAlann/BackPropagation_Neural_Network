import numpy as np
from tkinter import *
import color as c
from random import random
import _thread
from typing import List


class NeuralNetwork:

    def __init__(self, input_neurons: int, hidden_neurons: int, output_neurons: int) -> None:
        self.network = list()  # Create a network array that contains the layers
        # Dictionaries for the weights of each layer
        # We use n_inputs + 1 to include weight for the Bias Neuron
        hidden_layer = [{'network_weights': [random() for w_n in range(input_neurons + 1)]} for w_n in
                        range(hidden_neurons)]
        self.network.append(hidden_layer)  # Creating weights into the network adds neurons
        output_layer = [{'network_weights': [random() for w_n in range(hidden_neurons + 1)]} for w_n in
                        range(output_neurons)]
        self.network.append(output_layer)  # Adding the weights for the output neurons
        np.random.seed(1)

    def traverse_network(self, layer_inputs: 'np.ndarray') -> list:
        for layer in self.network:  # Travel through the network one layer at a time
            layer_outputs = []  # Collection of outputs of a particular layer
            for neuron in layer:  # Calculate each individual neuron
                # Calculate the Riemann Sum of inputs * weights
                # Then input that into the sigmoid activation function
                # sig( i1w1 + i2w2 + ... + inwn)
                neuron['layer_outputs'] = self.sigmoid(self.riemann_sum(neuron['network_weights'], layer_inputs))
                layer_outputs.append(neuron['layer_outputs'])
            layer_inputs = layer_outputs  # Add to the next layers inputs
        return layer_inputs

    # Activation Function
    def sigmoid(self, x: float) -> List[float]:
        return 1.0 / (1.0 + np.exp(-x))

    # Calculate the slope of the output value
    def sigmoid_derivative(self, x: List[float]) -> List[float]:
        return x * (1.0 - x)

    # Pass in the list of expected values
    # Then perform backward propagation on each layer of the network
    # calculate the error and each neurons delta to be used in next layer
    def backward_propagation(self, y: list) -> None:
        for k in reversed(range(len(self.network))):  # Start from the end of the network and work your way back
            current_layer = self.network[k]  # defines what layer of the network we are changing
            error_cost_matrix = list()  # Create a list of each weight cost
            if k != len(self.network) - 1:
                for n in range(len(current_layer)):
                    error_cost = 0
                    for neuron in self.network[k + 1]:
                        error_cost += (neuron['network_weights'][n] * neuron['delta'])
                    error_cost_matrix.append(error_cost)
            else:
                for n in range(len(current_layer)):
                    neuron = current_layer[n]
                    error_cost_matrix.append(y[n] - neuron['layer_outputs'])
                    neuron['delta'] = error_cost_matrix[n] * self.sigmoid_derivative(neuron['layer_outputs'])

    # Calculates the Reimann sum of the inputs and the outputs
    def riemann_sum(self, neuron_weights: list, neuron_inputs: 'np.ndarray') -> float:
        # Weights are passed in from the network traversal
        r_sum = neuron_weights[-1]  # Assume the bias is the last input
        for n in range(len(neuron_weights) - 1):
            r_sum += neuron_weights[n] * neuron_inputs[n]
        return r_sum

    # Train the network
    # Iterate the training set through the network n amount of times
    # Then calculating the error and back propagating through the weights
    # 1: Travers the Network and get the networks outputs
    # 2: Expected value (y) will be placed on the end of training_inputs array
    # 3: Calculate the error using the sum squared of the y - outputs
    # 4: Back Propagate
    # 5: Update the weights
    def train_network(self, training_inputs: 'np.ndarray', training_outputs: int, learning_iterations: int) -> None:
        for iterator in range(learning_iterations):
            total_layer_error = 0
            for current_layer in training_inputs:
                layer_outputs = self.traverse_network(current_layer)
                y = [0 for k in range(training_outputs)]  # y is the expected output
                y[current_layer[-1]] = 1
                total_layer_error += sum([(y[k] - layer_outputs[k]) ** 2 for k in range(len(y))])  # sum squared
                self.backward_propagation(y)  # Backwards propogate through the code to update weights
                self.update_layers_weights()

    # Update each weight by 0.1 of its error (delta)
    # Add this to change to each weights
    def update_layers_weights(self) -> None:
        for n in range(len(self.network)):
            if n != 0:
                current_inputs = [neuron['layer_outputs'] for neuron in self.network[n - 1]]
                for neuron in self.network[n]:
                    for k in range(len(current_inputs)):
                        neuron['network_weights'][k] += 0.1 * neuron['delta'] * current_inputs[k]
                    neuron['network_weights'][-1] += 0.1 * neuron['delta']

    # Run non-training data through the network
    # Network predicts the outcome.
    def think(self, n_input: 'np.ndarray') -> str:
        output = self.traverse_network(n_input)
        if output[0] > output[1]:
            return 'BRIGHT'
        else:
            return 'DARK'


class AI_Frame(Frame):

    def __init__(self) -> None:
        Frame.__init__(self)
        # Create the grid for the 2x2 AI
        self.cells = []
        self.grid()
        self.master.title('AI Neural Network')
        self.main_grid = Frame(self, bd=4, bg=c.GRID_COLOR, width=500, height=500)
        self.main_grid.grid(pady=(100, 0))
        self.make_GUI()
        # Top Frame holds the matrix input along with start button
        top_frame = Frame(self)
        top_frame.place(relx=0.5, y=45, anchor='center')
        Label(top_frame, text='Enter Neuron Inputs', font=c.LABEL_FONT).grid(row=0)
        self.user_input = Entry(top_frame, borderwidth=5)
        self.user_input.grid(row=1, column=0)
        self.bot_frame = Frame(self, bd=4, height=100)
        self.bot_frame.grid()
        self.start_ai_btn = Button(self.bot_frame, text='Prediction', font=c.BUTTON_FONT,
                                   command=lambda: button_click())
        self.start_ai_btn.grid(row=0, column=2, padx=50, pady=10, rowspan=2)
        self.results_label = Label(self.bot_frame, text='')
        self.results_label.grid(row=3, column=2)

        def button_click() -> None:
            input_ = self.user_input.get()
            input_ = np.array(input_.split(','))
            input_ = [int(s) for s in input_]
            try:
                _thread.start_new_thread(self.run, (input_,))
            finally:
                print("Finished")

        self.mainloop()

    def make_GUI(self) -> None:
        for i in range(2):
            row = []
            for j in range(2):
                cell_frame = Frame(self.main_grid, bg=c.BRIGHT_CELL_COLOR, width=150, height=150)
                cell_frame.grid(row=i, column=j, padx=5, pady=5)
                cell_holder = Label(self.main_grid, bg=c.BRIGHT_CELL_COLOR)
                cell_holder.grid(row=i, column=j)
                cell_data = {"frame": cell_frame, "holder": cell_holder}
                row.append(cell_data)
            self.cells.append(row)

    def run(self, input_array: 'np.ndarray') -> None:
        # First 4 are inputs [1, 1, 1, 1]
        # Next to last is your bias node
        # Last is your expected output
        training_set = np.array([[-1, -1, 1, -1, 1, 1],
                                 [-1, 1, -1, -1, 1, 1],
                                 [1, -1, -1, -1, 1, 1],
                                 [1, 1, 1, -1, 1, 0],
                                 [1, -1, 1, -1, 1, 0],
                                 [-1, 1, -1, 1, 1, 0]])

        n_inputs = len(training_set[0]) - 2
        n_outputs = len(set([row[-1] for row in training_set]))
        nn = NeuralNetwork(n_inputs, 2, n_outputs)
        nn.train_network(training_set, n_outputs, 2000)
        network_input = np.array(input_array)
        network_input = np.append(network_input, 1)
        index = 0
        for row in range(2):
            for col in range(2):
                if network_input[index] == -1:
                    self.cells[row][col]['frame'].configure(bg=c.DARK_CELL_COLOR)
                    self.cells[row][col]['holder'].configure(bg=c.DARK_CELL_COLOR)
                    index += 1
                    self.update_idletasks()
                else:
                    self.cells[row][col]['frame'].configure(bg=c.BRIGHT_CELL_COLOR)
                    self.cells[row][col]['holder'].configure(bg=c.BRIGHT_CELL_COLOR)
                    index += 1
                    self.update_idletasks()
        ouput = nn.think(network_input)
        self.results_label.configure(text=ouput)


AI_Frame()
