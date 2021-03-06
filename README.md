# GliaML

GliaML is simple, yet powerful machine learning library written in Python.

Currently, it allows a user to easily create a multi-layer perceptron neural network with
the following features:

1. Allow backpropagation with the sigmoid, tanh, and ReLU activation functions.
2. Allow normalisation of a classifier with the softmax function.
3. Allow learning rate hyperparameter functionality (including biases)
4. Allow L2 regularisation to prevent over-fitting

## Creating a Network

In GliaML, neural networks are implemented as objects containing a collection of layers,
which are in turn objects. To create a layer, first a list of activation functions for 
each neuron in the list must be chosen. Then the layer can be created. Bias functionality
can be 

    activations1 = [activation_id for i in range(0, num_neurons)]
    layer1 = NetworkLayer(num_inputs, num_neurons, activations1, bias=False)

After a satisfactory number of layers have been created, the neural network itself can be 
invoked by calling the constructor and providing all the layers. If 

    network = NeuralNetwork(l2_regularization=0.03, layer1, layer2, ...) 
    
## Training a network

After creating your network, you'll want to provide training and testing data. These should
be formatted as NumPy arrays. For example, let us consider the following training set.

    training_inputs = np.array([[1, 0, 1], [0, 0, 1], [1, 1, 0], [0, 1, 0]])
    training_outputs = np.array([[1, 1, 0, 0]]).T
    
The solution is whether or not there is a 1 or 0 in the 3rd place of the input array. We now
need to train the network. We can either use mean-squared error or cross-entropy loss.

    network.train_mean_squared_error(training_inputs, training_outputs, num_iterations, 
                                     learning_rate=0.05, bias_learning_rate=0.05)
    
We can now see the network's response on a problem it hasn't seen before.

    answer = network.think(np.array([1, 1, 1]))
    print(answer[-1])

See [example.py](https://github.com/Archiecool4/GliaML/blob/master/example.py) for another
usage example.