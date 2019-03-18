"""
Copyright 2019 Archie Shahidullah

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
from gliaml import NetworkLayer, NeuralNetwork


np.random.seed(1)  # to recreate results

# following block generates activation function IDs and initialises layers
activations1 = [2 for i in range(0, 4)]
activations2 = [2]

layer1 = NetworkLayer(3, 4, activations1, True)
layer2 = NetworkLayer(4, 1, activations2, True)

# creates network
network = NeuralNetwork(layer1, layer2)

# 7 training samples in a 2D NumPy array
inputs = np.array([[1, 1, 0], [1, 0, 1], [1, 1, 1], [0, 1, 0],
                   [0, 0, 1], [0, 1, 1], [0, 0, 0]])

# Solution: Apply XOR gate to 2nd and 3rd values
# Transposed to match inputs array
outputs = np.array([[1, 1, 0, 1, 1, 0, 0]]).T

# specify how many iterations to change
# can add learning rates if so desired
network.train_mean_squared_error(inputs, outputs, 6)

# answer will contain the output for every neuron in every layer
answer = network.think(np.array([1, 0, 0]))

# return the last layer's output
print(answer[-1])
