"""Parametric PINN Architecture.

Defines the neural network model that takes (x, y, Re) as input
and predicts (u, v, p) — velocity components and pressure.
"""

import tensorflow as tf
import numpy as np


class ParametricPINN(tf.keras.Model):
    """Parametric Physics-Informed Neural Network.
    
    Input: (x, y, Re) — spatial coordinates + Reynolds number
    Output: (u, v, p) — velocity field + pressure
    """

    def __init__(self, layers=6, neurons=64, activation='tanh'):
        super(ParametricPINN, self).__init__()
        self.hidden = [
            tf.keras.layers.Dense(neurons, activation=activation,
                                  kernel_initializer='glorot_normal')
            for _ in range(layers)
        ]
        self.output_layer = tf.keras.layers.Dense(3)  # [u, v, p]

    def call(self, inputs):
        """Forward pass.
        Args:
            inputs: Tensor of shape (N, 3) — [x, y, Re]
        Returns:
            Tensor of shape (N, 3) — [u, v, p]
        """
        x = inputs
        for layer in self.hidden:
            x = layer(x)
        return self.output_layer(x)
