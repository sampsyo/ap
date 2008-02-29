#!/usr/bin/env python

"""Extremely simple demonstration of CompetitiveLearner on 2-D Euclidean space.
"""

from cl import CompetitiveLearner, euclidean
from numpy import *
from time import sleep

num_neurons = 5

# Initialize the learning environment.
learner = CompetitiveLearner(euclidean.new_neuron, euclidean.distance,
    euclidean.learn, num_neurons)

# Some sample input with (very) clear clusters...
stimuli = array([[0.11, 0.11], [0.12, 0.11], [0.09, 0.12], [0.13, 0.08],
                 [0.51, 0.31], [0.54, 0.31], [0.49, 0.27], [0.48, 0.32],
                 [0.71, 0.91], [0.73, 0.89], [0.72, 0.88], [0.68, 0.91]])

# Show initial state.
euclidean.depict(learner.neurons, stimuli)

# Train on the stimuli for 10 epochs, calling depict after each.
learner.train(stimuli, 10, debug_afterepoch=euclidean.depict)

euclidean.depict(learner.neurons, stimuli, learner.quantize)
sleep(5)