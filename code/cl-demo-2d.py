#!/usr/bin/env python
import cl, cl.euclidean, cl.fscl
from numpy import *
from time import sleep
import sys

"""Simple demonstration of CompetitiveLearner on 2-D Euclidean space.
"""

num_neurons = 5
num_epochs = 10

# Learning style can be set on the command line. Defaults to 'cl', plain
# competitive learning.
if len(sys.argv) <= 1:
    style = 'cl'
else:
    style = sys.argv[1]

# Some sample input with (very) clear clusters...
stimuli = array([[0.11, 0.11], [0.12, 0.11], [0.09, 0.12], [0.13, 0.08],
                 [0.51, 0.31], [0.54, 0.31], [0.49, 0.27], [0.48, 0.32],
                 [0.71, 0.91], [0.73, 0.89], [0.72, 0.88], [0.68, 0.91]])

# Initialize the learning environment.
if style == 'cl':
    learner = cl.CompetitiveLearner(distance=cl.euclidean.distance,
                                       learn=cl.euclidean.learn,
                                  new_neuron=cl.euclidean.new_random_neuron,
                                     stimuli=stimuli,
                                 num_neurons=num_neurons)
elif style == 'rpcl':
    learner = cl.fscl.RPCLLearner(distance=cl.euclidean.distance,
                                     learn=cl.euclidean.learn,
                                new_neuron=cl.new_nearby_neuron,
                                   stimuli=stimuli,
                               num_neurons=num_neurons)

# Show initial state.
cl.euclidean.depict(learner)

# Train on the stimuli for 10 epochs, calling depict after each.
learner.train(num_epochs, stimuli, debug_afterepoch=cl.euclidean.depict)

# Show the clusters as colors and text.
cl.euclidean.depict(learner, True)
sleep(3)
print learner.cluster(stimuli)
