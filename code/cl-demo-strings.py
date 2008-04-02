#!/usr/bin/env python
import cl, cl.string, cl.fscl
from numpy import *
import sys

"""Simple demonstration of CompetitiveLearner on strings.
"""

# A clustering problem: discover the names of five-letter fruits.
stimuli = array(["peach", "pbach", "ceach", "pecbe", "pelch",
		   		 "apple", "fpple", "apcle", "appee", "appge",
				 "grape", "grepe", "grapa", "glape", "graae"])
stimuli = map(cl.string.MutableString, stimuli)

# Learning style can be set on the command line. Defaults to 'cl', plain
# competitive learning.
if len(sys.argv) <= 1:
    style = 'cl'
else:
    style = sys.argv[1]

# Set up the learning environment.
if style == 'fscl':
    learner = cl.fscl.FSCLLearner(distance=cl.string.distance,
                                     learn=cl.string.learn,
                                new_neuron=cl.string.new_random_neuron,
                                   stimuli=stimuli,
                               num_neurons=3)
elif style == 'rpcl':
    learner = cl.fscl.RPCLLearner(distance=cl.string.distance,
                                     learn=cl.string.learn,
                                new_neuron=cl.new_nearby_neuron,
                                   stimuli=stimuli,
                               num_neurons=10)
else: # style == 'cl'
    learner = cl.CompetitiveLearner(distance=cl.string.distance,
                                       learn=cl.string.learn,
                                  new_neuron=cl.string.new_random_neuron,
                                     stimuli=stimuli,
                                 num_neurons=3)

# A function we'll use to show training progress.
def show_neurons(learner):
    out = ''
    for neuron in learner.neurons:
        out += str(neuron) + ' '
    print out

# Show the initial, randomized neurons.
show_neurons(learner)

# Train on the stimuli for 40 epochs.
# Add the final parameter to print out progress with show_neurons.
learner.train(100) #, debug_afterepoch=show_neurons)

# Show final neurons & clusters.
show_neurons(learner)
print learner.cluster(stimuli)