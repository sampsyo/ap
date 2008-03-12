#!/usr/bin/env python
import cl, cl.string
from numpy import *

"""Simple demonstration of CompetitiveLearner on strings.
"""

# A clustering problem: discover the names of five-letter fruits.
stimuli = array(["peach", "pbach", "ceach", "pecbe", "pelch",
		   		 "apple", "fpple", "apcle", "appee", "appge",
				 "grape", "grepe", "grapa", "glape", "graae"])
stimuli = map(cl.string.MutableString, stimuli)
num_neurons = 12

# Set up the learning environment.
learner = cl.CompetitiveLearner(distance=cl.string.distance,
                                   learn=cl.string.learn,
                              new_neuron=cl.string.new_random_neuron,
                                 stimuli=stimuli,
                             num_neurons=num_neurons)

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
learner.train(40) #, debug_afterepoch=show_neurons)

# Show final clusters.
print learner.cluster(stimuli)