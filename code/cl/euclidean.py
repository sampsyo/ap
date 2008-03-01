#!/usr/bin/env python

"""Provides the necessary functions to use CompetitiveLearner in Euclidean
space. For now, hard-coded to two dimensions, but this could easily be changed
in the future.

Our neurons and stimuli are coordinates (arrays of size 2). Distance is
Euclidean. Learning moves a neuron along the line between it and the stimulus.
The plane is in [0,1]x[0,1].

At the expense of a dependency on matplotlib, can draw neurons and stimuli in
a 2-D plane for debugging purposes.

Users can set the package-global float learning_rate. This value is the
largest proportion of the difference between neuron and stimulus that a neuron
can travel per stimulus per epoch. Less motion occurs as training progresses.
learning_rate defaults to 0.5.
"""

from pylab import *
from time import sleep
from numpy import *



####
# LEARNING PARAMETERS FOR THE EUCLIDEAN PLANE
####

learning_rate = 0.5 # base learning rate; decreases linearly with progress

def new_neuron():
    # Neurons are randomly placed in the [0,1]x[0,1] plane.
    return random.random(2)

def distance(v1, v2):
    # Simple Euclidean distance metric.
    return sqrt(sum((v1-v2)**2))

def learn(neuron, stimulus, progress):
    # Find the line between the neuron and the stimulus and move the neuron
    # along it toward the stimulus by some proportion. In our case, we begin
    # by moving the neuron (learning_rate) * (distance between neuron and
    # stimulus) but decrease this linearly as learning progresses. 
    
    # distance to move the neuron toward the stimulus
    dist = distance(neuron, stimulus) * (1 - progress) * learning_rate
    
    # translate this distance into delta-X and delta-Y to change coords
    slope = (stimulus[1] - neuron[1])/(stimulus[0] - neuron[0])
    dx = sqrt(abs(dist**2/(slope**2 + 1)))
    if stimulus[0] < neuron[0]: # ensure we travel in the correct direction
        dx = -dx
    dy = slope * dx
    
    # update neuron position
    neuron[0] += dx
    neuron[1] += dy



####
# GRAPHICAL DEBUGGING
####

ion() # "interactive mode" to show animations

# A few utility functions for coping with MPL.
def draw_certainly():
	"""
	A ridiculous hack for eliminating a race condition in matplotlib when using
	the GTKAgg backend. Replaces draw() but really, honestly does it.
	"""
	draw()
	sleep(0.01)
	draw()
def clear_scatters(axes=None):
	"""
	Remove all scatter plots from the given subplot (or the first one, by
	default).
	"""
	if axes is None:
		axes = subplot(111)
	scatters = axes.collections
	del scatters[0:len(scatters)-1]

# The matter at hand.
def depict(neurons, stimuli, quantize=None):
    """Show, via matplotlib, the current positions of the neurons (red squares)
    and stimuli (blue circles, by default).
    
    If quantize is specified, use it as a one-argument function mapping stimuli
    to color indices.
    """

    # matplotlib needs arrays of x-coords and y-coords
    [neurons_x, neurons_y] = transpose(neurons)
    [stimuli_x, stimuli_y] = transpose(stimuli)

    ax = subplot(111)
    clear_scatters(ax)
    ax.scatter(neurons_x, neurons_y, marker='s', c='r')
    if quantize is None:
        stimulus_color = 'b'
    else:
        stimulus_color = map(quantize, stimuli)
    ax.scatter(stimuli_x, stimuli_y, marker='o', c=stimulus_color)

    draw_certainly()
    sleep(0.2)