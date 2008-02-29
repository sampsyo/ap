#!/usr/bin/env python
from numpy import *

#fixme clustering code here?

class CompetitiveLearner:
    def __init__(self, new_neuron, distance, learn, num_neurons=0,
                    neighborhood=1):
        """Initialize a competitive learning environment.
        
        Takes some arguments, beginning with some functions defining learning
        behavior:

            new_neuron(): returns an initialized neuron (an array); called
            	if num_neurons (below) is nonzero or when setup() is called
				manually
				
			distance(v1, v2): returns the distance, by some metric, between
				two vectors (i.e., either neurons or stimuli)
				
			learn(neuron, stimulus, progress): called when neuron is selected
				to learn for stimulus, so neuron should be adjusted (in-place)
				to be "closer" to stimulus; progress is the proportion in
				[0.0,1.0) of completion of the training epochs
				
		Also, a few optional numerical parameters:
		
			num_neurons: number of neurons to create during this
				initialization; defaults to zero, so the user can add more
				later manually or using setup()
		
			neighborhood: the number of neurons to update per stimuls per
				epoch; defaults to 1
		
		After the CompetitiveLearner is initialized with at least one neuron,
		it is suitable to invoke train() to execute the Competitive Learning
		Algorithm.
        """
        
        self.new_neuron = new_neuron
        self.distance = distance
        self.neighborhood = neighborhood
        self.learn = learn
        
        self.setup(num_neurons)
    
    def setup(self, num_neurons=None):
        """Initialize the neuron set with num_neurons nuerons.
        
        Removes any older neurons that might be present. Calls new_neuron() to
        create neurons. If num_nuerons is ommitted, creates as many neurons as
        were present initially.
		"""
        
        # if num_neurons is not provided, reinitialize current neuron set
        if num_neurons is None:
            num_neurons = len(self.neurons)
        
        self.neurons = [] # remove old neurons
        for i in range(num_neurons):
            self.neurons.append(self.new_neuron())
    
    def train_single(self, stimulus, progress):
        """ Train for a single stimulus for a single training epoch.
        
        Progress is the proportion in [0.0,1.0) of completion of the training
        epochs (i.e., current_epoch/total_epochs).
		"""
        #fixme slow, probably : hardcoded to neigh=1
        neurdist = []
        for neuron in self.neurons:
            neurdist.append((neuron, self.distance(neuron, stimulus)))
        neurdist.sort(key=(lambda pair: pair[1])) #fixme no need to sort all
        self.learn(neurdist[0][0], stimulus, progress)
    
    def train(self, stimuli, epochs, debug_afterepoch=None):
        """ Execute the Competitive Learning Algorithm.
        
		stimuli: a list of stimuli that make up the training set
		
		epochs: the number of epochs (iterations over the entire training
			set) to execute
		
		debug_afterepoch(neurons, stimuli): an optional callback function
			invoked after every training epoch with the environment's
			neurons and the training stimuli
		"""
        for epoch in range(epochs):
            progress = (epoch + 0.0)/epochs
            
            for stimulus in stimuli: #fixme: train in random order
                self.train_single(stimulus, progress)
            
            # invoke per-epoch debug callback
            if debug_afterepoch is not None:
                debug_afterepoch(self.neurons, stimuli)