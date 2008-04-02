#!/usr/bin/env python
from __future__ import division
from numpy import *
import copy

"""A module for using the Competitive Learning Algorithm.

All functionality is accessed by initializing a CompetitiveLearner object with
the algorithm's parameter. One can then train() on datasets.
"""

#####
# USEFUL GENERIC PARAMETERS
#####

def new_nearby_neuron(learner):
    """A new_neuron() callback that creates neurons equal to random stimuli.
    
    Uses learner.stimuli, so this should be set (i.e., passed to the
    constructor, probably).
    
    For large spaces, this can increase the chances that a neuron will be
    placed in a given cluster. Best used with repulsion to avoid overfitting.
    """
    neuron = learner.stimuli[random.randint(len(learner.stimuli))]
    if hasattr(neuron, 'copy'):
        return neuron.copy()
    else:
        return copy.deepcopy(neuron)



#####
# COMPETITIVE LEARNING ENVIRONMENT OBJECT
#####

class CompetitiveLearner(object):
    """An environment that keeps track of the parameters to a simple
    competitive learning algorithm.
    
    To use the algorithm, initialize a CompetitiveLearner with the desired
    parameters. Use train() to execute the algorithm and obtain trained values
    for neurons. Then, either read the neurons array directly or use cluster()
    or quantize() to do VQ.
    
    This base class does no simulated annealing, repulsion, density awareness,
    or anything beyond classic CL. Neurons learn by a constant learning_rate
    passed to the constructor or set later.
    """
    
    def __init__(self, distance, learn, new_neuron, stimuli=None,
            num_neurons=0, learning_rate=0.2):
        """Initialize a competitive learning environment.
        
        Takes some arguments, beginning with some functions defining learning
        behavior:
                
            distance(learner, v1, v2): returns the distance, by some metric,
                between two vectors (i.e., either neurons or stimuli)
                
            learn(learner, neuron, stimulus, amount): called when neuron is
                selected to learn for stimulus, so neuron should be adjusted
                (in-place) to be "closer" to stimulus in proportion to amount,
                e.g.:
                    - if amount is 1.0, neuron should equal stimulus after
                      learning
                    - if amount is -1.0, the neurons should be adjusted by the
                      same amount in the opposite direction
                    - if amount is 0.0, no learning should take place

            new_neuron(learner): returns an initialized neuron; called if
                num_neurons (below) is nonzero or when setup() is called
                manually
        
        These callbacks may access the CompetitiveLearner's state via the
        learner reference. It may be useful, for instance, to use
        learner.progress, a proportion in [0.0,1.0) of completion of the
        current set of training epochs.
        
            stimuli: a list or array of stimuli to train on; these may be
                omitted and passed with the first call to train()
        
            num_neurons: number of neurons to create during this
                initialization; defaults to zero, so the user can add more
                later manually or using setup()
            
            learning_rate: a constant passed as "amount" to the learn()
                function
        
        After the CompetitiveLearner is initialized with at least one neuron,
        it is suitable to invoke train() to execute the Competitive Learning
        Algorithm.
        """
        
        self.new_neuron = new_neuron
        self.distance = distance
        self.learn = learn
        self.progress = 0.0
        self.stimuli = stimuli
        self.learning_rate = learning_rate
        self.setup(num_neurons)
    
    def setup(self, num_neurons=None):
        """Initialize the neuron set with num_neurons neurons.
        
        Removes any older neurons that might be present. Calls new_neuron() to
        create neurons. If num_nuerons is omitted, creates as many neurons as
        were present before setup() was invoked.
        """
        
        # if num_neurons is not provided, reinitialize current neuron set
        if num_neurons is None:
            num_neurons = len(self.neurons)
        
        self.neurons = [] # remove old neurons
        for i in range(num_neurons):
            self.neurons.append(self.new_neuron(self))
    
    def train_single(self, stimulus):
        """Train for a single stimulus for a single training epoch.
        
        Progress is the proportion in [0.0,1.0) of completion of the training
        epochs (i.e., current_epoch/total_epochs).
        
        If more than one neuron has the minimum distance from the stimulus, a
        random neuron with this distance learns.
        """
        min_dist = None
        nearest = [] # neurons with minimum distance from stimulus
        
        for neuron in self.neurons:
            dist = self.distance(self, neuron, stimulus)
            if min_dist is None or dist < min_dist: # new minimum
                min_dist = dist
                nearest = [neuron]
            elif dist == min_dist: # a tie for the current minimum distance
                nearest.append(neuron)
                
        # choose a random neuron with minimum distance from stimulus
        self.learn(self, nearest[random.randint(len(nearest))],
                stimulus, self.learning_rate)
    
    def train(self, epochs, stimuli=None, debug_afterepoch=None):
        """Execute the Competitive Learning Algorithm.
        
        Stimuli are presented in a random order each epoch (invoking
        train_single() on each stimulus each epoch).
        
            epochs: the number of epochs (iterations over the entire training
                set) to execute
                
            stimuli: a list of stimuli that make up the training set; if
                present, overrides the current value of learner.stimuli; if
                absent, uses the current value of learner.stimuli
        
            debug_afterepoch(learner): an optional callback function invoked
                after every training epoch with the environment's neurons and
                the training stimuli
        
        This method sets the learner object's progress field and stimuli field
        (if passed as a parameter) for use by callbacks.
        """
        
        if stimuli is not None:
            self.stimuli = stimuli
        
        for epoch in range(epochs):
            self.progress = epoch/epochs
            
            # choose a random order for presentation of stimuli
            for stimulus_idx in random.permutation(len(self.stimuli)):
                self.train_single(self.stimuli[stimulus_idx])
            
            # invoke per-epoch debug callback
            if debug_afterepoch is not None:
                debug_afterepoch(self)
    
    def quantize(self, stimulus):
        """Return the index of a neuron nearest (by the provided distance
        metric) to stimulus.
        
        If two neurons have equal distance to stimulus, the first is returned.
        If no neurons are present, -1 is returned. This function is only
        meaningful after train() has been run.
        """
        min_dist = None # i.e., infinity
        nearest = -1
        for i in range(len(self.neurons)):
            dist = self.distance(self, self.neurons[i], stimulus)
            if min_dist is None or dist < min_dist:
                nearest = i
                min_dist = dist
        return nearest
    
    def cluster(self, stimuli):
        """Given a list or array of stimuli, group them into clusters (lists
        of stimuli) of equal quantization code.
        
        Returns an array of cluster-lists such that clusters[i] is a list of
        stimuli with quantization i. This function is only meaningful after
        train() has been run.
        """
        
        # initialize an empty array of lists (no better way?)
        clusters = empty(len(self.neurons), dtype=list)
        for i in range(len(clusters)):
            clusters[i] = []
        
        for stimulus in stimuli:
            clusters[self.quantize(stimulus)].append(stimulus)
        
        return clusters
