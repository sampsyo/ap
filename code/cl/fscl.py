#!/usr/bin/env python
from numpy import *
import cl
import operator

"""Subclasses CompetitiveLearner to use robust competitive learning variants
based on Frequency-Sensitive Competitive Learning.
"""

class FSCLLearner(cl.CompetitiveLearner):
    """
    Frequency-Sensitive Competitive Learning environment.
    
    An implementation of the "conscience" technique for CL, described in
    "Competitive learning algorithms for vector quantization" by
    Ahalt, Krishnamurthy, Chen, and Melton (DOI 10.1016/0893-6080(90)90071-R).
    """
    
    def train(self, epochs, stimuli=None, debug_afterepoch=None):
        # initialize win-counts to zero
        self.neuron_wins = {}
        for neuron in self.neurons:
            self.neuron_wins[id(neuron)] = 0
        
        super(FSCLLearner, self).train(epochs, stimuli, debug_afterepoch)
    
    def activation(self, neuron, stimulus):
        return self.neuron_wins[id(neuron)] * \
                self.distance(self, neuron, stimulus)

    def train_single(self, stimulus):
        winner = self.find_winner(stimulus)
        self.neuron_wins[id(winner)] += 1
        self.learn(self, winner, stimulus, self.learning_rate)

class RPCLLearner(FSCLLearner):
    """
    Rival Penalized Competitive Learning environment.
    
    This class implements simple repulsion of the second-nearest neuron. The
    technique is described in "Rival Penalized Competitive Learning for
    Clustering Analysis, RBF Net, and Curve Detection" by Xu, Krzyzak, and
    Oja (DOI 10.1109/72.238318).
    """
    
    def train_single(self, stimulus):
        
        [winner, rival] = self.find_winner(stimulus, 2)
        self.neuron_wins[id(winner)] += 1
        
        # Learn and unlearn the selected neurons.
        self.learn(self, winner, stimulus, self.learning_rate)
        self.learn(self, rival,  stimulus, -0.8*self.learning_rate)
            #fixme customizable negative learning rate