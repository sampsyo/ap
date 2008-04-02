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
        return self.neuron_wins[id(neuron)] / self.epoch * \
                self.distance(self, neuron, stimulus)

    def train_single(self, stimulus):
        winner = self.find_winner(stimulus)
        self.neuron_wins[id(winner)] += 1
        self.learn(self, winner, stimulus, self.learning_rate)

class RPCLLearner(cl.CompetitiveLearner):
    """
    Rival Penalized Competitive Learning environment.
    
    This class implements simple repulsion of the second-nearest neuron. The
    technique is described in "Rival Penalized Competitive Learning for
    Clustering Analysis, RBF Net, and Curve Detection" by Xu, Krzyzak, and
    Oja (DOI 10.1109/72.238318).
    """
    
    def train_single(self, stimulus):
        
        # Get an ordered list of the neurons nearest the stimulus.
        neurs_with_dists = []
        for i in range(len(self.neurons)):
            neurs_with_dists.append(
                    (self.neurons[i],
                     self.distance(self, stimulus, self.neurons[i])))
        neurs_with_dists.sort(key=operator.itemgetter(1)) #fixme partialsort
        
        print neurs_with_dists
        
        # Get all neurons tied for minimum distance to stimulus. Also find the
        # neuron nearest to the minimum distance (next_nearest).
        nearest_neurons = []
        min_dist = neurs_with_dists[0][1]
        for neur_with_dist in neurs_with_dists:
            if neur_with_dist[1] <= min_dist: # dist to stimulus is min_dist
                nearest_neurons.append(neur_with_dist[0])
            else:
                next_nearest = neur_with_dist[0]
                break
        
        # If there are multiple nearest neurons, randomly select one as the
        # winner and another as the rival. Otherwise, the nearest neuron is the
        # winner and next_nearest is the rival.
        if len(nearest_neurons) > 1:
            print "RANDOM WINNER"
            winner_idx = random.randint(len(nearest_neurons))
            winner = nearest_neurons[winner_idx]
            del nearest_neurons[winner_idx]
            rival_idx = random.randint(len(nearest_neurons))
            rival = nearest_neurons[rival_idx]
        else:
            print "DETERMINISTIC WINNER"
            winner = nearest_neurons[0]
            rival = next_nearest
        
        print str(stimulus[0]) + ', ' + str(stimulus[1])
        print 'WINNER: ' + str(self.distance(self, winner, stimulus))
        print '    ' + str(winner[0]) + ', ' + str(winner[1])
        print 'RIVAL:  ' + str(self.distance(self, rival, stimulus))
        print '    ' + str(rival[0]) + ', ' + str(rival[1])
        print
        
        # Learn and unlearn the selected neurons.
        self.learn(self, winner, stimulus, self.learning_rate)
        self.learn(self, rival,  stimulus, -0.1*self.learning_rate)
            #fixme customizable negative learning rate