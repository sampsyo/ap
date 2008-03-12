#!/usr/bin/env python
from __future__ import division
from numpy import *

"""Provides some support for using CompetitiveLearner with sequences (i.e.,
strings, base-pair sequences, &c). The assumption is that any two items in a
sequence are either completely different or completely equal; for instance, the
character 'a' is as different from 'z' as it is from 'b'.

distance_hamming() is a very simple distance function. Distance functions
tolerating insertions and deletions are forthcoming. learn() is also provided
to adjust sequences to become more similar.
"""

def distance_hamming(learner, seq1, seq2):
    """Extremely simple distance metric for sequences; equal to the number of
    positions at which the sequences differ.
    
    If one sequence is longer than the other, the excess is ignored.
    """
    length = min(len(seq1), len(seq2))
    
    num_diffs = 0
    for i in range(length):
        if seq1[i] != seq2[i]: num_diffs += 1
    return num_diffs

class LearnFunctor(object):
    """Facilitates creation of learning functions for sequences.
    
    An instance of LearnFunctor is a callable object that can be used for
    CompetitiveLearner's learn() callback. It must be paramaterized with a
    random_element function for creating new elements in the sequence.
    """
    
    def __init__(self, random_element):
        """Create a new learn() callback functor.
        
        random_element() takes no arguments and returns a random sequence
        element. The callback is used to create differences when unlearning.
        """
        self.random_element = random_element
    
    def __call__(self, learner, neuron, stimulus, amount):
        """Adjusts values in the neuron stimulus to match or differ from the
        stimulus sequence.
        
        amount is the maximum proportion of indices in neuron that will be
        changed. random_element is used to create differences. This only occurs
        when amount is negative.
        """
        length = min(len(neuron), len(stimulus))
    
        # Build up a list of the indices at which we can make adjustments.
        # If the learning amount is positive, we will look for differences
        # (which will be turned into similarities). Otherwise, we look for
        # similarities (to be made different).
        possible_indices = []
        for i in range(length):
            if neuron[i] != stimulus[i]: possible_indices.append(i)
    
        adaptations = 0 # How many adaptations have we completed so far?
        max_adaptations = abs(int(amount * length))
        while (len(possible_indices) > 0 and adaptations < max_adaptations):
            # Adapt until we have created the requested adjustment or no more
            # adaptations are possible.
        
            # Complete one random adaptation.
            adapt_idx = random.randint(0,len(possible_indices))
            if amount >= 0.0: # Create a similarity.
                neuron[possible_indices[adapt_idx]] = \
                            stimulus[possible_indices[adapt_idx]]
            else: # Create a difference.
                while (neuron[possible_indices[adapt_idx]] == \
                            stimulus[possible_indices[adapt_idx]]):
                    neuron[possible_indices[adapt_idx]] = self.random_element()
            del possible_indices[adapt_idx]
            adaptations += 1
