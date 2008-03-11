#!/usr/bin/env python
from __future__ import division
from numpy import *

"""Provides some support for using CompetitiveLearner with sequences (i.e.,
strings, base-pair sequences, &c). The assumption is that any two items in a
sequence are either completely different or completely equal; for instance, the
character 'a' is as different from 'z' as it is from 'b'.

distance_subtitutions() is a very simple distance function. Distance functions
tolerating insertions and deletions are forthcoming. learn() is also provided
to adjust sequences to become more similar.

Users can set the package-global integer learning_rate. This is the maximum
number of adaptations that can occur in a neuron per stimulus per epoch. No
simulated annealing currently occurs. Defaults to 1.
"""

learning_rate = 1 # parameter to learn() bounds number of character subs

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

def learn(learner, neuron, stimulus):
    """Changes at most learning_rate items in neuron to match the corresponding
    slots in stimulus.
    """
    length = min(len(neuron), len(stimulus))
    
    # build up a list of the indices at which neuron and stimulus differ
    diffs = []
    for i in range(length):
        if neuron[i] != stimulus[i]: diffs.append(i)
    
    adaptations = 0 # how many adaptations have we completed so far?
    while (len(diffs) > 0 and adaptations < learning_rate):
        # repeat permutation until sequences match or we've done learning_rate
        # substitutions
        
        diff_idx = random.randint(0,len(diffs)) # select a random difference
        neuron[diffs[diff_idx]] = stimulus[diffs[diff_idx]]
        del diffs[diff_idx] # don't reselect this difference
        adaptations += 1
