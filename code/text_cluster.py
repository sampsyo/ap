#!/usr/bin/env python
from cl import CompetitiveLearner
from numpy import *

length = 5

class MutableString():
    """A wrapper around a character array from numpy that implements a very
    limited mutable string. Attempts to be a little bit efficient.
    """
    def __init__(self, val=None):
        """Initializes a new string.
        
        val is an optional parameter. If it is a string, the MutableString is
        initialized to equal val. If it is an iterable containing ints, it is
        interpreted as a sequence of character codes with which to fill the
        new MutableString. If val is, in particular, a numpy array containing
        ints, it is used as the character array for the MutableString so that
        val and the MutableString are aliased.
        """
        if val is None:
            self.chars = empty(0,int)
        else:
            if type(val) is ndarray and val.dtype is dtype('int'):
                # O(1) creation from int array with aliasing
                self.chars = val
            else:
                # works for both integer lists and strings but copies
                self.chars = empty(len(val),int)
                i = 0
                for c in val:
                    self[i] = c
                    i += 1
    
    def __str__(self):
        return reduce(lambda x,y:x+y, self, '')
    def __repr__(self):
        return 'MutableString(' + repr(str(self)) + ')'
    def __cmp__(self, other):
        for i in range(min(len(self), len(other))):
            if self[i] < other[i]:
                return -1
            elif self[i] < other[i]:
                return 1
            # if character is equal, keep comparing
        # loop finished => strings are either equal or of unequal length
        if len(self) < len(other):
            return -1
        elif len(self) > len(other):
            return 1
        else:
            return 0
    def __delitem__(self, i):
        del self.chars[i]
    def __getitem__(self, i):
        return chr(self.chars[i])
    def __len__(self):
        return len(self.chars)
    def __setitem__(self, i, item):
        if type(item) is str and len(item) == 1: # character (1-length string)
            self.chars[i] = ord(item)
        elif type(item) is int:
            self.chars[i] = item
        else:
            raise ValueError, 'can only substitute characters and integers'

def new_neuron():
	out = random.random_integers(97,122,length)
	return MutableString(out)

def distance(seq1, seq2):
	# for now just proportion of sequences that differ
	num_diffs = 0
	for i in range(length):
		if seq1[i] != seq2[i]: num_diffs += 1
	return float(num_diffs)/length


def learn(neuron, stimulus, progress):
	diffs = []
	for i in range(length):
		if neuron[i] != stimulus[i]: diffs.append(i)
	adaptations = 0
	while (len(diffs) > 0 and adaptations < 1): # at most 1 change
		diff_idx = random.randint(0,len(diffs))
		neuron[diffs[diff_idx]] = stimulus[diffs[diff_idx]]
		del diffs[diff_idx]
		adaptations += 1

stimuli = array(["peach", "pbach", "ceach", "pecbe", "pelch",
		   		 "apple", "fpple", "apcle", "appee", "appge",
				 "grape", "grepe", "grapa", "glape", "graae"])

learner = CompetitiveLearner(new_neuron, distance, learn, 12)

def show_neurons(neurons, stimuli):
    out = ''
    for neuron in learner.neurons:
        out += str(neuron) + ' '
    print out

show_neurons(learner.neurons, stimuli)
learner.train(stimuli, 40, debug_afterepoch=show_neurons)