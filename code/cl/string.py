#!/usr/bin/env python
from numpy import *
import cl.sequence

"""Provides the necessary functions to use CompetitiveLearner with strings.

This module provides a class called MutableString based on numpy character
arrays. Neurons in this module are MutableStrings.

The distance and learn functions are provided by the cl.sequence module. 

Users should probably set the following package globals:

    length: the length of string to be generated by new_neuron; defaults to 5
        for no good reason
    
    low_char, high_char: the range in which new neurons are randomized; default
        to 97 and 122 (lowercase alphabet in ASCII)
"""

####
# MUTABLE STRING CLASS
####

class MutableString(object):
    """A wrapper around an int array from numpy that implements a very
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
            if type(val) is ndarray: # assume int array (numpy092)
                # O(1) creation from int array with aliasing
                self.chars = val
            else:
                # works for both integer lists and strings but copies
                self.chars = empty(len(val),int)
                i = 0
                for c in val:
                    self[i] = c
                    i += 1
    
    def __unicode__(self):
        # numpy's tostring() is not Unicode (c'mon, Python 3000)
        out = u''
        for code in self.chars:
            out += unichr(code)
        return out
    def __str__(self):
        out = ''
        for code in self.chars:
            if code in range(128):
                out += chr(code)
            else:
                out += '?'
        return out
    def __repr__(self):
        return repr(unicode(self))
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
        return unichr(self.chars[i])
    def __len__(self):
        return len(self.chars)
    def __setitem__(self, i, item):
        if (type(item) is str or type(item) is unicode) and len(item) == 1:
            # character (1-length string)
            self.chars[i] = ord(item)
        elif type(item) is int or type(item) is byte or type(item) is long:
            self.chars[i] = item
        else:
            raise ValueError('can only substitute characters and integers')

    def copy(self):
        return MutableString(self.chars.copy())


####
# CL TRAINING PARAMETERS
####

length = 5
low_char = 97
high_char = 122

def new_random_neuron(learner=None):
    """A new_neuron() callback returning a random MutableString with characters
    in [low_char,high_char] of length length.
    
    learner is ignored.
    """
    out = random.random_integers(low_char, high_char, length)
    return MutableString(out)

# get distance and learning functions from sequence module
distance = cl.sequence.distance_hamming
def random_element():
    return random.random_integers(low_char, high_char)
learn = cl.sequence.LearnFunctor(random_element)
