#!/usr/bin/env python
from cl import CompetitiveLearner
from numpy import *

length = 5

def new_neuron():
	out = random.random_integers(65,122,length) # some ASCII chars
	return map(chr, out)

def distance(vec1, vec2):
	# for now just proportion of vectors that differ
	num_diffs = 0
	for i in range(length):
		if vec1[i] != vec2[i]: num_diffs += 1
	return (num_diffs + 0.0)/length

neighborhood = 1

def learn(neuron, stimulus, progress):
	#fixme should settle with progress
	#fixme slow, probably
	diffs = []
	for i in range(length):
		if neuron[i] != stimulus[i]: diffs.append(i)
	adaptations = 0
	while (len(diffs) > 0 and adaptations < 2): # at most 2 changes
		if len(diffs) == 1: # hacky: randint doesn't like max=min
			diff_idx = 0
		else:
			diff_idx = random.randint(0,len(diffs)-1)
		neuron[diffs[diff_idx]] = stimulus[diffs[diff_idx]]
		del diffs[diff_idx]
		adaptations += 1

num_neurons = 3

def strarr(string):
	out = []
	for c in string:
		out.append(c)
	return array(out)

stimuli = map(strarr,["peach", "pbach", "ceach", "pecbe", "pelch",
		   			  "apple", "fpple", "apcle", "appee", "appge",
					  "grape", "grepe", "grapa", "glape", "graae"])

learner = CompetitiveLearner(new_neuron, distance, neighborhood,
	learn, num_neurons)
print learner.neurons
learner.train(stimuli, 20)
print learner.neurons