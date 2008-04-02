#!/usr/bin/env python
import cl, cl.euclidean, cl.fscl
from numpy import *
from time import sleep

num_neurons = 20
num_epochs = 10

stimuli = array([[0.1, 0.5], [0.9, 0.5]])
neurons = array([[0.2, 0.5], [0.4, 0.5], [0.45, 0.5], [0.5, 0.5], [0.55, 0.5],
                [0.6, 0.5], [0.8, 0.5]])

learner = cl.fscl.RPCLLearner(distance=cl.euclidean.distance,
                                 learn=cl.euclidean.learn,
                            new_neuron=cl.euclidean.new_random_neuron,
                           num_neurons=0,
                               stimuli=stimuli)
learner.neurons = neurons

cl.euclidean.depict(learner)

learner.train(num_epochs, stimuli, debug_afterepoch=cl.euclidean.depict)

cl.euclidean.depict(learner, True)
sleep(3)
print learner.cluster(stimuli)
