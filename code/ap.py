#!/usr/bin/env python
from numpy import *
import cl, cl.fscl, cl.string
from cl.string import MutableString
import sys


######## UTILITIES

def ngrams(txt, n):
    """Returns a list of MutableStrings corresponding to txt's n-grams.
    """
    i = 0
    maxi = len(txt) - n
    out = []
    for i in range(maxi + 1):
        out.append(MutableString(txt[i:i+n]))
    return out

def mode(arr):
    """
    Returns a value of arr that has the maximum frequency over all values in
    arr.
    """
    return bincount(asarray(arr)).argmax()


######## ADAPTIVE PARSING

def popularities(vecs, k, debug=False, filt=True):
    """
    Given a matrix whose rows are observations, return a vector of "popularity"
    scores for each observation. Popularity is defined by the size of the
    cluster in which the vector is found. Attempts to create k clusters total.
    Also returns an "id" for each vector that is in common with all similar
    vectors.
    """
    
    cl.string.length = len(vecs[0])
    learner = cl.fscl.RPCLLearner(distance=cl.string.distance,
                                     learn=cl.string.learn,
                                new_neuron=cl.string.new_random_neuron,
                                   stimuli=vecs,
                               num_neurons=k)
    learner.train(100) #fixme param
    
    clusters = learner.cluster(vecs, True)

    pops = zeros(len(vecs), int)
    ids  = zeros(len(vecs), int)

    clust_idx = 0
    for cluster in clusters:
        clust_idx += 1
        for vec_idx in cluster:
            pops[vec_idx] = len(cluster)
            ids[vec_idx] = clust_idx
    
    return (pops, ids)
    
def unipops(pops):
    """
    Transform popularities to be 1 where they are near-modal (that is, the
    vector has approximately the most popular frequency).
    """
    
    # for near-modal tolerance:
    """
    freqs = bincount(pops)
    pafs = [(i, freqs[i]) for i in range(len(freqs))]
    pafs.sort(lambda (pop1, freq1), (pop2,freq2): cmp(freq2,freq1))
    max_freq = pafs[0][1]
    likely_ns = []
    for (pop, freq) in pafs:
        if freq > max_freq - 2: # 2=param!
            likely_ns.append(pop)
        else:
            break
    """
    
    likely_n = mode(pops)
    return [abs(pop - likely_n) < 1 for pop in pops] # 1 = param!
        # Make this a lot more intelligent. Handle multiple modes and near-
        # modes.

def records(pops, ids):
    """
    Return a list of indices of vectors that seem to be the beginnings of
    records in the text.
    """
    # find first contiguous string of popular vectors (perhaps could require
    # that the sequence be at least a certain length?)
    inseq = False
    initseq = [] # the sequence of cluster ids indicating the start of a record
    for i in range(len(pops)):
        if not inseq and pops[i] >= 1: # param?
            inseq = True
        elif inseq and pops[i] < 1: # same param
            break
        if inseq:
            initseq.append(ids[i])
    
    # search for initseq in ids
    found = [] # indices of vectors that begin occurrences of initseq
    finding = [] # first-vec-indices of candidates during search
    for i in range(len(ids)):
        finding = [ finding[j] for j in range(len(finding))
                    if ids[i] == initseq[i - finding[j]] ] # check candidates
        if ids[i] == initseq[0]: # new candidate?
            finding.append(i)
        
        j = 0
        while j < len(finding): # any candidates finished?
            if i - finding[j] + 1 == len(initseq):
                found.append(finding[j])
                del finding[j]
            else:
                j += 1
                
    return found

def fields(vecs, pops, isrec, txt):
    """Return a list of list representing the fields in the records of the
    text.
    """
    w = len(vecs[0])
    
    i = recs[0]+1 # first possible field is after first delimiting vector
    cur_record = []
    out = [cur_record]
    candidate = False # index of possible first vector of seq where pop=0
    while i < len(vecs):
    
        if not candidate and pops[i] == 0: # param
            candidate = i
        
        if candidate and (pops[i] != 0 or isrec[i]): # field must end
            if i - candidate >= w: # candidates succeed only if they were w ago
                cur_record.append(txt[candidate+w-1:i])
            candidate = False
        
        if isrec[i]: # start a new record
            cur_record = []
            out.append(cur_record)
        
        i += 1
    
    return out


######## DIAGNOSTICS

def depict_pops(txt, m, pops, isrec=None):
    w = len(m[0])
    
    charpops = zeros(len(txt), int)
    for i in range(len(m)):
        #print v2s(m[i]), pops[i]
        #print charpops
        charpops[i:i+w] += pops[i]
        #print charpops
    maxcharpop = max(charpops)
    
    out = '<pre>'
    for i in range(len(txt)):
        if isrec != None and isrec[i]: # record starts here!
            out += '<span style="background:blue;font-weight:bold;">|</span>'
        weight = (1 - float(charpops[i]) / maxcharpop) * 100
        #print txt[i], charpops[i], weight
        out += '<span style="background:rgb(100%%,%d%%,%d%%);">%s</span>' % \
            (weight, weight, txt[i])
    return out + '</pre>'


######## MAIN

if __name__ == '__main__':
    instr = sys.stdin.read()
    sys.stdin.close()
    
    # PARAMETERS
    w = 2 # granularity of vectorization
    k = len(instr) # number of vector clusters
    depiction = False # output debug visualization in HTML?
    
    # Optional arguments:
    # . w (n-gram size)
    # . output a depiction (boolean)
    if len(sys.argv) > 1:
        w = int(sys.argv[1])
        if len(sys.argv) > 2:
            depiction = True
    
    m = ngrams(instr, w)
    (pops,ids) = popularities(m, k)
    ups = unipops(pops)
    recs = records(ups, ids)
    
    # flatten recs list to be booleans indexed by character
    isrec = zeros(len(instr), bool)
    for r in recs:
        isrec[r] = True
    
    if depiction:
        print depict_pops(instr, m, ups, isrec)
    else:
        print fields(m, ups, isrec, instr)
