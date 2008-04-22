#!/usr/bin/env python
from numpy import *
import cl, cl.fscl, cl.string
from cl.string import MutableString
import sys
from optparse import OptionParser
import hotshot


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
    """Returns a value of arr that has the maximum frequency over all values in
    arr.
    """
    return bincount(asarray(arr)).argmax()

def dp(txt):
    """Print debug text to stderr if debug mode is enabled.
    """
    if debug:
        print >>sys.stderr, txt


######## ADAPTIVE PARSING

def popularities(vecs, numclusters=100, epochs=100):
    """
    Given a matrix whose rows are observations, return a vector of "popularity"
    scores for each observation. Popularity is defined by the size of the
    cluster in which the vector is found. Attempts to create numclusters clusters
    total. Also returns an "id" for each vector that is in common with all similar
    vectors.
    """
    
    # perform the clustering
    cl.string.length = len(vecs[0])
    learner = cl.fscl.RPCLLearner(distance=cl.string.distance,
                                     learn=cl.string.learn,
                                new_neuron=cl.string.new_random_neuron,
                                   stimuli=vecs,
                               num_neurons=k)
    learner.train(epochs)
    clusters = learner.cluster(vecs, True)
    if debug: # show the clusters
        out = ''
        for cluster in clusters:
            out += '['
            for idx in cluster:
                out += str(vecs[idx]) + ','
            out += "]\n"
        dp(out)
    
    # generate pops/ids output
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

def find_initseq(pops, ids, threshold=1):
    """Find the first contiguous string of popular vectors.
    """
    #fixme could require that the sequence have a minimum length
    inseq = False
    initseq = [] # the sequence of cluster ids indicating the start of a record
    for i in range(len(pops)):
        if not inseq and pops[i] >= 1: # param?
            inseq = True
        elif inseq and pops[i] < 1: # same param
            break
        if inseq:
            initseq.append(ids[i])
    return initseq

def find_records(pops, ids):
    """Return a list of indices of vectors that seem to be the beginnings of
    records in the text.
    """
    initseq = find_initseq(pops, ids)
    
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

def find_fields(vecs, pops, recs, txt):
    """Return a list of list representing the fields in the records of the
    text.
    """
    #fixme "i in recs" is a low-hanging optimization fruit
    
    w = len(vecs[0])
    
    i = recs[0]+1 # first possible field is after first delimiting vector
    cur_record = []
    out = [cur_record]
    candidate = False # index of possible first vector of seq where pop=0
    while i < len(vecs):
    
        if not candidate and pops[i] == 0: # param
            candidate = i
        
        if candidate and (pops[i] != 0 or i in recs): # field must end
            if i - candidate >= w: # candidates succeed only if they were w ago
                cur_record.append(txt[candidate+w-1:i])
            candidate = False
        
        if i in recs: # start a new record
            cur_record = []
            out.append(cur_record)
        
        i += 1
    
    return out

######## DIAGNOSTICS

def depict_pops(txt, m, pops, recs):
    w = len(m[0])
    
    charpops = zeros(len(txt), int)
    for i in range(len(m)):
        charpops[i:i+w] += pops[i]
    maxcharpop = max(charpops)
    
    out = '<pre>'
    for i in range(len(txt)):
        if i in recs: # record starts here!
            out += '<span style="background:blue;font-weight:bold;">|</span>'
        weight = (1 - float(charpops[i]) / maxcharpop) * 100
        #print txt[i], charpops[i], weight
        out += '<span style="background:rgb(100%%,%d%%,%d%%);">%s</span>' % \
            (weight, weight, txt[i])
    return out + '</pre>'


######## MAIN

def main(instr, w, k, epochs):
    # perform text analysis
    m = ngrams(instr, w)
    (pops,ids) = popularities(m, k, epochs)
    ups = unipops(pops)
    recs = find_records(ups, ids)
    fields = find_fields(m, ups, recs, instr)
    
    return (fields, m, ups)

if __name__ == '__main__':
    # read options
    usage = """usage: %prog [options] file"""
    op = OptionParser(usage=usage)
    op.add_option('-v', dest='debug', action='store_true', default=False,
                  help='output debug information to stderr')
    op.add_option('-d', '--depict', dest='depict', action='store_true',
                  default=False, help='output an HTML depiction of parsing'
                  ' process instad of results')
    op.add_option('-p', '--profile', dest='prof', action='store_true',
                  default=False, help='run hotshot, a performance profiler; do'
                  ' not print parsing results; log profiler data to ap.log')
    op.add_option('-w', dest='w', default='2', metavar='NUM', type='int',
                  help='split text into NUM-grams (default %default)')
    op.add_option('-e', dest='epochs', default='100', metavar='NUM', type='int',
                  help='use NUM epochs when clustering')
    op.add_option('-k', dest='k', default=None, metavar='NUM', type='int',
                  help='try clustering into at most NUM clusters (default'
                  ' length of input)')
    (options, args) = op.parse_args()
    global debug
    debug = options.debug
    depict = options.depict
    w = options.w
    k = options.k
    epochs = options.epochs
    prof = options.prof
    
    # make sure we have a file to parse
    if len(args) < 1:
        op.error('no file to parse')
    elif len(args) > 1:
        op.error('too many arguments')
        
    # read the input file
    infile = args[0]
    infh = open(infile)
    instr = infh.read()
    infh.close()
    
    # if k was not set as option, use the default value
    if not k:
        k = len(instr)
    
    if prof: # run in hotshot profiler
        print >>sys.stderr, 'running profiler... ',
        profile = hotshot.Profile("ap.log")
        profile.runcall(main, instr, w, k, epochs)
        profile.close()
        print >>sys.stderr, 'done'
        
    else: # run normally
        (fields, m, ups) = main(instr, w, k, epochs)
    
        # output
        if depict:
            print depict_pops(instr, m, ups, recs)
        else:
            print fields
