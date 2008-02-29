#!/usr/bin/env python
from numpy import *
import sys


######## UTILITIES

def s2v(s):
    """
    Convert a string to a binary vector where each eight-bit block corresponds
    to a single character.
    """
    return array(
                # list of 8-bit lists for each character
                [ [(ord(c) >> i) & 1 for i in range(7,-1,-1)] for c in s ]
           ).flatten()
def v2s(v):
    """
    Convert a binary vector to a string.
    """
    return ''.join([chr(
                # the value of the ith 8-bit string in v
                sum([ v[i*8 + j] * 2**(7-j) for j in range(8) ])
            ) for i in range(len(v)/8)])

def vectext(txt, w):
    """
    Returns a matrix whose rows correspond to the vector encodings of each
    offset in txt from 0 to len(txt)-w.
    """
    i = 0
    maxi = len(txt) - w
    m = []
    while (i <= maxi):
        m.append(s2v(txt[i:i+w]))
        i += 1
    return array(m)

def myvq(obs, codebook, retClusters=False):
    """
    Computes the best entry in the codebook for each observation. The vq
    methods from scipy seem broken (C version overflows, Python versions fail
    mysteriously). retClusters causes the function to return a list of lists
    with containing observation-distortion pairs where each list represents a
    cluster (rather than the return values yielded by scipy's vq).
    """
    if retClusters:
        clusters = [[] for i in range(len(codebook))]
        obsnum = 0
    else:
        codes = []
        dists = []
    for o in obs:
        mindist = inf
        bestcode = -1
        code = 0
        for c in codebook:
            d = sqrt(nansum([ (o[i] - c[i])**2 for i in range(len(o)) ]))
            if d < mindist:
                mindist = d
                bestcode = code
            code += 1
        if retClusters:
            clusters[bestcode].append((obsnum, mindist))
        else:
            codes.append(bestcode)
            dists.append(mindist)
        if retClusters:
            obsnum += 1
    if retClusters:
        return clusters
    else:
        return (array(codes), array(dists))

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
    
    # identify vectors with clusters
    from scipy.cluster.vq import whiten, kmeans
    whitened = whiten(vecs)
    (book, d) = kmeans(nan_to_num(whitened), k) # kmeans doesn't like NaNs
    
    if filt: # filter for items roughly equidistant from neurons
        clusters = myvq(whitened, book, True)
    
        pops = zeros(len(vecs), int)
        ids  = zeros(len(vecs), int)
   
        idx = 0
        for clust in clusters:
            dists = array([dist for (obs, dist) in clust])
            avg = dists.mean()
            thresh = dists.std() # *parameter?
            
            if debug:
                for (obs,dist) in clust:
                    print v2s(vecs[obs]), dist, (abs(avg-dist) <= thresh)
            
            clust = [(obs, dist) for (obs,dist) in clust
                        if abs(avg-dist) <= thresh]
            dists = array([dist for (obs, dist) in clust])
            
            # drop entire cluster if, even after filtering outliers, stdev is
            # too high
            if dists.std() > 0.001: # parameter!
                if debug:
                    print 'GARBAGE CLUSTER', dists.std()
                clust = []
            
            pop = len(clust)
            if clust: # non-garbage
                clust_id = idx
            else:
                clust_id = -1
            for (obs,dist) in clust:
                pops[obs] = pop
                ids[obs]  = clust_id
            
            if debug:
                print '-----------'
            
            idx += 1
    
    else: # unfiltered
        (codes, d) = myvq(whitened, book)
        
        freqs = bincount(codes)
        pops = vectorize(lambda c: freqs[c])(codes)
    
        if debug:
           cafs = [(code, freqs[code]) for code in range(len(freqs))]
           cafs.sort(lambda x,y: -cmp(x[1], y[1]))
           for caf in cafs:
               print caf[1]
               for i in range(len(codes)):
                   if codes[i] == caf[0]:
                       print v2s(vecs[i]), d[i]
               print '-----------'
        
        ids = codes
    
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
    """
    Return a list of list representing the fields in the records of the
    text.
    """
    w = len(v2s(vecs[0]))
    
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
    w = len(v2s(m[0]))
    
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
    
    if len(sys.argv) > 1:
        w = int(sys.argv[1])
        if len(sys.argv) > 2:
            depiction = True
    
    m = vectext(instr, w)
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
