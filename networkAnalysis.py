# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import test
import igraph
import networkx as nx
import numpy as np
import tempfile
import os
import matplotlib.pyplot as plt
import paths

# <markdowncell>

# Given a matrix $X \in \Re^{m \times n}$ such that $X_{ij} \in \{0, \ldots, p\}$, this will create an array $x \in \Re^m$ of integers computed as $x_i = \sum_{j} {X_{ij}(p+1)^j}$ which is a unique mapping from the rows to $\{1, \ldots (p+1)^n\}$

# <codecell>

def packValues( matrix ):
    """ 
    Creates a vector corresponding to a unique representation of rows of an integer valued matrix

    Input:
        matrix - matrix to encode
    Output:
        vector - encoded vector
    """
    p = np.max( matrix ) + 1
    values = p**np.array( [np.arange( matrix.shape[1] )[::-1]] ).T
    return np.dot( matrix, values ).T[0]

# <markdowncell>

# "To study the transitions between codewords, we build a complex weighted directed network for pitch, timbre, and loudness descriptions by representing each codeword by a node and placing a directed link between any two beat-consecutive codewords (self-links from a codeword to itself are not considered). Link weights are set to the frequency of occurrence of codeword transitions... we can
# safely use undirected versions of the networks by removing link directionality and summing up the weights in the two directions"

# <codecell>

def createGraph( codewordVector, trackIndices, filename=None ):
    """
    Given a vector whose entries are indices of codewords, construct a weighted undirected graph
    
    Input:
        codewordVector - vector of codeword indices
        trackIndices - indices in the matrix where a track starts/ends
        filename - where to save the .graphml file; default None which doesn't save a file
    Output:
        g - a weighted undirected graph
    """
    # nx.Graph is the undirected graph class
    G = nx.Graph()
    # Grab each track
    for trackStart, trackEnd in zip( trackIndices, np.append( trackIndices[1:], codewordVector.shape[0] ) ):
        trackVector = codewordVector[trackStart:trackEnd]
        # Iterate over codewords
        for start, end in zip( trackVector[:-1], trackVector[1:] ):
            # Don't store self-links
            if start == end:
                continue
            # Make sure all links go from smaller to larger number - this will effectively sum up weights for the undirected graph.
            elif start > end:
                temp = start
                start = end
                end = temp
            # If we already have this edge, just add to its weight
            if G.has_edge( start, end ):
                G[start][end]['weight'] += 1
            # If we don't have it, add it
            else:
                G.add_edge( start, end, {'weight':1} )
    # Embarrassingly, this is the fastest way to do this I can figure out.
    if filename is None:
        fd, tempFilename = tempfile.mkstemp()
        nx.write_graphml( G, tempFilename )
        g = igraph.Graph.Read_GraphML( tempFilename )
        os.close( fd )
        os.remove( tempFilename )
    else:
        nx.write_graphml( G, filename )
        g = igraph.Graph.Read_GraphML( filename )
    return g

# <markdowncell>

# "The global properties of the network are probed by means of the average shortest path length"

# <codecell>

def averageShortestPathLength( g ):
    """
    Computes the average shortest path length of a graph.
    
    Input:
        g - A graph
    Output:
        averageShortestPathLength - ...
    """
    shortestPathLengths = np.array( g.shortest_paths( weights='weight' ) ).flatten()
    # This will return inf if there is no path
    return np.ma.masked_invalid( shortestPathLengths ).mean()

# <markdowncell>

# "The local order of the network is measured by the clustering coecient C. This coecient is obtained as an average of the local clustering coecient of all nodes of degrees above 1"

# <codecell>

def clusteringCoefficient( g ):
    """
    Computes the clustering coefficient of a graph.
    
    Input:
        g - A graph
    Output:
        clusteringCoefficient - ...
    """
    localTransitivity = np.array( g.transitivity_local_undirected( weights='weight' ) )
    # Sometimes this returns NaNs!
    return np.ma.masked_invalid( localTransitivity ).mean()

# <markdowncell>

# "A link accounting for a fraction of the total node's strength is considered relevant if the probability of observing such a value under the null model is smaller than $\alpha$, where $1 - \alpha$ is the confidence level."

# <codecell>

def disparityFilter( G, alpha=0.01 ):
    '''
    Applies the "disparity filter" to a graph.
    
    Input:
        g - graph to process
        alpha - alpha confidence parameter, default = .01
    Output:
        g - processed graph
    '''
    g = G.copy()
    # A link of weight wij...
    wij = np.array( g.es['weight'] )
    # attached to two nodes of degrees ki and kj...
    ki = np.array( g.degree( [e.source for e in g.es] ) )
    kj = np.array( g.degree( [e.target for e in g.es] ) )
    # and strengths si and sj...
    si = np.array( g.strength( [e.source for e in g.es], weights='weight' ) )
    sj = np.array( g.strength( [e.target for e in g.es], weights='weight' ) )
    # will be preserved iff (below is integral solved)
    relevancei = np.power( (1 - wij/si), ki - 1 )
    # or 
    relevancej = np.power( (1 - wij/sj), kj - 1 )
    # is less than alpha
    edgesToDelete = np.flatnonzero( np.logical_and( relevancei > alpha, relevancej > alpha ) )
    g.delete_edges( list( edgesToDelete ) )
    return g

# <codecell>

def loadGraph( f ):
    '''
    Loads in a .graphml file using igraph.

    Input:
        f - filename or handle
    output:
        g - graph
    '''
    return igraph.Graph.Read_GraphML( f )

# <codecell>

# Create the .graphml files if run as a script
if __name__ == "__main__":
    # Cheap parallelization
    import sys
    seed = 0
    if len( sys.argv ) > 1:
        seed = int( sys.argv[1] )
    # Load in pitch vectors for each year
    for n, year in enumerate( np.arange( 1980, 2009 ) ):
            pitchVectors = np.load( os.path.join( paths.subsamplePath, 'msd-pitches-{}-{}.npy'.format( year, seed ) ) )
            trackIndices = np.load( os.path.join( paths.subsamplePath, 'msd-trackIndices-{}-{}.npy'.format( year, seed ) ) )
            # Create network
            G = createGraph( packValues( pitchVectors ), trackIndices, os.path.join( paths.graphmlPath, 'pitches-{}-{}.graphml'.format( year, seed ) ) )

