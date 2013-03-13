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

def createGraph( codewordVector, trackIndices ):
    """
    Given a vector whose entries are indices of codewords, construct a weighted undirected graph
    
    Input:
        codewordVector - vector of codeword indices
        trackIndices - indices in the matrix where a track starts/ends
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
    fd, tempFilename = tempfile.mkstemp()
    nx.write_gml( G, tempFilename )
    g = igraph.Graph.Read_GML( tempFilename )
    os.close( fd )
    os.remove( tempFilename )
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
    shortestPathLengths = g.shortest_paths( weights='weight' )
    return np.mean( shortestPathLengths )

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
    localTransitivity = np.array( g_read.transitivity_local_undirected( weights='weight' ) )
    # Sometimes this returns NaNs!
    localTransitivity = np.delete( localTransitivity, np.nonzero( np.isnan(localTransitivity) )[0] )
    return np.mean( localTransitivity )

# <codecell>

pitchVectors = np.load( './Data/msd-pitches-2005.npy' )
trackIndices = np.load( './Data/msd-trackIndices-2005.npy' )
g = createGraph( packValues( pitchVectors ), trackIndices )
print averageShortestPathLength( g )
print clusteringCoefficient( g )

