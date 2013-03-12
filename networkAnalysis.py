# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import test
import igraph
import numpy as np
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
    # Start with a directed graph.  We'll combine parallel edges later.
    g = igraph.Graph( directed=True )
    # Add all codewords present in the vector as vertices in the graph
    uniqueCodewords = np.unique( codewordVector )
    g.add_vertices( uniqueCodewords.shape[0] )
    # Dictionary which maps codeword to codeword index
    codewordDictionary = dict((key, value) for (value, key) in enumerate( uniqueCodewords ))
    # Store the codeword name
    g.vs['codeword'] = list( uniqueCodewords )
    # Grab each track
    for trackStart, trackEnd in zip( trackIndices, np.append( trackIndices[1:], codewordVector.shape[0] ) ):
        trackVector = codewordVector[trackStart:trackEnd]
        # Get the vertex indices of the starts end ends of each edge in trackVector
        startIndices = [codewordDictionary[n] for n in trackVector[:-1]]
        endIndices = [codewordDictionary[n] for n in trackVector[1:]]
        # Add all of the edges
        g.add_edges( zip( startIndices, endIndices ) )
    # Set all edge's weight to 0
    g.es['weight'] = 1
    # Remove parallel edges (combining their weight) and self-loops
    g.simplify( combine_edges=sum )
    # Convert from directed to undirected, summing edges which are parallel again
    g.to_undirected( combine_edges = sum )
    return g

# <codecell>

if __name__ == "__main__":
    pitchVectors = np.load( './Data/msd-pitches-2005.npy' )
    trackIndices = np.load( './Data/msd-trackIndices-2005.npy' )
    %prun g = createGraph( packValues( pitchVectors ), trackIndices )
    # Some statistics they use:
    #nx.average_shortest_path_length( G )
    #nx.average_clustering( G )
    #G.degree(0)
    #nx.degree_pearson_correlation_coefficient( G )
    #nx.degree_assortativity_coefficient( G )
    #nx.double_edge_swap( G, 1000000 )

# <codecell>


