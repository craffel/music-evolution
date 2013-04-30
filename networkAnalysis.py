# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import test
import igraph
import numpy as np
import tempfile
import os
import matplotlib.pyplot as plt
import paths
import scipy.sparse

# <markdowncell>

# Given a matrix $X \in \Re^{m \times n}$ such that $X_{ij} \in \{0, \ldots, p\}$, this will create an array $x \in \Re^m$ of integers computed as $x_i = \sum_{j} {X_{ij}(p+1)^j}$ which is a unique mapping from the rows to $\{1, \ldots, (p+1)^n\}$

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
    edges = []
    # Grab each track
    for trackStart, trackEnd in zip( trackIndices, np.append( trackIndices[1:], codewordVector.shape[0] ) ):
        trackVector = codewordVector[trackStart:trackEnd]
        # Remove self-links
        trackVector = trackVector[np.append( 1, np.diff( trackVector ) ) != 0]
        # Add in edge list
        edges += zip( trackVector[:-1], trackVector[1:] )
    edgeAttributes = {}
    edgeAttributes['weight'] = [1]*len( edges )
    # Create graph
    g = igraph.Graph( n=np.max( codewordVector ) + 1, edges=edges, edge_attrs=edgeAttributes )
    # Turn parallel edges into weights
    g.simplify( combine_edges='sum' )
    # Delete unconnected vertices
    g.delete_vertices( np.flatnonzero( np.array( g.degree() ) == 0 ) )
    return g

# <codecell>

def createAdjacencyMatrix( codewordVector, trackIndices ):
    """
    Given a vector whose entries are indices of codewords, construct an adjacency matrix for a weighted undirected network
    
    Input:
        codewordVector - vector of codeword indices
        trackIndices - indices in the matrix where a track starts/ends
    Output:
        g - adjacency matrix of weighted undirected graph
    """
    # Number of nodes is the max number in the codeword vector + 1 - assuming nodes go from 0 to max
    nNodes = np.max( codewordVector ) + 1
    g = np.zeros( (nNodes, nNodes) )
    # Grab each track
    for trackStart, trackEnd in zip( trackIndices, np.append( trackIndices[1:], codewordVector.shape[0] ) ):
        trackVector = codewordVector[trackStart:trackEnd]
        # Add weights for these edges
        g[trackVector[:-1], trackVector[1:]] += 1
    # "self-links from a codeword to itself are not considered"
    g[np.arange( nNodes ), np.arange( nNodes )] = 0
    # "We can safely use undirected versions of the networks by removing link directionality and summing up the weights in the two directions"
    g += g.T
    # "Link weights are set to the frequency of occurrence of codeword transitions"
    g /= np.sum( g )
    return scipy.sparse.csr_matrix( g )

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
    return g.average_path_length()

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
    return g.transitivity_avglocal_undirected()

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
    # Delete unused vertices
    g.delete_vertices( np.flatnonzero( np.array( g.degree() ) == 0 ) )
    return g

# <markdowncell>

# "Randomized networks were obtained by swapping pairs of links chosen at random, avoiding multiple links and self-connections."
# 
# From the reference:
# 
# "A convenient numerical algorithm performing such randomization consists of ﬁrst randomly selecting a pair of directed edges A-B and C-D.  The two edges are then rewired in such a way that A becomes connected to D, while C to B. However, in case if one or both of these new links already exist in the network this step is aborted and a new pair of edges is selected. This last restriction prevents the appearance of multiple edges connecting the same pair of nodes. A repeated application of the above rewiring step leads to a randomized version of the original network."

# <codecell>

def randomizeNetworkMaslov( g, nSteps=None ):
    '''
    Randomly swaps pairs of link in a network.  Based on http://www.cmth.bnl.gov/~maslov/sym_generate_srand.m
    *** Removes weight information! ***
    
    Input:
        g - igraph graph object
        nSteps - number of rewiring steps (default 4*nEdges)
    Output: 
        g - randomized graph with no weight information
    '''
    # Set up parameters according to the variable names used in the .m file
    s1 = np.array( g.get_adjacency().data, dtype=np.bool )
    ntry = nSteps
    
    #function srand=sym_generate_srand(s1,ntry)
    #% Syntax: 
    #% srand=sym_generate_srand(s1)
    #% s1 - the adjacency matrix of an undirected network  
    #% ntry - (optional) the number of rewiring steps. If none is given ntry=4*(# of edges in the network)
    #% Output: srand - the adjacency matrix of a randomized network with the same set of in- and out-degrees as the original one 
    
    #nrew=0;
    nrew = 0
    #srand=s1;
    srand = s1
    #[i_srand,j_srand]=find(srand);
    i_srand, j_srand = np.nonzero( srand )
    #aux=find(i_srand>j_srand);
    aux = np.flatnonzero( i_srand > j_srand )
    #i_srand=i_srand(aux);
    i_srand = i_srand[aux]
    #j_srand=j_srand(aux);
    j_srand = j_srand[aux]
    #Ne=length(i_srand);
    Ne = i_srand.shape[0]
    
    #if (nargin < 2) ntry=4*Ne; end;
    if ntry is None:
        ntry = 4*Ne
    
    #for i=1:ntry
    for i in xrange( ntry ):
        #e1=1+floor(Ne*rand);
        e1 = np.random.randint( 0, Ne )
        #e2=1+floor(Ne*rand);
        e2 = np.random.randint( 0, Ne )
        #v1=i_srand(e1);
        v1 = i_srand[e1]
        #v2=j_srand(e1);
        v2 = j_srand[e1]
        #v3=i_srand(e2);
        v3 = i_srand[e2]
        #v4=j_srand(e2);
        v4 = j_srand[e2]
        #if (v1~=v3)&(v1~=v4)&(v2~=v4)&(v2~=v3);
        if v1 != v3 and v1 != v4 and v2 != v4 and v2 != v3:
            #if rand>0.5;
            if np.random.rand() > .5:
                #if (srand(v1,v3)==0)&(srand(v2,v4)==0);
                if srand[v1, v3] == 0 and srand[v2, v4] == 0:
                    #srand(v1,v2)=0;
                    srand[v1, v2] = 0
                    #srand(v3,v4)=0;
                    srand[v3, v4] = 0
                    #srand(v2,v1)=0;
                    srand[v2, v1] = 0
                    #srand(v4,v3)=0; 
                    srand[v4, v3] = 0
                    #srand(v1,v3)=1;
                    srand[v1, v3] = 1
                    #srand(v2,v4)=1;
                    srand[v2, v4] = 1
                    #srand(v3,v1)=1;
                    srand[v3, v1] = 1
                    #srand(v4,v2)=1;    
                    srand[v4, v2] = 1
                    #nrew=nrew+1;
                    nrew = nrew + 1
                    #i_srand(e1)=v1;
                    i_srand[e1] = v1
                    #j_srand(e1)=v3;
                    j_srand[e1] = v3
                    #i_srand(e2)=v2;
                    i_srand[e2] = v2
                    #j_srand(e2)=v4;
                    j_srand[e2] = v4
                #end;
            #else
            else:
                #v5=v3;
                v5 = v3
                #v3=v4;
                v3 = v4
                #v4=v5;
                v4 = v5
                #clear v5;
                del v5
                #if (srand(v1,v3)==0)&(srand(v2,v4)==0);
                if srand[v1, v3] == 0 and srand[v2, v4] == 0:
                    #srand(v1,v2)=0;
                    srand[v1, v2] = 0
                    #srand(v4,v3)=0;
                    srand[v4, v3] = 0
                    #srand(v2,v1)=0;
                    srand[v2, v1] = 0
                    #srand(v3,v4)=0; 
                    srand[v3, v4] = 0
                    #srand(v1,v3)=1;
                    srand[v1, v3] = 1
                    #srand(v2,v4)=1;
                    srand[v2, v4] = 1
                    #srand(v3,v1)=1;
                    srand[v3, v1] = 1
                    #srand(v4,v2)=1; 
                    srand[v4, v2] = 1
                    #nrew=nrew+1;
                    nrew = nrew + 1
                    #i_srand(e1)=v1;
                    i_srand[e1] = v1
                    #j_srand(e1)=v3;
                    j_srand[e1] = v3
                    #i_srand(e2)=v2;
                    i_srand[e2] = v2
                    #j_srand(e2)=v4;
                    j_srand[e2] = v4
                #end;         
            #end;
        #end;
    #end;
    
    # Convert from sparse adjacency matrix back to igraph format
    return igraph.Graph.Adjacency( srand.tolist(), mode=igraph.ADJ_LOWER )

# <codecell>

def randomizeNetwork( g, nSteps=None ):
    '''
    Randomly swaps pairs of link in a network.  Based on Joan Serra's code.
    
    Input:
        g - igraph graph object
        nSteps - number of rewiring steps (default 4*nEdges)
    Output: 
        g - randomized graph with no weight information
    '''
    #r=RANDOM_NUM_XSWAPS*M
    if nSteps is None:
        r = 4*len( g.es )
    else:
        r = nSteps
    #linkednodes = G.adj.keys()
    linkednodes = [v.index for v in g.vs]
    #while r>0:
    while r > 0:
        #i=random.choice(linkednodes)
        i = linkednodes[np.random.randint( 0, len( linkednodes ) )]
        #k=random.choice(linkednodes)
        k = linkednodes[np.random.randint( 0, len( linkednodes ) )]
        #if k in G.adj[i]: continue
        if k in g.neighbors( i ):
            continue
        #j=random.choice(G.adj[i].keys())
        j = np.random.choice( g.neighbors(i) )
        #if k in G.adj[j]: continue
        if k in g.neighbors( j ):
            continue
        #l=random.choice(G.adj[k].keys())
        l = np.random.choice( g.neighbors(k) )
        #if l in G.adj[i] or l in G.adj[j]: continue
        if l in g.neighbors( i ) or l in g.neighbors( j ):
            continue
        #G.add_edge(i,k,weight=G[i][j]['weight'])
        g.add_edge( i, k, weight=g.es[g.get_eid( i, j, 0 )]['weight'] )
        #G.add_edge(j,l,weight=G[k][l]['weight'])
        g.add_edge( j, l, weight=g.es[g.get_eid( k, l, 0 )]['weight'] )
        #G.remove_edge(i,j)
        g.delete_edges( g.get_eid( i, j, 0 ) )
        #G.remove_edge(k,l)
        g.delete_edges( g.get_eid( k, l, 0 ) )
        #r-=1
        r -= 1
    return g

# <codecell>

def plotGraph( g, filename ):
    '''
    Saves a picture of a network (even though they are hard to plot) 

    Input:
        g - igraph graph to plot
        filename - filename to write out to (should end in .pdf probably)
    '''
    visual_style = {}
    visual_style["vertex_size"] = .1
    visual_style["edge_width"] = [.001*float(e['weight']) for e in g.es]
    visual_style["bbox"] = (300, 300)
    visual_style["margin"] = 20
    igraph.plot( g, filename, **visual_style )

# <markdowncell>

# "To distinguish real trends from effects purely induced by heterogeneity, we remove the 10 most connected nodes in the original network."

# <codecell>

def removeTopNodes( g, n=10 ):
    '''
    Removes the n most connected nodes in a graph.
    
    Input:
        g - graph to process - processed in place!
        n - number of nodes to remove
    '''
    # Get the n nodes with highest degree
    topNodes = np.argsort( g.degree() )[-n:]
    g.delete_vertices( topNodes )
    # Delete unconnected vertices
    g.delete_vertices( np.flatnonzero( np.array( g.degree() ) == 0 ) )

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

def saveGraph( g, f ):
    '''
    Saves in a .graphml file using igraph.

    Input:
        g - an igraph graph
        f - filename or handle
    output:
        g - graph
    '''
    g.write_graphml( f )

# <codecell>

# Create the .graphml files if run as a script
if __name__ == "__main__":
    # Load in pitch vectors for each year
    for n, year in enumerate( np.arange( 1955, 2009 ) ):
        print "Writing graphs for year {}".format( year )
        for seed in np.arange( 10 ):
            pitchVectors = np.load( os.path.join( paths.subsamplePath, 'msd-pitches-{}-{}.npy'.format( year, seed ) ) )
            trackIndices = np.load( os.path.join( paths.subsamplePath, 'msd-trackIndices-{}-{}.npy'.format( year, seed ) ) )
            # Create network
            g = createGraph( packValues( pitchVectors ), trackIndices )
            # Perform filtering
            g = disparityFilter( g )
            # Write out
            with open( os.path.join( paths.graphmlPath, 'pitches-{}-{}.graphml'.format( year, seed ) ), 'wb' ) as f:
                saveGraph( g, f )

