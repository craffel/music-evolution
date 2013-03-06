# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import matplotlib.pyplot as plt

# <markdowncell>

# "We first count the frequency of usage of pitch codewords (i.e. the number of times each codeword type appears in a sample)"

# <codecell>

def getRankFrequency( matrix ):
    """
    Given a matrix, find the percentage of how often each unique row appears.
    
    Input:
        matrix - matrix whose rank-frequency will be calculated
    Output:
        rankFrequency - vector where the i'th element is the frequency of the i'th most common row
        rankCounts - vector where the i'th element is the COUNT of the i'th most common row
        sortedMatrix - matrix where the i'th row is the i'th most common row from matrix
    """
    # Sort the matrix
    order = np.lexsort(matrix.T)
    matrix = matrix[order]
    # Compute the change across rows
    diff = np.diff(matrix, axis=0)
    # Get rows where the diff is zero - no change
    unique = np.ones(matrix.shape[0], 'bool')
    unique[1:] = (diff != 0).any(axis=1) 
    # Get the unique rows of the matrix
    sortedMatrix = matrix[unique]
    # Get the counts of each row - the space between nonzero indices of unique
    rankCounts = np.diff( np.append( np.nonzero( unique )[0], matrix.shape[0] ) )
    # Sort the matrices
    sortedIndices = np.argsort( rankCounts )[::-1]
    rankCounts = rankCounts[sortedIndices]
    sortedMatrix = sortedMatrix[sortedIndices]
    return rankCounts/(1.0*matrix.shape[0]), rankCounts, sortedMatrix

# <markdowncell>

# This is used to compute the probability of each codeword count.

# <codecell>

def computeDistribution( vector ):
    """
    Given a vector, determine the probability of each entry of the vector.
    
    Input:
        vector - vector whose distribution to compute
    Output:
        elements - unique elements of the vector
        distribution - the distribution of the data
    """
    vector = np.sort( vector )
    elements, indices = np.unique( vector, return_index=True )
    counts = np.diff( np.append( indices, vector.shape[0] ) )
    counts = counts[counts > 0]
    indices = np.argsort( counts )[::-1]
    counts = counts[indices]
    elements = elements[indices]
    return elements, counts/(1.0*vector.shape[0])
    

# <markdowncell>

# Make plots that look like, eg, Fig 2a by using this function with rank-frequency distributions for various years, each multiplied by $10^{year index}$, with different color arguments.

# <codecell>

def plotRankFrequency( rankFrequency, color ):
    """
    Plots the rank-frequency distribution on a log-log scale.
    
    Input:
        rankFrequency - the rank-frequency distribution.
        color - the color of the dots, eg 'b'
    """
    plt.loglog( rankFrequency, '.' + color )

# <codecell>

if __name__ == "__main__":
    # Get .npy files for pitch vectors
    import glob
    fileList = glob.glob('./Data/msd-pitches-*5.npy')
    colors = ['b', 'c', 'g', 'y', 'r', 'm']
    # Generate figure 2a
    plt.figure( figsize = (6, 12) )
    for n, filename in enumerate( fileList ):
        pitchVectors = np.load( filename )
        # Get rank frequency, etc. and plot it (with shifting)
        rankFrequency, rankCounts, sortedMatrix = getRankFrequency( pitchVectors )
        plotRankFrequency( rankFrequency*(10**n), colors[n] )
    plt.legend( fileList, loc='lower left' )

# <codecell>

pitchVectors = np.load( './Data/msd-pitches-2005.npy' )
rankFrequency, rankCounts, sortedMatrix = getRankFrequency( pitchVectors )
counts, probabilities = computeDistribution( rankCounts )
plt.figure()
plt.loglog( counts, probabilities, '.' )
_ = plt.axis( [1, 1e8, 1e-4, 1e-1] )

