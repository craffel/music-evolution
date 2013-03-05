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

# Test
if __name__ == "__main__":
    import createDatasets
    fileList = createDatasets.getFiles( 'Data', '.npy' )
    colors = ['b', 'c', 'g', 'y', 'r', 'm']
    plt.figure( figsize = (8, 16) )
    for n, filename in enumerate( fileList ):
        pitchVectors = np.load( filename )
        rankFrequency, rankCounts, sortedMatrix = getRankFrequency( pitchVectors )
        plotRankFrequency( rankFrequency*(10**n), colors[n] )
    plt.legend( fileList )

