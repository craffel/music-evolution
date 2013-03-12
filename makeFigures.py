# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# This should replicate all of the figures in "Measuring the Evolution of Contemporary Western Popular Music"

# <codecell>

import test
import paths
from rankFrequency import *
from distributions import *
from networkAnalysis import *
import matplotlib.pyplot as plt
import numpy as np
import os

# <markdowncell>

# "Figure 2(a): Examples of the rank-frequency distribution (relative frequencies $z^\prime$ such that $\sum_r z^\prime_r = 1$). For ease of visualization, curves are chronologically shifted by a factor of 10 in the vertical axis."

# <codecell>

# Get .npy files for pitch vectors for 1955, 1965, ...
import glob
fileList = glob.glob( os.path.join( paths.subsamplePath, 'msd-pitches-*5.npy' ) )
# Colors for each line (a vaguely close match)
colors = ['b', 'c', 'g', 'y', 'r', 'm']
# Their figure is of odd size.
plt.figure( figsize = (6, 12) )
for n, filename in enumerate( fileList ):
    # Get pitch vectors for this file
    pitchVectors = np.load( filename )
    # Get rank frequency, etc. and plot it (with shifting)
    rankFrequency, rankCounts, sortedMatrix = getRankFrequency( pitchVectors )
    # Use the rankFrequency module's plotRankFrequency, with a vertical shift
    plotRankFrequency( rankFrequency*(10**n), colors[n] )
# Get year names out of file names (kind of a hack but hey) and make them the legend
yearNames = [os.path.split( filename )[1][12:16] for filename in fileList]
plt.legend( yearNames, loc='lower left' )
    

# <markdowncell>

# Figure 2(b) "Examples of the density values and their fits, taking $z$ as the random variable. Curves are chronologically shifted by a factor of 10 in the horizontal axis."

# <codecell>

# Just plotting three year's worth here
fileList = [os.path.join( paths.subsamplePath, 'msd-pitches-1965.npy' )]
fileList += [os.path.join( paths.subsamplePath, 'msd-pitches-1985.npy' )]
fileList += [os.path.join( paths.subsamplePath, 'msd-pitches-2005.npy' )]
for n, filename in enumerate( fileList ):
    pitchVectors = np.load( filename )
    rankFrequency, rankCounts, sortedMatrix = getRankFrequency( pitchVectors )
    # Compute the distribution P(z) of codeword counts z
    counts, probabilities = computeDistribution( rankCounts )
    # Plot, shifted horizontally.
    plt.loglog( counts*(10**n), probabilities, '.' )
    # Fit with power law
    zmin, beta, c = fitShiftedDiscretePowerLaw( probabilities )
    # Get power law fit curve
    probabilityCurve = computeShiftedDiscretePowerLaw( zmin, beta, c, np.sort( counts ) )
    plt.loglog( np.sort( counts )*(10**n), probabilityCurve, 'k' )
# Make the axes the same as theirs
#_ = plt.axis( [1, 1e8, 1e-4, 1e-1] )
yearNames = [os.path.split( filename )[1][12:16] for filename in fileList]
# We plotted two lines per year... double the yea rlist
yearNames = [item for sublist in zip( yearNames, yearNames ) for item in sublist]
plt.legend( yearNames, loc='lower left' )

# <markdowncell>

# "Averages hortest path length $l$ versus clustering coefficient $C$ for pitch networks (right) and their randomized versions (left). Randomized networks were obtained by swapping pairs of links chosen at random, avoiding multiple links and self-connections. Values $l$ and $C$ calculated without considering the 10 highest degree nodes (see SI)"

# <codecell>

# The variables we'll be computing
averageShortestPathLength = np.zeros( 2009 - 1955 )
clusteringCoefficient = np.zeros( 2009 - 1955 )
# Load in pitch vectors for each year
for n, year in enumerate( np.arange( 1955, 2009 ) ):
    pitchVectors = np.load( os.path.join( paths.subsamplePath, 'msd-pitches-' + str(year) + '-0.npy' ) )
    trackIndices = np.load( os.path.join( paths.subsamplePath, 'msd-trackIndices-' + str(year) + '-0.npy' ) )
    # Create network
    G = createGraph( packValues( pitchVectors ), trackIndices )
    averageShortestPathLength[n] = nx.average_shortest_path_length( G )
    clusteringCoefficient[n] = nx.average_clustering( G )
plt.scatter( clusteringCoefficient, averageShortedPathLength, c=plt.cm.rainbow( 2009-1955 ) )

# <codecell>

