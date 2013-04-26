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
import igraph

# <markdowncell>

# "Figure 2(a): Examples of the rank-frequency distribution (relative frequencies $z^\prime$ such that $\sum_r z^\prime_r = 1$). For ease of visualization, curves are chronologically shifted by a factor of 10 in the vertical axis."

# <codecell>

# Get .npy files for pitch vectors for 1955, 1965, ...
import glob
fileList = glob.glob( os.path.join( paths.subsamplePath, 'msd-pitches-*5-0.npy' ) )
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
plt.show()
    

# <markdowncell>

# Figure 2(b) "Examples of the density values and their fits, taking $z$ as the random variable. Curves are chronologically shifted by a factor of 10 in the horizontal axis."

# <codecell>

# Just plotting three year's worth here
fileList = [os.path.join( paths.subsamplePath, 'msd-pitches-1965-0.npy' )]
fileList += [os.path.join( paths.subsamplePath, 'msd-pitches-1985-0.npy' )]
fileList += [os.path.join( paths.subsamplePath, 'msd-pitches-2005-0.npy' )]
for n, filename in enumerate( fileList ):
    pitchVectors = np.load( filename )
    rankFrequency, rankCounts, sortedMatrix = getRankFrequency( pitchVectors )
    # Compute the distribution P(z) of codeword counts z
    counts, probabilities = computeDistribution( rankCounts )
    # Plot, shifted horizontally.
    plt.loglog( counts*(10**n), probabilities, '.' )
    # Fitting seems broken right now
    '''
    # Fit with power law
    beta, c, zmin = fitShiftedDiscretePowerLaw( probabilities )
    # Get power law fit curve
    probabilityCurve = computeShiftedDiscretePowerLaw( beta, c, zmin, np.sort( counts ) )
    plt.loglog( np.sort( counts )*(10**n), probabilityCurve, 'k' )
    '''
# Make the axes the same as theirs
#_ = plt.axis( [1, 1e8, 1e-4, 1e-1] )
yearNames = [os.path.split( filename )[1][12:16] for filename in fileList]
# We plotted two lines per year... double the year list
#yearNames = [item for sublist in zip( yearNames, yearNames ) for item in sublist]
plt.legend( yearNames, loc='lower left' )
plt.show()

# <markdowncell>

# This is DAn's intepretation of what is actually being plotted in Figure 2(b), as the above obviously does not match the text.

# <codecell>

# Just plotting three year's worth here
fileList = [os.path.join( paths.subsamplePath, 'msd-pitches-1965-0.npy' )]
fileList += [os.path.join( paths.subsamplePath, 'msd-pitches-1985-0.npy' )]
fileList += [os.path.join( paths.subsamplePath, 'msd-pitches-2005-0.npy' )]
for n, filename in enumerate( fileList ):
    pitchVectors = np.load( filename )
    rankFrequency, rankCounts, sortedMatrix = getRankFrequency( pitchVectors )
    # Compute a log-binned histogram of the counts
    hist, bins = np.histogram( np.log( rankCounts ), 25 )
    # Normalize the histogram bins by bin width
    hist = hist/np.diff( np.exp( bins ) )
    # Plot, shifted horizontally.
    plt.loglog( np.exp( bins[:-1] )*10**n, hist, '.' )
    # Fitting seems broken right now
    '''
    # Fit with power law
    beta, c, zmin = fitShiftedDiscretePowerLaw( probabilities )
    # Get power law fit curve
    probabilityCurve = computeShiftedDiscretePowerLaw( beta, c, zmin, np.sort( counts ) )
    plt.loglog( np.sort( counts )*(10**n), probabilityCurve, 'k' )
    '''
# Make the axes the same as theirs
#_ = plt.axis( [1, 1e8, 1e-4, 1e-1] )
yearNames = [os.path.split( filename )[1][12:16] for filename in fileList]
# We plotted two lines per year... double the year list
#yearNames = [item for sublist in zip( yearNames, yearNames ) for item in sublist]
plt.legend( yearNames, loc='lower left' )
plt.show()

# <markdowncell>

# "Averages hortest path length $l$ versus clustering coefficient $C$ for pitch networks (right) and their randomized versions (left). Randomized networks were obtained by swapping pairs of links chosen at random, avoiding multiple links and self-connections. Values $l$ and $C$ calculated without considering the 10 highest degree nodes (see SI)"

# <codecell>

%%prun
# The variables we'll be computing
averageShortestPathLengths = np.zeros( (10, 2009 - 1955) )
clusteringCoefficients = np.zeros( (10, 2009 - 1955) )
randomizedAverageShortestPathLengths = np.zeros( (10, 2009 - 1955) )
randomizedClusteringCoefficients = np.zeros( (10, 2009 - 1955) )
# Load in pitch vectors for each year
for n, year in enumerate( np.arange( 1955, 2009 ) ):
    for seed in np.arange( 1 ):
        with open( os.path.join( paths.graphmlPath, 'pitches-{}-{}.graphml'.format( year, seed ) ), 'r' ) as f:
            # Read in network
            G = loadGraph( f )
        # Remove top 10 links
        removeTopNodes( G )
        averageShortestPathLengths[seed, n] = averageShortestPathLength( G )
        clusteringCoefficients[seed, n] = clusteringCoefficient( G )
        # Randomly swap pairs of links
        G = randomizeNetwork( G )
        randomizedAverageShortestPathLengths[seed, n] = averageShortestPathLength( G )
        randomizedClusteringCoefficients[seed, n] = clusteringCoefficient( G )
plt.figure( figsize=(8, 8) )
plt.scatter( clusteringCoefficients.flatten(), averageShortestPathLengths.flatten(), c=range( 2009-1955 )*10 )
plt.scatter( randomizedClusteringCoefficients.flatten(), randomizedAverageShortestPathLengths.flatten(), c=range( 2009-1955 )*10, marker='s' )
cbar = plt.colorbar()
cbar.set_ticks( np.arange( 0, 2009-1955, 5 ) )
cbar.set_ticklabels( 1955 + np.arange( 0, 2009-1955, 5 ) )
plt.axis( [0.1, 0.7, 35, 55] )
plt.show()

# <markdowncell>

# "Scattered plot of the strength vs. degree in the original pitch network without any filtering procedure applied."

# <codecell>

# Year is fixed at 1992 I believe
year = 1992
# Create the graph
pitchVectors = np.load( os.path.join( paths.subsamplePath, 'msd-pitches-' + str(year) + '-0.npy' ) )
trackIndices = np.load( os.path.join( paths.subsamplePath, 'msd-trackIndices-' + str(year) + '-0.npy' ) )
G = createGraph( packValues( pitchVectors ), trackIndices )
plt.figure( figsize=(8, 8) )
ax = plt.subplot( 111 )
# Plot strength vs. degree with guideline
ax.scatter( np.array( G.degree( G.vs ) ), np.array( G.strength( G.vs, weights='weight' ) ), c='r', linewidths=0 )
plt.plot( np.arange( 3e3 ), 'k', linewidth=3 )
# Set limits to match
plt.axis( [1, 3e3, 1, 2e5] )
ax.set_xscale('log')
ax.set_yscale('log')

