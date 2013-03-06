# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import msd
import os
import tables
import scipy.stats.mstats

# <codecell>

def getFiles( path, extension ):
    """ 
    Get files of a certain type in a directory, recursively
    
    Inputs:
        path - The path to look for files in
        extension - The extension to look for
    Outputs:
        fileList - a list of all files with this extension, including their paths relative to path
    """
    fileList = []
    for root, subdirectories, files in os.walk( path ):
        for file in files:
            # Only get files of the given type
            if os.path.splitext(file)[1] == extension:
                fileList.append(os.path.join(root, file))
    return fileList

# <markdowncell>

# "For each year, we sample one million beat-consecutive codewords, considering entire tracks and using a window length of 5 years"

# <codecell>

def getRandomSubsample( fileList, year, **kwargs ):
    """
    Given a year, returns nVectors of pitch/timbre/loudness descriptions from tracks in a 5-year window centered around that year.

    Input:
        fileList - list of hdf5 files to retrieve information from
        year - Year to sample from
        nVectors - the number of vectors to grab (default 1,000,000)
        seed - integer to seed the random number generator (default 0)
    Output: 
        pitchVectors - Matrix of random pitch vectors grabbed, size = (maxYear-minYear) x nVectors x 12
        timbreVectors - Same for timbre...
        loudnessValues - Vectors of random loudness values grabbed, size = (maxYear-minYear) x nVectors
        trackIndices - eg pitchVectors[year, trackIndices[year, n]:trackIndices[year, n+1]] are the pitch vectors for track n for some year
    """
    # Get keyword arguments
    nVectors = kwargs.get( 'nVectors', 1000000 )
    seed = kwargs.get( 'seed', 0 )
    np.random.seed( seed )
    # Create a copy of the file list because we'll be deleting elements from it
    fileList = list( fileList )
    # Matrix allocation
    pitchVectors = np.zeros( (nVectors, 12) )
    timbreVectors = np.zeros( (nVectors, 11) )
    loudnessValues = np.zeros( nVectors )
    # Will trim this later...
    trackIndices = np.zeros( nVectors )
    tracksWritten = 0
    # Where in the matrix are we writing values? Start from zero.
    index = 0
    # Keep going until the file list is empty (there will be other stopping conditions)
    while len( fileList ) > 0:
        # Get a random file from the remaining list
        randomIndex = np.random.randint( 0, len( fileList ))
        filename = fileList[randomIndex]
        # Remove so we don't retrieve this entry again
        del fileList[randomIndex]
        # Get hdf5 object
        h5 = tables.openFile( filename, mode='r' )
        # Get this entry's year
        trackYear = msd.hdf5_getters.get_year( h5 )
        # If the year isn't within two years, skip
        if trackYear > year + 2 or trackYear < year - 2:
            h5.close()
            continue
        # Get the data we care about
        trackPitchVectors = msd.beat_aligned_feats.get_btchromas( h5 )
        # Sometimes get_btchromas returns None
        if trackPitchVectors is None:
            continue
        else:
            trackPitchVectors = trackPitchVectors.T
        # We only want the last 11 timbre values but we don't know if get_bttimbre will fail so store in a temp variable first
        tempTimbreVectors = msd.beat_aligned_feats.get_bttimbre( h5 )
        if tempTimbreVectors is None:
            continue
        else:
            trackTimbreVectors = (tempTimbreVectors.T)[:, 1:]
        # They use the 0th timbre value as the loudness
        trackLoudnessValues = tempTimbreVectors[:, 0]
        h5.close()
        # Store values
        nBeats = trackPitchVectors.shape[0]
        storeRange = np.r_[index:np.clip( index+nBeats, 0, nVectors)]
        valuesToStore = pitchVectors[storeRange].shape[0]
        pitchVectors[storeRange] = trackPitchVectors[:valuesToStore]
        timbreVectors[storeRange] = trackTimbreVectors[:valuesToStore]
        loudnessValues[storeRange] = trackLoudnessValues[:valuesToStore]
        trackIndices[tracksWritten] = index
        tracksWritten += 1
        index += nBeats
        # Have we gotten enough vectors for all years yet?
        if (index >= nVectors):
            break
    # Trim track index list
    trackIndices = trackIndices[:tracksWritten]
    # Did we end without getting enough vectors?
    if len( fileList ) == 0:
        print "WARNING -- {0} vectors requested, only {1} found for year {2}".format( nVectors, index, year )
        pitchVectors = pitchVectors[:index]
        timbreVectors = timbreVectors[:index]
        loudnessValues = loudnessValues[:index]
    return pitchVectors, timbreVectors, loudnessValues, trackIndices

# <markdowncell>

# "We use a single threshold set to 0.5 and map the original pitch vector values to 0 or 1"
# "..we make use of a ternary, equal-frequency encoding"

# <codecell>

def quantize( matrix, quantiles ):
    """
    Quantizes to values using thresholds in quantiles

    Input:
        matrix - input matrix to quantize
        quantiles - list of quantiles
    Output: 
        quantizedMatrix - ...
    """
    # Create output matrix
    quantizedMatrix = np.zeros( matrix.shape, dtype=np.int )
    # Convert to int values...
    for quantile in quantiles:
        quantizedMatrix += matrix > quantile
    # Make matrix binary if ony one quantile was given
    if len( quantiles ) == 1:
        quantizedMatrix = np.array( quantizedMatrix, dtype=np.bool )
    return quantizedMatrix

# <markdowncell>

# "Before discretization, pitch descriptions of each track are automatically transposed to an equivalent main tonality, such that all pitch codewords are considered within the same tonal context or key. For this process we employ a circular shift strategy, correlating (shifted) per-track averages to cognitively-inspired tonal pro
# 
# les."

# <codecell>

def shiftPitchVectors( pitchVectors, trackIndices ):
    """
    Circularly shift pitch vectors to a common key.

    Input:
        pitchVectors - Matrix of pitch vectors
        trackIndices - indices of the starts of tracks in pitchVectors
    Output: 
        shiftedPitchVectors - Matrix of shifted
    """
    # These values from http://musicweb.ucsd.edu/~sdubnov/CATbox/miditoolbox/refstat.m and elsewhere
    reference = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    shiftedPitchVectors = np.zeros( pitchVectors.shape )
    # Cycle through track start and end times
    for trackStart, trackEnd in zip( trackIndices, np.append( trackIndices[1:], np.array( [pitchVectors.shape[0]] ) ) ):
        # Grab pitch vectors for this track
        trackPitchVectors = pitchVectors[trackStart:trackEnd]
        # Compute mean of pitch vectors for this track
        meanPitchVector = np.mean( trackPitchVectors, axis=0 )
        # Compute dot products for all shifts
        dotProducts = np.zeros( 12 )
        for shift in xrange( 12 ):
            dotProducts[shift] = np.dot( np.roll( meanPitchVector, shift ), reference )
        # Get best shift and store shifted vector
        bestShift = np.argmax( dotProducts )
        shiftedPitchVectors[trackStart:trackEnd] = np.roll( trackPitchVectors, bestShift, axis=1 )
    return shiftedPitchVectors

# <markdowncell>

# "Thresholds are set to the 33 and 66% quantiles of a representative sample of beat-based timbre description values."

# <codecell>

def getQuantiles( matrix, quantiles ):
    """
    Given a matrix, compute quantiles for each column

    Input:
        matrix - matrix nVectors x nVariables of values
        quantiles - the quantiles to compute
    Output: 
        quantilesPerColumn - array of size nVariables x nQuantiles indicating the quantile for each column
    """
    # Get the number of columns (variables)
    nVariables = matrix.shape[1]
    quantilesPerColumn = np.zeros( (nVariables, len( quantiles )) )
    for n in xrange( nVariables ):
        quantilesPerColumn[n] = scipy.stats.mstats.mquantiles( matrix[:, n], quantiles )
    return quantilesPerColumn

# <codecell>

# Create the data subset
if __name__ == "__main__":
    # For now just get stuff from the subset... fewer vectors possible.
    fileList = getFiles( "Data/MillionSongSubset/data", ".h5" )
    for year in np.array([1955, 1965, 1975, 1985, 1995, 2005]):
        pitchVectors, timbreVectors, loudnessValues, trackIndices = getRandomSubsample( fileList, year, nVectors=4000 )
        # Shift and quantize (simple binary threshold) the pitch vectors
        shiftedPitchVectors = shiftPitchVectors( pitchVectors, trackIndices )
        quantizedShiftedPitchVectors = quantize( shiftedPitchVectors, [.5] )
        # Save file
        np.save( 'Data/subset-pitches-' + str( year ) + '.npy', quantizedShiftedPitchVectors )
    pitchVectors, timbreVectors, loudnessValues, trackIndices = getRandomSubsample( fileList, 2005, nVectors=1000000 )
    # Shift and quantize the pitch vectors
    shiftedPitchVectors = shiftPitchVectors( pitchVectors, trackIndices )
    quantizedShiftedPitchVectors = quantize( shiftedPitchVectors, [.5] )
    # Save file
    np.save( 'Data/subset-pitches-2005-lots.npy', quantizedShiftedPitchVectors )

