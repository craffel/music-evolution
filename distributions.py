# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import test
import powerlaw
import plfit
import numpy as np
import scipy.special

# <markdowncell>

# "Specifically, we find that the distribution of codeword frequencies for a given year nicely fits to $P(z) \propto (c + z)^{-\beta}$ for $z > z_{min}$, where we take $z$ as the random variable, $\beta = 1 + 1/\alpha$  as the exponent, and $c$ as a constant"

# <codecell>

def generateLogSpacedIntegers(limit, n):
    """ 
    Generate log spaced integers from http://stackoverflow.com/questions/12418234/logarithmically-spaced-integers
    
    Input:
        limit - integer to generate numbers up to
        n - number of integers to generate
    Output: 
        logSpace - log spacing of integers
    """
    result = [1]
    if n > 1:
        ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))
    while len( result ) < n:
        next_value = result[-1]*ratio
        if next_value - result[-1] >= 1:
            # safe zone. next_value will be a different integer
            result.append(next_value)
        else:
            # problem! same integer. we need to find next_value by artificially incrementing previous value
            result.append(result[-1]+1)
            # recalculate the ratio so that the remaining values will scale correctly
            ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))
    # round, re-adjust to 0 indexing (i.e. minus 1) and return np.uint64 array
    return np.array(map(lambda x: round(x)-1, result), dtype=np.uint64)

def fitShiftedDiscretePowerLaw( data ):
    """
    Fit a shifted discrete power law distribution to some input data.
    
    Input:
        data - data to fit to
    Output:
        zmin - zmin paramter as defined in the text
        beta - ...
        c - ...
    """
    # I don't know how to estimate c explicitly so just find the one with the best KS score
    nIterations = 50
    # Store D, zmin, beta, c for each iteration
    results = np.zeros( ( nIterations, 4 ) )
    # Perform the fit for each value in a log range from 0 to 1000
    for n, c in enumerate( generateLogSpacedIntegers( 1000, nIterations ) ):
        MyPL = plfit.plfit( data + c )
        results[n] = np.array( [MyPL._ks, MyPL._xmin, 1 + 1/MyPL._alpha, c] )
    # Sort the resutls by D
    results = results[np.argsort( results[:, 0] )]
    # Return the best params
    return results[0, 1], results[0, 2], results[0, 3]

def computeShiftedDiscretePowerLaw( zmin, beta, c, z ):
    """
    Computes the power probability curve for the given parameters at the points in points.
    
    Input:
        zmin - zmin paramter as defined in the text
        beta - ...
        c - ...
        z - points to compute the curve at
    Output:
        curve - the value of the function at all points in points
    """
    # Based on equation 1 in the supplementary materials
    # y=scipy.special.zeta(x,q) returns the Riemann zeta function of two arguments: sum((k+q)**(-x),k=0..inf)
    return 1.0/(scipy.special.zeta( beta, c + zmin )*np.power( (c + z), beta ) )

