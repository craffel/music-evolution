# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import test
import powerlaw
import plfit
import numpy as np
import scipy.special
import scipy.optimize

# <codecell>

def computeShiftedDiscretePowerLaw( beta, c, zmin, z ):
    """
    Computes the power probability curve for the given parameters at the points in points.
    
    Input:
        beta - beta paramter as defined in the text
        c - ...
        zmin - ...
        z - points to compute the curve at
    Output:
        curve - the value of the function at all points in points
    """
    # Based on equation 1 in the supplementary materials
    # y=scipy.special.zeta(x,q) returns the Riemann zeta function of two arguments: sum((k+q)**(-x),k=0..inf)
    return 1.0/(scipy.special.zeta( beta, c + zmin )*np.power( (c + z), beta ) )

# <markdowncell>

# "Specifically, we find that the distribution of codeword frequencies for a given year nicely fits to $P(z) \propto (c + z)^{-\beta}$ for $z > z_{min}$, where we take $z$ as the random variable, $\beta = 1 + 1/\alpha$  as the exponent, and $c$ as a constant"

# <markdowncell>

# Given $P(z | \theta) = \frac{(c+z)^{-\beta }}{\zeta[\beta ,c+z_{\text{min}}]}$ where $\theta = \{\beta, c, z_{\text{min}}\}$ we want to maximize the log likelihood over all $N_m$ samples of the random variable $z$ that we have: $\mathcal{L}(\beta, c, z_{min}) = \frac{1}{N_m} \sum_{i = 1}^{N_m} L$ where $L = \log P(z_i | \theta)$.
# 
# Taking partial derivatives, we have 
# 
# $\frac{\delta L}{\delta c} = (c+z)^{\beta } \zeta[\beta ,c+z_{\text{min}}] \left(-\frac{(c+z)^{-1-\beta } \beta }{\zeta[\beta ,c+z_{\text{min}}]}+\frac{(c+z)^{-\beta } \beta  \zeta[1+\beta ,c+z_{\text{min}}]}{\zeta[\beta ,c+z_{\text{min}}]^2}\right) = \frac{\beta  (-\zeta[\beta ,c+z_{\text{min}}]+(c+z) \zeta[1+\beta ,c+z_{\text{min}}])}{(c+z) \zeta[\beta ,c+z_{\text{min}}]}$
# 
# $\frac{\delta L}{\delta z_{\text{min}}} = \frac{\beta  \zeta[1+\beta ,c+z_{\text{min}}]}{\zeta[\beta ,c+z_{\text{min}}]}$
# 
# $\frac{\delta L}{\delta \beta} = (c+z)^{\beta } \zeta[\beta ,c+z_{\text{min}}] \left(-\frac{(c+z)^{-\beta } \log[c+z]}{\zeta[\beta ,c+z_{\text{min}}]}-\frac{(c+z)^{-\beta } \zeta^\prime[\beta ,c+z_{\text{min}}]}{\zeta[\beta ,c+z_{\text{min}}]^2}\right) = -\log[c+z]-\frac{\zeta^\prime[\beta ,c+z_{\text{min}}]}{\zeta[\beta ,c+z_{\text{min}}]}$
# 
# Note that the derivative of the Hurwitz Zeta function with respect to its first argument (here $\beta$) is not defined, so we need to use the approximation
# 
# $\zeta^\prime[ \beta, c + z_{\text{min}} ] \approx -\frac{(c + z_{\text{min}} - \frac{1}{2})^{-\beta + 1}}{\beta - 1} \left( \frac{1}{\beta - 1} + \log[c + z_{\text{min}} - \frac{1}{2}] \right)$

# <codecell>

def fitShiftedDiscretePowerLaw( z ):
    """
    Fit a shifted discrete power law distribution to some input data.
    
    Input:
        z - data to fit to
    Output:
        beta - beta paramter as defined in the text
        c - ...
        zmin - ...
    """
    # Compute the negative log likelihood of the data.  x is a vector of the parameters.
    def negativeLogLikelihood( theta, *args ):
        beta = theta[0]
        c = theta[1]
        zmin = theta[2]
        # Compute the probability for all points
        P = np.power( c + z, -beta )/scipy.special.zeta( beta, c + zmin )
        # Sum the negative log probability over all data points
        L = -np.mean( np.log( P ) )
        return L, dL( theta, *args )
    # Partial derivatives of L
    def dL( theta, *args ):
        return np.array( [dLdbeta( theta, *args ), dLdc( theta, *args ), dLdzmin( theta, *args )] )
    # Partial derivative of L with respect to c
    def dLdc( theta, *args ):
        beta = theta[0]
        c = theta[1]
        zmin = theta[2]
        zetaBCZ = scipy.special.zeta( beta, c + zmin )
        logProbabilities = beta*(-zetaBCZ + (c + z)*scipy.special.zeta( 1 + beta, c + zmin ))/((c + z)*zetaBCZ)
        return -np.mean( logProbabilities )
    # Partial derivative of L with respect to zmin
    def dLdzmin( theta, *args ):
        beta = theta[0]
        c = theta[1]
        zmin = theta[2]
        zetaBCZ = scipy.special.zeta( beta, c + zmin )
        logProbabilities = beta*scipy.special.zeta( 1 + beta, c + zmin )/scipy.special.zeta( beta, c + zmin )
        return -np.mean( logProbabilities )
    # Partial derivative of L with respect to beta
    def dLdbeta( theta, *args ):
        beta = theta[0]
        c = theta[1]
        zmin = theta[2]
        zetaPrimeApprox = -(np.power( c + zmin - .5, -beta + 1 )/(beta - 1))*(1/(beta - 1) + np.log( c + zmin - .5 ))
        logProbabilities = -np.log( c + z ) - zetaPrimeApprox/scipy.special.zeta( beta, c + zmin )
        return -np.mean( logProbabilities )
    
    # Optimize the negative log likelihood
    thetaBest, _, _ = scipy.optimize.fmin_l_bfgs_b( negativeLogLikelihood, np.array( [2.0, 70.0, 20.0] ), bounds=[(2.0, None), (0.0, None), (np.min( z ), np.max( z ))], iprint=1 )
    return thetaBest[0], thetaBest[1], thetaBest[2]

