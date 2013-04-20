# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# This should test that all of the paths are set correctly and necessary modules are installed.

# <codecell>

# Test for modules needed
try:
    import os
    import csv
    import collections
    import random
    import sys
    import glob
    import tempfile
except:
    print "Failed to import base Python modules - is there something wrong with your Python install?  Are you using Python 2.7?"
    sys.exit(1)
try:
    import numpy as np
except:
    print "numpy not installed.  Please install from http://www.numpy.org/"
    sys.exit(1)
try:
    import matplotlib.pyplot as plt
except:
    print "matplotlib not installed.  Please install from http://matplotlib.org/"
    sys.exit(1)
try:
    import msd
except:
    print "msd not found - this should have been part of the package you downloaded.  Did you mess with the directory structure?"
    sys.exit(1)
try:
    import tables
except:
    print "pytables not installed.  Please install from http://www.pytables.org/moin"
    sys.exit(1)
try:
    import scipy.stats.mstats
except:
    print "scipy(.stats) not found.  Please install from http://scipy.org/"
    sys.exit(1)
try:
    import igraph
except:
    print "igraph note found.  Please install from http://igraph.sourceforge.net/"
    sys.exit(1)
try:
    import paths
except:
    print "Couldn't find paths.py, which should have been part of the package you downloaded.  Did you mess with the directory structure?"
    sys.exit(1)

# <codecell>

# Test that paths.py was setup correctly
if not os.path.exists( paths.msdPath ):
    print "The msdPath variable you specified in paths.py doesn't exist.  Please create it or edit paths.py and try again."
    sys.exit(1)
if not os.path.exists( paths.subsamplePath ):
    print "The subsamplePath variable you specified in paths.py doesn't exist.  Please create it or edit paths.py and try again."
    sys.exit(1)
if not os.path.exists( os.path.split( paths.fileListName )[0] ):
    print "The directory of the fileListName you specified in paths.py doesn't exist, so the file will not be able to be created.  Please create it or edit paths.py and try again."
    sys.exit(1)
if not os.path.exists( os.path.split( paths.yearToFileMappingName )[0] ):
    print "The directory of the yearToFileMappingName you specified in paths.py doesn't exist, so the file will not be able to be created.  Please create it or edit paths.py and try again."
    sys.exit(1)

# <codecell>

# Simple way to actually print a diagnostic message if the script runs OK
if __name__ == "__main__":
    print "All tests passed OK!"

