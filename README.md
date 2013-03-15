## Reproducing "Measuring the Evolution of Contemporary Western Popular Music"

This code is intended to reproduce the results presented in "Measuring the Evolution of Contemporary Western Popular Music" by Serrà et al.

To run, you need the following:

- Python 2.7 (may work with earlier or later versions of Python) http://python.org/
- Networkx http://networkx.github.com/
- Numpy http://www.numpy.org/
- Matplotlib http://matplotlib.org/
- Pytables http://www.pytables.org/moin
- Scipy http://scipy.org/
- Igraph http://igraph.sourceforge.net/
- The Million Song Dataset http://labrosa.ee.columbia.edu/millionsong/

Python and its packages can be installed easily with a package manager, for example MacPorts.  The Million Song Dataset probably needs to be obtained directly from Professor Dan Ellis; otherwise you will need to download it (it is ~300GB). 

You also need to edit paths.py such that the variables point to actual locations accessible by Python.  Specifically, you need to list where the MSD dataset lives, where the file tracks_per_year.txt lives (should be included with the MSD), and valid locations to save some files.

Once you've edited paths.py, you can run test.py to verify that everything will run correctly.  From your shell of choice, runv

python test.py

It will complain if any packages are missing or if any of the paths you specified don't exist.  This file is imported by all other files so you will not be able to run anything until test.py runs without complaining.  If tests.py runs without printing any messages (and returns an exit code 0), you are all set to go.

Once test.py runs without issue, you need to create the subsampling of the dataset they use in the paper.  To do so you can run

python createDatasets.py

This will save various .npy files into the path specified as subsamplePath in paths.py and will also save the fileList csv file according to fileListName in paths.py

Once you've created the subsampling, you can reproduce the figures in the paper by running 

python makeFigures.py

In general, I suggest running these files using IPython (get it here http://ipython.org/ or with your package manager).  I also recommend reading the comments in each file - particularly those in Markdown cells.  They describe what each part of the code is doing and trying to reproduce from the original paper.

If you have any issues, please feel free to contact me at craffel (at) (google's mail service)
