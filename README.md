## Reproducing "Measuring the Evolution of Contemporary Western Popular Music"

This code is intended to reproduce the results presented in "Measuring the
Evolution of Contemporary Western Popular Music" by Serrà et al.

To run, you need the following:

- Python 2.7 (may work with earlier or later versions of Python)
  http://python.org/
- Numpy http://www.numpy.org/
- Matplotlib http://matplotlib.org/
- Pytables http://www.pytables.org/moin
- Scipy http://scipy.org/
- Igraph http://igraph.sourceforge.net/
- The Million Song Dataset http://labrosa.ee.columbia.edu/millionsong/

Python and its packages can be installed easily with a package manager, for
example MacPorts.  The Million Song Dataset probably needs to be obtained
directly from Professor Dan Ellis; otherwise you will need to download it (it is
~300GB). 

You also need to edit paths.py such that the variables point to actual locations
accessible by Python.  Specifically, you need to list where the MSD dataset
lives, where the file tracks_per_year.txt lives (should be included with the
MSD), and valid locations to save some files.

Once you've edited paths.py, you can run test.py to verify that everything will
run correctly.  From your shell of choice, run

python test.py

It will complain if any packages are missing or if any of the paths you
specified don't exist.  This file is imported by all other files so you will not
be able to run anything until test.py runs without complaining.  If tests.py
prints "All tests passed OK!" (and returns an exit code 0), you're all set.

Once test.py runs without issue, you need to create the subsampling of the
dataset they use in the paper (this will take a very very long time).  To do so
you need to run

python createDatasets.py 0
python createDatasets.py 1
python createDatasets.py 2
...
python createDatasets.py 9

This will save various .npy files into the path specified as subsamplePath in
paths.py and will also save the fileList csv file according to fileListName in
paths.py.  The number at the end of each line is the random seed used to
generate the random sampling; in the experiment the random sampling is done 10
times.  Each sampling is split up like this so you can cheaply parallelize by
running each command in a separate shell concurrently.  Alternatively, if you
would like to only have to enter one command and have it all run sequentially
you can run

for i in {0..9}; do python createDatasets.py $i; done;

Once the subsampled datasets have all been saved, you need to create the
.graphml files for the networks (this speeds up loading the graphs a lot).  As
above, this needs to be done for each seed from 0 to 9:

python networkAnalysis.py 0
python networkAnalysis.py 1
python networkAnalysis.py 2
...
python networkAnalysis.py 9

Again, you can run these commands in separate shell instances for cheap
parallelization or you can run them sequentially with the command

for i in {0..9}; do python networkAnalysis.py $i; done;

Once you've created the subsampling and the .graphml files, you can reproduce
the figures in the paper by running 

python makeFigures.py

Some of the figures may take a very very very long time to appear (particularly
the first time you run).  I recommend letting it run overnight.

In general, I suggest running these files using IPython (get it here
http://ipython.org/ or with your package manager).  I also recommend reading the
comments in each file - particularly those in Markdown cells.  They describe
what each part of the code is doing and trying to reproduce from the original
paper.

If you have any issues, please feel free to contact me at craffel (at) (google's
mail service)
