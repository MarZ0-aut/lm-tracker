"""
Demo programm for the module 'automation'

A series of 16 images with sidelengths 400px is being analyzed.
Each image consists of two features (location of interest with
interference fringes). The position of the features varies
throughout the series. It is a demonstration of how particles
are tracked using the given modules in this software package.

If the program is executed from a console, uncomment line 16+17
to ensure proper display of the plots.
"""

# load necessary modules and functions
from numpy import load
#import matplotlib
#matplotlib.use('qt5agg')
from initializer import Initialization

# define debug keyword
debug_top ="loc"
"""
For more keywords check the individual Readme.txt files from
the individual modules
"""

# initialize all modules
modules = Initialization(debug_top, fast=True)
# load the tracking module
Tracking = modules.Tracker

# load images to be analyzed (particles to be tracked)
images = load("example_series.npy")


# perform an initial search on the first image of the series
result = Tracking.perform_initial_search(images[0], output="um")
#print(result)
# the following images are evaluated using the track_particles routine
for image in images[1:]:
    result = Tracking.track_particles(image, output="um")
    #print(result)

"""
Also checkout the 'parameters' file to tune the software as there are
a lot of different values that influence the work of all the modules.
"""