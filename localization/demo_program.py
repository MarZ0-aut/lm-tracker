"""
Demo programm for the submodules 'lateral' and 'axial'

An image with size 1920x1200 containing 10 interference fringes,
made out of concentic rings, is being analyzed and the 3D position
estimates of the scattering particles that produce those interference
fringes are returned.

If the program is executed from a console, uncomment line 16+17
to ensure proper display of the plots.
"""


# import necessary modules and functions
import numpy as np
#import matplotlib
#matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

# loading the two localization modules
from lateral import Lateral
from axial import Axial

# load tuning and theory parameters, also edit these if wanted
from demo_parameters import tuning_parameters, theory_parameters

# load example image
image = np.load("demo_image.npy")

# plot image that is being analyzed
fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot(111)
ax.imshow(image)
plt.show()
plt.close()

# debug output keywords
"""
full    : shows every debug plot
save    : saves debug plot images to folder
ST      : symmetry transform of performed image and subimages
grid    : cut out subimages for subsequent analysis
search  : all possible candidates plotted in the original image
sorted  : all possible particles after sorting algorithm
2D      : 2d fitted particles

edit debug_str to achieve visualize different evaluation steps
"""

# initialize lateral localization class
Lat = Lateral(ThreadPoolExecutor(1), tuning_parameters, debug_str="2D")
# execute the main function to determine center of interference fringes
positions_2D = Lat.determine_xy_positions(image)
print(positions_2D, "\n")

# additional debug output keywords for next step
"""
filter  : showcase of the filter algorithm and reconstructed phase
phase   : instantaneous phase of possible particle with image
envelope: show the radial envelope fit
3D      : displays the image with the 3D estimates of the particles
interf  : shows interference localization debug plot

edit debug_str to achieve visualize different evaluation steps
"""

# initialize the axial localization class
Ax = Axial(tuning_parameters, theory_parameters, debug_str="3D+save")
# execute the main function to estimate the axial positions using the fringes
positions_3D = Ax.estimate_z_positions(image, positions_2D)
print(positions_3D, "\n")