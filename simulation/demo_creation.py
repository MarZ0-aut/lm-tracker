"""
Demo programm for the module 'simulation'

The simulation module contains creation and fit of inline holographic
images. It is generalized to N particles and arbitrary image sizes.
It is used as a reference point for subsequent fitting in the analysis
of such images.

If the program is executed from a console, uncomment line 16+17
to ensure proper display of the plots.
"""


# import necessary modules and functions
import numpy as np
#import matplotlib
#matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import pyopencl as cl

# load the simulation module
import simulation as Sim

# load tuning and theory parameters, also edit these if wanted
from demo_parameters import tuning_parameters, theory_parameters

# pyopencl queue and ctx to pass to InlineHoloCL class
# you might need to choose a platform here dependent on your system
platforms = cl.get_platforms()
ctx = cl.Context(dev_type=cl.device_type.GPU,
        properties=[(cl.context_properties.PLATFORM, platforms[0])])
queue = cl.CommandQueue(ctx)

# initialize class, hand over necessary parameters
Ihc = Sim.InlineHoloCL(queue, ctx, tuning_parameters, theory_parameters)

# do the initial setup (image size)
shape = (1200, 1920)
Ihc.set_shape(shape, numerical_aperture=True, mode="gpu")

# define particle positions and their properties
"""
particle = [diameter, posX, posY, posZ, refractive index, alpha_value]

Particle positions are not in pixels, they depend on the definition of one
pixel in micrometers.
"""
particles = [
             [3, -40, -50, +10, 1.57, 1],
             [3, -80, -30, +25, 1.57, 1],
             [3, -30, -16, +30, 1.57, 1],
             [3, +20, +60,  +7, 1.57, 1],
             [3, +20, +30, -12, 1.57, 1],
             [3, +50, -20,  -8, 1.57, 1],
             [3, +60, +30, -27, 1.57, 1],
             [3, +80, +60, -17, 1.57, 1],
             [3, -20, +60, +13, 1.57, 1],
             [3, +10, -10, +10, 1.57, 1],
             ]

# create the image
image = Ihc.calcholo(particles)
# add noise if wanted
image += np.random.random(shape)*0.1

# save the image
np.save("example_saved.npy", image)

# plot the image
fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111)
ax.imshow(image)
plt.show()
plt.close()