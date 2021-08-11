"""
Example file for the creation and fit of inline holographic images.
Especially interesting are the fit capabilities that can be tested here.
Single- or multi-particle fits are also possible if good enough start
values are provided.

an example with 1 particle:  creation and fit
an example with 2 particles: creation and fit

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
Ihc = Sim.InlineHoloCL(queue, ctx, tuning_parameters, theory_parameters, "full")

# do the initial setup (image size)
shape = (256, 256)
Ihc.set_shape(shape, numerical_aperture=True, mode="gpu")

# define particle positions and their properties
"""
particle = [diameter, posX, posY, posZ, refractive index, alpha_value]

Particle positions are not in pixels, they depend on the definition of one
pixel in micrometers.
"""

# define fit array, which values should have a degree of freedom in the fit
"""
array = [diameter, posX, posY, posZ, refr_idx, alpha, 3x incident wave, pol. orientation]
"""
arr = np.array([False, True, True, True, False, False, False, False, False, False], dtype=np.bool)


"""
1 particle example (creation and fit)
"""
def one_particle_example():
    # define particle parameters, single particle
    p0 = [3, 0, 0, -10, 1.57, 1]
    # calculate the image
    img0 = Ihc.calcholo([p0])+np.random.random(shape)*0.1

    # show the image
    fig = plt.figure(dpi=100)
    ax = fig.add_subplot(111)
    ax.imshow(img0, 'gray')
    plt.show()
    plt.close()

    # give parameter estimation for 1 particle fit
    p0_est = [3, 0, 0, -9, 1.57, 1]
    # execute the fit
    parfit0 = Ihc.fit_sel(img0,    # image
                          [p0_est],# parameter list
                          arr,     # fit array
                          [10],    # mask radius for that particle (in px)
                          )

    return parfit0

"""
2 particle example (creation and fit)
"""
def two_particle_example():
    # define particle paramters, two particles
    p00 = [3, -5, -5, -1.0, 1.57, 1]
    p01 = [3, +5, +5, +8.0, 1.57, 1]
    # calculate the image, particles are passed as a list [particle1, particle2]
    img1 = Ihc.calcholo([p00, p01])+np.random.random(shape)*0.1

    # show the image
    fig1 = plt.figure(dpi=100)
    ax1 = fig1.add_subplot(111)
    ax1.imshow(img1, 'gray')
    plt.show()
    plt.close()

    # give parameter estimation for 2 particle fit
    p10 = [3, -5, -5, -0, 1.57, 1]
    p11 = [3, +5, +5, +7, 1.57, 1]
    parfit1 = Ihc.fit_sel(img1, [p10, p11], arr, [10, 10])

    return parfit1

"""
try examples here
(or change parameters in the examples)
"""

one_particle_example()
two_particle_example()
