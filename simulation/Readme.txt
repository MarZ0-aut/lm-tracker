Description:

Simulation module to create and fit inline holographic images.
The folder 'algorithms' holds all the necessary programs and
kernels that are used to simulated such images.


Main class to load: 'InlineHoloCL'

Input command queue and context from opencl module. Also
hand over tuning and theory parameters that are used for
creation and fit.


First function to call: 'set_shape'

This function is used after initialization of the class
and handles the arrays that are needed in the further
creation and fit of the images.
Input a 2-tuple which is the image size in pixels. Here
you can also define if some corrections (like NA) are used
or not; also computation mode "gpu" or "cpu" can be chosen
here.


Function to create images: 'calcholo'

Input a list of particle parameters [particle1, particle2, ...]
whereas a particle consist of (atleast) a 6-tuple of values
that are handed over as a list.
[diameter, posX, posY, posZ, refr_idx, alpha_value]


Function to fit images with solution estimates: 'fit_sel'

First define a (boolean) fit array which defines what parameters
are degrees of freedom in the fit. Then, input the image to be
fitted, the list of particle estimates, the fit array, a list of
mask values and choose if u want the output plots or not.

--> Looking through the demo programs and playing with the values
    can help a lot in understanding how the simulation package works.


Dependencies:
- os
- datetime
- numpy
- scipy
- matplotlib

- pyopencl
- mkl_fft
- gpyfft