Description:

The module 'fit' builds up on the existing modules
'localization' and 'simulation' as it handles some
of the decisions the user would have to do by himself.


Main class to load: 'OpenCLFit'

Input tuning and theory parameters, the "InlineHoloCL"
class from the 'simulation' package, the image to
be analyzed and a debug_string that defines the visual
output.


First function to load: 'distinguish_pm_particles'

This function takes a list of 5-tuples which are
essentially particle parameters [particle1, particle2, ..]
with a particle being [posX, posY, posZ, diameter, mask_value].
It takes the 'simulation' package to create synthetic images
that are used for comparison to determine the sign of the
axial position.


Second function to load: 'perform_opencl_fit'

Basically a fit-handler for the 'simulation' module.
After handing over the particle estimates and choosing if
the diameter should be a degree of freedom in the fit or not
it determines by itself if particles are fitted alone
or together.


Dependencies:
- os
- datetime
- numpy
- matplotlib


Additional debug phrases:
- pm      : plus, minus determination
- fit     : opencl fit, combined image of original and fitted
- resid   : plot of the residual image