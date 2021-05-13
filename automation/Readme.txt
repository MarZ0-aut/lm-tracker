Description:

The 'tracking' module builds up on the existing modules
'localization', 'simulation' and 'fit'. It combines them
all to create a fully automated experience, where the user
only hands over tuning and theory parameters and the image
that is to be analysed.


Main class to load: 'Tracking'

The tracking routine handles both single image localization
and multi-frame localization which functions as tracking
of preexisting particle positions in subsequent images.
Input theory and tuning parameters, the 'InlineHoloCL'
class from the 'simulation' module, the command queue
and context from pyopenCL and u are ready to go.


First function to call: 'perform_initial_search'

It acts as a first analysis of the given input image
and does a full search. Only an image has to be passed
to the function. It returns a list of particle parameters
that have been found.
OPTION 'Fast': skips the OpenCL fitting in returns
the estimated 3D positions without fit. This option
is included in the inizializer program.


Second function to call: 'track_particles'

Dependent on the first image that has been given and the
parameters received, the function 'track_particles' analyses
the following images that are put in. Taking the old positions
of found particles, it searches there for a change in 3D position.
OPTION 'Fast': skips the OpenCL fitting in returns
the estimated 3D positions without fit. This option
is included in the inizializer program.


Dependencies:
 - time
 - numpy
 
Additional debug phrases:
 - time     : displays the time elapsed
 - loc      : shows the localized particles after evaluation