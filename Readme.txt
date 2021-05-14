Overview of the Tracking module:

This tracking module evaluates individual or a series of
inline holographic images where the features of interest
consist of concentric interference rings that are visibly
separated from eachother. The background should be normalized
to 1.
The software is fully modular and separated into different
submodules that fulfil different tasks by themselves. 


The lowest level is the localization module which has demo
programs itself. It consists of lateral and axial position
determination algorithms for said inline holographic images.


The simulation module is also standalone and has its own
demo programs for showcase. It handles the creation and
fitting of inline holographic images with the Lorenz-Mie-
Theory.


The fit module is a higher level module which is a handler
module for the simulation module. It determines how the 
individual fits should be performed and how time can be 
saved.


The highest level module is the automation module which makes
use of all the lower level modules and wrappes them up into
handy functions that can be executed by the user. Full image
analysis aswell as a tracking function is included there 
that makes it easy to evaluate a whole series of images.


In this folder we also have the initializer program which
grabs all the different modules and makes them ready for usage.
Further it also reads in the parameters that are necessary 
for tuning the software aswell as inserting important theory
parameters which can vary dependent on the experimental setup.


An example of a series of synthetic images is also included
which demonstrates the power of the tracking module and its
easy usage.


As this software package is quite extensive I suggest trying
out the individual modules and reading through the demo-programs
aswell as checking out the big debug plot section to get an
insight on the functionality of this software.
