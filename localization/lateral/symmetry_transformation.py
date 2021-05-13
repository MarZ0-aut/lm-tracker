"""
This definition has been written on the ideas of 
Bowman, Richard & Preece, Daryl & Gibson, Graham & Padgett, Miles. (2011).
Stereoscopic particle tracking for 3D touch, vision and closed-loop control
in optical tweezers. Journal of Optics. 13. 044003. 10.1088/2040-8978/13/4/044003. 

theory: S(x) = integral[ p(x-x')*p(x+x') dx' ] where p(x) is a row/column
adapted to half-pixel shifts that increase accuracy and are easier implemented
"""

# import necessary modules and functions
from scipy.signal import fftconvolve

# calculate autocorrelation function for an image and sum over an axis afterwards
def sym_trafo_Bowman(img, axs):
    ST = fftconvolve(img, img, axes=axs, mode='full').sum((axs+1)%2)
    return ST