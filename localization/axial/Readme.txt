Description:

Submodule which makes use of the 2D shape of the concentric
interference fringes in an inline holographic image and
estimates the absolut value of the axial position of the
scattering particle. Background should have an average of 1.


Main class to load: "Axial"

Input a dictionary containing tuning parameters and a second
dictionary containing theory parameters to initialize the class.


Main function to execute:"estimate_z_positions"

- Input an image and a list of 2-tuples [(x1, y1), (x2, y2)]
  which define the center of the interference fringes in 2D.
- Returns a list of 4-tuples [(x1, y1, z1, val1), (x2, y2, z2, val2),...]
  which are the 3D position estimates of a scattering particle
  and a mask value to block the center of the interference pattern.


Dependencies:
- os
- datetime
- numpy
- scipy
- matplotlib
- mkl_fft
- PyAstronomy


Attention:
- Estimated axial positions appear to have the particle diameter
  subtracted! z_estimated = z_real - particle_diameter
- Function only returns absolut value of axial position with an
  experimental sign estimation!