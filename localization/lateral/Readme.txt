Description:

Submodule to determine the centers of concentric interference
fringes in inline holographic images. Background should have
an average of 1.


Main class to load: "Lateral"

Input an executor which is part of the futures package (multiprocessing)
and a dictionary of tuning parameters to initialize the class.


Main function to execute: "determine_xy_positions"

- Input an image.
- Returns list of 2-tuples [(x1, y1), (x2, y2), ...]


Dependencies:
 - os
 - datetime
 - numpy
 - scipy
 - itertools
 - matplotlib
 - concurrent.futures