tuning_parameters = {# image processing (zeroth stage, tesselation)
                 "image_cut_percent":0.005,       # percentage of image width/height that can be cut off
                                                  # for finding adequate sub images

                # tuning parameters (first stage, finding candidates)
                 "candidate_contrast":0.025,       # minimum normalized contrast to find candidate
                 "candidate_threshold":200,       # minimum threshold to determine candidate
                 "candidate_distance":10,         # minimal distance for the peaks to be apart from eachother

                 # tuning parameters (second stage, sorting candidates)
                 "candidate_shapes":(64,128),     # side length of candidate image (small, large)
                 "candidate_decentering":7,       # value for decentering of the symmetry transform

                 # tuning parameters (third stage, 2D fitting candidates)
                 "candidate_fit_width":5,         # lower / upper boundary for 2D fit

                 # tuning parameters (fourth stage, intensity analysis)
                 "candidate_intensity_width":2,   # width of array to grab intensity from one direction (2*width+1)
                 "intensity_distance":300,        # maximum distance for intensity to allow
                 "intensity_peak_height":0.025,   # height for finding peaks in the intensity
                 "intensity_first_peakpos":5,     # brightest peak upper boundary
                 "lower_frequency_factor":0.75,   # factor to manipulate the lower frequency to match the first peak
                 "intensity_number_peaks":5,      # number of peaks to take into account
                 "min_rad_backup":10,             # minimum radius for covering center mask
                 "signal_window_slope":0.1,       # slope of smoothing function (tangent hyperbolicus)
                 "frequency_window_slope":0.5,    # slope for filter function
                 "higher_frequency":1.5,          # value for higher frequency cutoff in units of 1/µm
                 "envelope_number_peaks":10,      # nr of peaks to take into account for sign estimate
                 "envelope_chi":0.05,             # chi value that determines first estimate of sign
                 "comparison_value":0.15,          # comparison value to rule out interference localizations

                 # tuning parameters (calculation of inlineholographic images)
                 "size":1.10,                     # multiplied with the shape of the array to prevent fft artifacts
                 "percentage":0.75,               # percentage of the image that is used for fitting (without NA)
                 "divisor":60,                    # divisor used to calculate good shape
                 "propagation_epsilon":0.1,       # diameter + epsilon is reference position
                 "safety_factor":2,               # multiple of radius, defines fitting mode
                 "Nmax":40,                       # array size for calculation of scattering coefficients
                 "number_variables":10,           # d, x, y, z, n2, alpha, phi, theta, psi, prop_distance

                 # LM fit tuning
                 "tau":0.01,                      # iteration distance multiplier
                 "eps1":1e-3,                     # break epsilon 1
                 "eps2":1e-3,                     # break epsilon 2
                 "kmax":25,                       # maximum iterations
                 "steps":[0.01,
                          0.25*70/542.,
                          0.25*70/542.,
                          0.025,
                          0.01,
                          0.01,
                          0.01,
                          0.01,
                          0.01,
                          0],                     # fit steps for different parameters

                 # localization fit tuning
                 "pm_width":64,                   # maximum side length of plus minus test image
                 "comp_mode":"gpu",               # choose computation mode "cpu" or "gpu", pure gpu is mostly faster
                 "components":["X", "Y", "Z"],    # scattering portions
                 "group_distance":150,            # distance value for 2 particles to be in a group (fitted together)
                 "NA_offset":25,                  # additional pixel value that is taken into account when considering NA
                 "opencl_width":256,              # maximum side length of opencl fitted image
                 "alpha_fit":True,                # global alpha value fitted or not

                 # tuning parameter (TRACKING and UPDATING)
                 "refreshrate":8,                # index when a new "initial search" is performed again
                 "track_width":80,                # side length of cut out images for further tracking
                 "workers":1,                     # number of workers for multiprocessing
                 }

theory_parameters = {
                "dx":70/542.,                     # scaling factor pixels into micrometer
                "lamda":0.633,                    # wavelength in micrometer
                "n_m":1.33,                       # refractive index of medium
                "n_p":1.57,                       # refractive index of particle
                "d_p":3,                          # estimated particle diameter in micrometer
                "NA":1.2,                         # numerical aperture (NA)
                "lc":300,                         # coherence length of laser in micrometer
                "phi":(0, True),                  # angle for incident wave
                "theta":(0, True),                # angle for incident wave
                "psi":(0, True),                  # angle for incident wave
                }