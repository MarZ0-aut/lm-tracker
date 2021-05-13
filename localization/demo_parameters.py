tuning_parameters = {
                 # image processing (zeroth stage, tesselation)
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
                 "comparison_value":0.1,          # comparison value to rule out interference localizations
}

theory_parameters = {
                "dx":70/542.,                     # scaling factor pixels into micrometer
                "lamda":0.633,                    # wavelength in micrometer
                "n_m":1.33,                       # refractive index of medium
                "d_p":3,                          # estimated particle diameter in micrometer
                }