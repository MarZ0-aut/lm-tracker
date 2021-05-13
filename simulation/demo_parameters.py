tuning_parameters = {
                 # tuning parameters (calculation of inlineholographic images)
                 "size":1.10,                     # multiplied with the shape of the array to prevent fft artifacts
                 "percentage":0.5,               # percentage of the image that is used for fitting (without NA)
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

        }

theory_parameters = {
                "dx":70/542.,                     # scaling factor pixels into Î¼m
                "lamda":0.633,                    # wavelength in Î¼m
                "n_m":1.33,                       # refractive index of medium
                "n_p":1.57,                       # refractive index of particle
                "d_p":3,                          # estimated particle diameter in Î¼m
                "NA":1.2,                         # numerical aperture (NA)
                "lc":300,                         # coherence length of laser in Î¼m
                "phi":(0, True),                  # angle for incident wave
                "theta":(0, True),                # angle for incident wave
                "psi":(0, True),                  # angle for incident wave
                }