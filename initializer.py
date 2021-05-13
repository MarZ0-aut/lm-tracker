# load necessary modules and functions
import pyopencl as cl
from concurrent.futures import ThreadPoolExecutor

import fit as Fit
import simulation as Sim
import automation as Aut
import localization as Loc

# define initializer class that handles all the objects that might be called
class Initialization(object):
    def __init__(self, debug_string, fast=False):
        # reinitialize the parameters
        from parameters import tuning_parameters, theory_parameters
        
        # pyopencl queue and ctx to pass to InlineHoloCL class
        platforms = cl.get_platforms()
        ctx = cl.Context(dev_type=cl.device_type.GPU,
                properties=[(cl.context_properties.PLATFORM, platforms[0])])
        queue = cl.CommandQueue(ctx)

        if debug_string == "":
            max_workers = tuning_parameters["workers"]
        else:
            max_workers = 1

        # start the executor for multithread evaluation of the first 4 stages
        executor = ThreadPoolExecutor(max_workers)

        # load classes
        self.Loc_Lat = Loc.Lateral(executor, tuning_parameters, debug_string)
        self.Loc_Axi = Loc.Axial(tuning_parameters, theory_parameters, debug_string)
        self.Sim_holo = Sim.InlineHoloCL(queue, ctx, tuning_parameters,
                                         theory_parameters, debug_string)
        self.Fit_func = Fit.OpenCLFit(tuning_parameters, theory_parameters,
                                      self.Sim_holo, debug_string)
        self.Tracker = Aut.Tracking(tuning_parameters, theory_parameters,
                                    self.Fit_func, self.Loc_Lat,
                                    self.Loc_Axi, fast, debug_string)