"""
--> The following code for calculating and evaluating holographic images
    in OpenCL is based on the ideas and developement of GRIER D.

--> InlineHoloCL class received from THALHAMMER G.

CHANGES / ADDITIONAL FUNCTIONALITY by ZOBERNIG M.:

 + outsourcing of the kernel file into separat file for a better overview
    -> "calcholo.cl"

 + introduction of multiple calculation variants and outputs
     -> "cpu" and "gpu" for general calculation
     -> "fields" for retrieval of individual field components

 + introduction of the numerical aperture correction on CPU and GPU
     -> 2D fourier transforms and shape correction of the arrays

 + introduction of a dynamic change in input shape to prevent artifacts
     in calculation of fourier transforms for the numerical aperture
     -> can increase the speed of the FFT and gpyfft
     -> works similar as padding

 + introduction of a scattering coefficient calculation logic that
     determines wheter it is necessary to calculate them new or not

 + introduction of a simple model for the coherence of the light source

 + introduction of a fit mask that hides center of the particle to prevent
     errors in the fit routine

 + generalization of the fit algorithm to N particles

 + introduction of a "partial" fit that does not consider all 100% of the
     image pixels
     -> can accellerate the fit

 + recycling of several arrays for further calculations

 + creation of necessary calculation arrays at the beginning when they are
     needed to prevent multiple creations that take time

 + introduction of flexibility for the calculation of the individual
     components of the electric field

 + definition of a propagation mechanism that closes the theory gap for positions
     between -d/2 and d/2 for a particle with diameter d
"""

# import of necessary modules and functions
import os
import numpy as np
import pyopencl as cl
import pyopencl.array as cla

from mkl_fft import fft2, ifft2
from numpy.fft import fftfreq
from gpyfft.fft import FFT

from .lm_routine import LM
from .miecoeff_calc import mie_f
from .debug_plots_sim import Debugging_Simulation

# InlineHoloCL class
class InlineHoloCL(object):

    # ---------- tuning parameters, received from initialization ---------- #

    # init definition
    def __init__(self, queue, ctx, tuning, params, debug_str):
        # main definitions
        self.queue = queue
        self.ctx = ctx
        self._debug = Debugging_Simulation(params)

        if "opencl" in debug_str or "full" in debug_str or "save" in debug_str:
            self.debug = True
        else:
            self.debug = False

        # theory parameters
        self.dx = params["dx"]
        self.lamda = params["lamda"]
        self.n_medium = params["n_m"]
        self.NA = params["NA"]
        self.lc = params["lc"]

        # calculation parameters
        self.Nmax = tuning["Nmax"]
        self.size = tuning["size"]
        self.percentage = tuning["percentage"]
        # backup because there cannot be more than 100%
        if self.percentage > 1:
            self.percentage = 1
        self.divisor = tuning["divisor"]

        # propagation parameters
        self.epsilon = tuning["propagation_epsilon"]
        self.safety_factor = tuning["safety_factor"]

        # load kernel files
        current_path = os.getcwd().split("\\")[-1]
        if current_path == "simulation":
            path_calcholo = './algorithms/kernels/calcholo.cl'
            path_multiply = './algorithms/kernels/multiply.cl'
        else:
            path_calcholo = './simulation/algorithms/kernels/calcholo.cl'
            path_multiply = './simulation/algorithms/kernels/multiply.cl'

        self.calcholo_kernel = cl.Program(self.ctx, open(path_calcholo).read()).build()
        self.multiply_kernel = cl.Program(self.ctx, open(path_multiply).read()).build()

        # wave number
        self.k = (2*np.pi*self.n_medium)/self.lamda

        # number of parameters
        self.number_params = tuning["number_variables"]

        # LM fit tuning
        self.tau = tuning["tau"]
        self.eps1 = tuning["eps1"]
        self.eps2 = tuning["eps2"]
        self.kmax = tuning["kmax"]
        self.step_backup = tuning["steps"]

    # ---------- preparation functions after shape is passed ---------- #

    # define radial coordinates to create aperture mask
    def _create_radial_coords(self, factor=1):
        ky = fftfreq(self.ny, factor*self.dx)
        kx = fftfreq(self.nx, factor*self.dx)
        Kx, Ky = np.meshgrid(kx, ky, sparse=True)
        K2 = Kx*Kx + Ky*Ky
        return K2

    # aperture in fourier plane (acts as hard mask)
    def _create_aperture(self):
        R = (self.NA/self.n_medium)/self.lamda
        radius = self._create_radial_coords(1)

        # critical radius for aperture
        self.Aperture = radius <= R*R
        # create integer aperture for graphics card
        if self.mode == "gpu":
            # move to graphics card
            self.Aperture_gpu = cla.to_device(self.queue, np.array(self.Aperture, dtype=np.int32))

    # initially set shape because of different image sizes (aperture dependence)
    def set_shape(self, shape, numerical_aperture=False, coherence=False,
                  mode="gpu"):
        # delete existing propagator and aperture
        if hasattr(self, 'K2'):
            del(self.K2)
        if hasattr(self, 'Aperture'):
            del(self.Aperture)

        self.coherence = coherence
        self.numerical_aperture = numerical_aperture
        self.ny0, self.nx0 = shape

        self.mode = mode
        # backup logic
        if not (mode == "cpu" or mode == "gpu"):
            print("This computation mode does not exist. Switch to 'gpu'.")
            self.mode = "gpu"

        # recalculate nx and ny to ensure a size that fits the FFT (60 is a good divisor) 2*2*3*5
        self.ny = int(round((self.ny0*self.size)/self.divisor)*self.divisor)
        self.nx = int(round((self.nx0*self.size)/self.divisor)*self.divisor)
        # rounding operation can cause to result a smaller shape than input
        if self.ny < self.ny0:
            self.ny += self.divisor
        if self.nx < self.nx0:
            self.nx += self.divisor

        valy = (self.ny-self.ny0)/2
        valx = (self.nx-self.nx0)/2
        self.cuty0, self.cuty1 = int(np.floor(valy)), int(np.ceil(valy))
        self.cutx0, self.cutx1 = int(np.floor(valx)), int(np.ceil(valx))
        self.shape = (self.ny, self.nx)

        if self.numerical_aperture == True:
            # create aperture
            self._create_aperture()

        # initialize scattered field array and intensity array
        self.EX = cla.zeros(self.queue, (self.ny, self.nx), np.complex64)
        self.EY = cla.zeros(self.queue, (self.ny, self.nx), np.complex64)
        self.EZ = cla.zeros(self.queue, (self.ny, self.nx), np.complex64)
        self.I_cl = cla.zeros(self.queue, (self.ny, self.nx), np.float32)

        # angle 4 vectors
        self.t_cl = cla.zeros(self.queue, 1, cla.vec.float4)
        self.T = np.zeros(1, dtype=cla.vec.float4)

    # ---------- scattering coefficients ---------- #

    # create scattering arrays
    def _create_arrays(self, N, propagation):
        shape = (N, self.Nmax+1)

        if propagation == False:
            self.A0 = np.zeros(shape, dtype=np.complex64)
            self.B0 = np.zeros(shape, dtype=np.complex64)
            self.P0 = np.zeros(N, dtype=cla.vec.float4)

            self.a_cl0 = cla.zeros(self.queue, shape, np.complex64)
            self.b_cl0 = cla.zeros(self.queue, shape, np.complex64)
            self.p_cl0 = cla.zeros(self.queue, N, cla.vec.float4)
        else:
            self.A1 = np.zeros(shape, dtype=np.complex64)
            self.B1 = np.zeros(shape, dtype=np.complex64)
            self.P1 = np.zeros(N, dtype=cla.vec.float4)

            self.a_cl1 = cla.zeros(self.queue, shape, np.complex64)
            self.b_cl1 = cla.zeros(self.queue, shape, np.complex64)
            self.p_cl1 = cla.zeros(self.queue, N, cla.vec.float4)

    # parameter handler function, moves parameters to graphics card
    def _calc_coefficients(self, parameters, N, calc_mie, reset, propagation):
        # check for propagation or not
        if propagation == False:
            A_calc, B_calc, P_calc = self.A0, self.B0, self.P0
            a_calc, b_calc, p_calc = self.a_cl0, self.b_cl0, self.p_cl0
        else:
            A_calc, B_calc, P_calc = self.A1, self.B1, self.P1
            a_calc, b_calc, p_calc = self.a_cl1, self.b_cl1, self.p_cl1

        # if radius is changed, recalcualte mie scattering coefficients
        # keep the previous ones for restoration
        if reset == True:
            A, B = np.copy(A_calc), np.copy(B_calc)

        for i in range(N):
            d, x, y, z, n2, alpha, phi, theta, psi, prop = parameters[i]
            r = d/2
            if z<0:
                z = -z
                n2 = self.n_medium*(self.n_medium/n2)

            if calc_mie == True:
                #a, b = self._mie_f(r, n2=n2)
                a, b = mie_f(r, n2, self.lamda, self.n_medium, self.Nmax)
                A_calc[i] = a
                B_calc[i] = b
            P_calc[i] = x, y, z, alpha
        # angles for all particles are the same
        self.T[0] = phi, theta, psi, 0

        # move to graphics card
        a_calc.set(A_calc)
        b_calc.set(B_calc)
        p_calc.set(P_calc)
        self.t_cl.set(self.T)

        # reset with kept arrays
        if reset == True:
            A_calc, B_calc = A, B

    # ---------- propagation preparation ---------- #

    # create propagator radius
    def _prepare_radius(self):
        denominator = 1 - self.K2
        self.prop_mask = denominator >= 0
        self.Rad = np.sqrt(denominator*self.prop_mask)

    # prepare variables for propagator
    def _prepare_propagator(self, distance):
        # check if arrays already exist
        if not hasattr(self, 'K2'):
            self.K2 = self._create_radial_coords(2)
            self._prepare_radius()
        if not hasattr(self, 'Aperture'):
            self._create_aperture()

        # create phase shift
        self.shift = np.exp((1j*self.k*distance)*self.Rad)*self.prop_mask
        # multiply aperture because it is necessary
        self.shift *= self.Aperture
        # gpu phase shift
        if self.mode == "gpu":
            self.shift_gpu = cla.to_device(self.queue,
                            np.array(self.shift, dtype=np.complex64))

    # ---------- CPU correction function ---------- #

    # function to correct fields / intensities on CPU
    def _cpu_correction(self, array, fourier_trafo, intensity,
                        aperture=False, propagation=False):

        # FFT correction for NA correction or propagation
        if fourier_trafo == True:
            fourier = fft2(array, axes=(1, 0))

            # if propagation is necessary
            if propagation == True:
                fourier *= self.shift
            # if only NA needs to be corrected
            if ((aperture == True) or (propagation == True)):
                fourier *= self.Aperture

            corrected = ifft2(fourier, axes=(1, 0))
        else:
            corrected = array

        # check if slicing is necessary
        if self.cuty0 != 0:
            corrected = corrected[self.cuty0:, :]
        if self.cuty1 != 0:
            corrected = corrected[:-self.cuty1, :]
        if self.cutx0 != 0:
            corrected = corrected[:, self.cutx0:]
        if self.cutx1 != 0:
            corrected = corrected[:, :-self.cutx1]

        # if it is not an intensity, calculate it
        if intensity == False:
            return (np.conj(corrected)*corrected).real
        else:
            return corrected

    # ---------- GPU correction functions ---------- #

    # create FFT plan
    def _create_plan(self):
        # sometimes creation of plan can fail
        try:
            self.transform = FFT(self.ctx, self.queue, self.EX, axes=(0, 1))
        # if plan creation failes (because of array shape) switch to cpu calculation
        except:
            self.mode = "cpu"

    # fourier transform function
    def _gpu_fft(self, field, forward):
        event, = self.transform.enqueue_arrays(data=field, forward=forward)
        event.wait()

    # function to correct fields / intensities on GPU
    def _gpu_correction(self, field, aperture, propagation):
        # run the fourier transform
        self._gpu_fft(field, forward=True)

        # propagate, aperture is already included
        if propagation == True:
            self.multiply_kernel.c_mult(self.queue, self.shape, None,
                                      field.data, self.shift_gpu.data)
        # multiply only with aperture
        elif aperture == True:
            self.multiply_kernel.i_mult(self.queue, self.shape, None,
                                      field.data, self.Aperture_gpu.data)

        # perform the back fourier transform
        self._gpu_fft(field, forward=False)

    # add to intensity
    def _gpu_calc_intensity(self, field):
        self.multiply_kernel.re_mult(self.queue, self.shape, None,
                                  self.I_cl.data, field.data)

    # ---------- calculation of inline holographic images ---------- #

    # check if incoming parameter list is a list of lists
    def _check_params(self, params):
        # set numpy array if not already
        par_arr = np.array(params)
        # retreive the shape
        shape = par_arr.shape
        # correct into list of lists if necessary
        if len(shape) == 1:
            par_arr =  np.array([par_arr])
        # check if parameters don't contain angles or propagation distance
        newx = par_arr.shape[1]
        par_arr = np.pad(par_arr, ((0, 0), (0, self.number_params-newx)), 'constant')
        return par_arr

    # parameter handler for calcholo that checks for propagation
    def calcholo(self, parameters, calc_mie=True, reset=False, components=["X", "Y", "Z"]):
        if not hasattr(self, "count"):
            self.count = 0
        # check parameters
        parameters  = self._check_params(parameters)

        need_propagation = []
        no_propagation = []
        for particle in parameters:
            # retreive parameters
            d, z = particle[0], particle[3]
            # set new values that are going to be corrected internally
            if abs(z) < d/2:
                particle[3] = -(d/2+self.epsilon)
                # give attention to the scattered wave that needs the diameter
                particle[-1] = -(z+d/2+self.epsilon)
                need_propagation.append(particle)
            else:
                no_propagation.append(particle)
        need_propagation = np.array(need_propagation)
        no_propagation = np.array(no_propagation)

        N = len(no_propagation)
        M = len(need_propagation)
        # no propagation
        if N != 0:
            if not hasattr(self, "p_cl0") or self.count == 0:
                self._create_arrays(N, False)
            if len(self.p_cl0) != N:
                self._create_arrays(N, False)
            self._calc_coefficients(no_propagation, N, calc_mie, reset, False)
        # propagation
        if M != 0:
            if not hasattr(self, "p_cl1") or self.count == 0:
                self._create_arrays(M, True)
            if len(self.p_cl1) != M:
                self._create_arrays(M, True)
            self._calc_coefficients(need_propagation, M, calc_mie, reset, True)

        # normal particles go together
        Int = self._calcholo_full(no_propagation, components, False)
        # propagated particles need to be calculated separately
        for index in range(M):
            Int += self._calcholo_full([need_propagation[index]], components, True, index)
        return Int

    # function for multiple particles / positions, full correction if wanted
    def _calcholo_full(self, parameters, components, propagation, index=0):
        # if no particle preset, return empty array
        if len(parameters) == 0:
            return self._cpu_correction(np.zeros(self.shape), False, True)

        # full write into memory
        if ((self.numerical_aperture == True) or (propagation == True)):
            full = 1
        else:
            full = 0

        # check for correct arrays
        if propagation == False:
            a_calc, b_calc, p_calc = self.a_cl0, self.b_cl0, self.p_cl0
        else:
            a_calc, b_calc, p_calc = self.a_cl1, self.b_cl1, self.p_cl1

        # perform openCL calculation and add up scattered fields
        self.calcholo_kernel.cl_holo(self.queue, self.shape, None,
              np.int32(self.Nmax), np.int32(len(parameters)), np.int32(index),
              np.float32(self.dx), np.float32(self.lamda), np.float32(self.n_medium),
              np.float32(self.lc), np.int32(int(self.coherence)), np.int32(full),
              self.t_cl.data, p_calc.data, a_calc.data, b_calc.data,
              self.EX.data, self.EY.data, self.EZ.data, self.I_cl.data)

        # modify fields if necessary
        if ((self.numerical_aperture == True) or (propagation == True)):
            # create propagator if necessary
            if propagation == True:
                self._prepare_propagator(parameters[0][-1])

            # plain CPU correction
            if self.mode == "cpu":
                Int = self._cpu_correction(self.I_cl.get(), False, True)
                # extract field components, correct them and add them up
                if "X" in components:
                    Int += self._cpu_correction(self.EX.get(), True, False,
                                self.numerical_aperture, propagation)
                if "Y" in components:
                    Int += self._cpu_correction(self.EY.get(), True, False,
                                self.numerical_aperture, propagation)
                if "Z" in components:
                    Int += self._cpu_correction(self.EZ.get(), True, False,
                                self.numerical_aperture, propagation)

            # GPU intensity calculation, CPU for shape correction at the end
            elif self.mode == "gpu":
                # create a plan if necessary
                if not hasattr(self, 'transform') or self.count == 0:
                    self._create_plan()
                # correct aperture and add to intensity
                if "X" in components:
                    self._gpu_correction(self.EX,
                                         self.numerical_aperture, propagation)
                    self._gpu_calc_intensity(self.EX)
                if "Y" in components:
                    self._gpu_correction(self.EY,
                                         self.numerical_aperture, propagation)
                    self._gpu_calc_intensity(self.EY)
                if "Z" in components:
                    self._gpu_correction(self.EZ,
                                         self.numerical_aperture, propagation)
                    self._gpu_calc_intensity(self.EZ)
                # retreive intensity and fit the shape if necessary
                Int = self._cpu_correction(self.I_cl.get(), False, True)

        # retreive intensity if no NA correction or propagation is set
        else:
            Int = self._cpu_correction(self.I_cl.get(), False, True)
        return Int

    # function for multiple particles / positions, almost no correction - fast
    def _calcholo_percentage(self, parameters, calc_mie, reset):
        # check parameters
        parameters = self._check_params(parameters)
        N = len(parameters)

        # check if arrays need to be calculated
        if not hasattr(self, "p_cl0") or self.count == 0:
            self._create_arrays(N, False)
        if len(self.p_cl0) != N:
            self._create_arrays(N, False)
        # set the coefficients and parameters
        self._calc_coefficients(parameters, N, calc_mie, reset, False)

        # perform openCL calculation and add up scattered fields
        self.calcholo_kernel.cl_holo_percentage(self.queue, self.sel_size, None,
              np.int32(self.Nmax), np.int32(N), np.int32(0), np.float32(self.dx),
              np.float32(self.lamda), np.float32(self.n_medium), np.float32(self.lc),
              np.int32(int(self.coherence)), np.int32(0),
              self.t_cl.data, self.p_cl0.data, self.a_cl0.data, self.b_cl0.data,
              self.EX.data, self.EY.data, self.EZ.data, self.I_cl.data,
              self.index_cl[0].data, self.index_cl[1].data,
              np.float32(self.ny), np.float32(self.nx))

        # retreive Intensity
        Icl = self.I_cl.get()
        return Icl

    # ---------- fit function routines ---------- #

    # calcholo fit handler function
    def _f_sel(self, parameters, calc_mie, reset, components, check):
        # calculate a hologram
        if ((self.numerical_aperture == True) or (check == True)):
            # calculate corrected hologram
            Int = self.calcholo(parameters, calc_mie, reset, components)
            # apply the fit mask, same mask as for pixelated calculation
            Int *= self.mask
            return Int.ravel()
        else:
            # without the numerical aperture, an image with way less pixels is calculated
            Int = self._calcholo_percentage(parameters, calc_mie, reset)
            return Int

    # fit images calculation function, calculates image and its derivatives
    def _fJ_sel(self, pars_vary, pars_all, vary, img, components, check):
        # load parameters and define fit space
        pars = pars_all.copy()
        len_p = len(pars)
        pars[vary] = pars_vary
        vary_idx = vary[0].nonzero()[0]
        len_v = len(vary_idx)

        # debug print
        if self.debug == True:
            for i in range(len_p):
                d, x, y, z, n2, alpha, phi, theta, psi, prop_dist = pars[i]
                print("p%1.0f x:%6.2f y:%6.2f z:%6.2f n2:%6.2f alpha:%6.2f d:%6.2f"
                      %(i, x, y, z, n2, alpha, d))
            print("\tphi:%6.2f theta:%6.2f psi:%6.2f"%(phi, theta, psi))
            if not len_p == 1:
                print()

        # calculate initial hologram
        holo = self._f_sel(pars, True, False, components, check)
        self.count += 1

        # calculate individual derivatives
        for i in range(len_p):
            for k, idx in enumerate(vary_idx):
                # change parameter
                pp = pars.copy()
                # ensure that laser angles remain the same for all particles
                if idx > 5:
                    pp[:, idx] += self.step[idx]
                else:
                    pp[i][idx] += self.step[idx]
                # calculate gradient image
                holo2 = self._f_sel(pp, idx==0, idx==0, components, check)
                holo2 -= holo
                holo2 *= 1./self.step[idx]
                # save image
                self.J[(len_v*i)+k] = holo2

        holo -= img
        return holo, self.J

    # main fit function
    def fit_sel(self, img0, parameters, vary, rads, components=["X", "Y", "Z"]):
        # retreive parameters
        parameters = self._check_params(parameters)
        pars0 = np.array(parameters, dtype=np.float32)
        N = len(pars0)
        # blow up fit parameter mask
        vary_all = np.array([vary for i in range(N)])
        pars_vary0 = pars0[vary_all]
        ny, nx = self.ny0, self.nx0

        # create central fitting mask
        self.mask = np.ones((ny, nx), dtype=np.bool)
        for i in range(len(rads)):
            px = parameters[i][1]/self.dx+0.5
            py = parameters[i][2]/self.dx+0.5
            x = np.linspace(-nx/2-px, nx/2-px, nx)
            y = np.linspace(-ny/2-py, ny/2-py, ny)
            X, Y = np.meshgrid(x, y, sparse=True)
            R2 = X*X + Y*Y
            M = R2 >= rads[i]*rads[i]
            self.mask *= M

        # check if propagation is necessary
        check = 0
        for par in pars0:
            d, z = par[0], par[3]
            if abs(z) < (d/2)*self.safety_factor:
                check += 1

        # create additional random mask
        mask = np.ones(self.nx0*self.ny0, dtype=np.bool)
        mask[:int(self.nx0*self.ny0*(1-self.percentage))] = False
        np.random.shuffle(mask)
        mask = mask.reshape((self.ny0, self.nx0))
        self.mask *= mask

        if self.numerical_aperture == True or check > 0:
            # masked fit image
            img1 = (img0*self.mask).ravel()
        else:
            # create indices
            selr, selc = np.where(mask)
            self.index = (selr.astype(np.uint32), selc.astype(np.uint32))
            self.sel_size = (selr.shape[0],)
            # masked fit image
            img1 = img0[self.index]
            # create arrays on graphics card
            self.I_cl = cla.zeros(self.queue, self.sel_size, np.float32)
            self.index_cl = (cla.to_device(self.queue, self.index[0]),
                             cla.to_device(self.queue, self.index[1]))

        # debug plot for fitting
        if self.debug == True:
            self._debug.debug_fit(img0, self.mask)

        # define steps and create reduction matrix J
        self.step = self.step_backup
        self.J = np.zeros( shape = pars_vary0.shape + img1.shape, dtype = np.float32)

        # run actual fit routine
        self.count = 0
        parfit = LM(self._fJ_sel,
                               pars_vary0,
                               (pars0, vary_all, img1, components, check>0),
                               tau=self.tau,
                         eps1 = self.eps1,
                         eps2 = self.eps2,
                         kmax=self.kmax,
                         verbose=False)

        parfit_all = pars0.copy()
        parfit_all[vary_all] = parfit

        if self.debug == True:
            for i in range(N):
                print("p"+str(i)+": ",np.round(parfit_all[i][1:4], 3))
        return parfit_all