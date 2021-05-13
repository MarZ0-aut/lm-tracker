# import necessary modules and functions
import time as t
import numpy as np

# Tracking routine for inline holographic images
class Tracking(object):

    # ---------- tuning parameters ---------- #

    # init definition
    def __init__(self, tuning, theory, FIT, LAT, AXI, fast=False, debug=''):
        # tuning parameter (TRACKING and UPDATING)
        self._track_width = tuning["track_width"]
        self._refreshrate = tuning["refreshrate"]
        self._dx = theory["dx"]
        self._debug_str = debug
        self._fast = fast

        # refresh rate has to be minimum 2
        if self._refreshrate < 2:
            print('Refreshrate has to be atleast 2.')
            self._refreshrate = 2

        # set classes
        self._FIT = FIT
        self._LAT = LAT
        self._AXI = AXI

   # ---------- helper functions ---------- # 

    # recalculate into µm
    def _transform_into_um(self, particles, img):
        um_particles = np.copy(particles)
        ly, lx = img.shape
        for i in range(len(um_particles)):
            um_particles[i][0] -= lx/2
            um_particles[i][1] -= ly/2
            um_particles[i][0] *= self._dx
            um_particles[i][1] *= self._dx
        return np.round(um_particles, 2)

    # check input image
    def _check_image(self, img):
        # retreive shape
        ny = len(img)
        if ny >= 1:
            nx = len(img[0])
        if nx >= 1:
            self.image = img
            return 1
        else:
            return 0

    # search in a smaller image
    def _small_search(self, img, part):
        # perform a search at the old position
        new_candidates = self._LAT._find_possible_candidates(img, part)
        new_candidates = np.array(np.round(new_candidates, 0), dtype=np.int)
        # sort the possible candidates
        sorted_candidates, candidate_images, STs = self._LAT._sort_out_candidates(img, new_candidates)
        # fit the remaining candidates = determine them as particles
        new_particle = self._LAT._estimate_2D_positions(candidate_images, sorted_candidates, STs)
        return new_particle

    # ---------- tracking and updating routine ---------- #
    
    # track the already found particles
    def _track_particles(self):
        tracked_particles = []
        for particle in self.fitted_particles:
            t1 = t.time()
            # restrict image to position, only small image around center
            fpos_x = int(round(particle[0], 0))
            fpos_y = int(round(particle[1], 0))
            wid0, wid1 = self._LAT._calc_max_width(self.image, [fpos_y, fpos_x], wi=self._track_width)
            cand_image = self.image[fpos_y-wid0:fpos_y+wid0,fpos_x-wid1:fpos_x+wid1]
            # perform search in small image
            new_particle = self._small_search(cand_image, particle)
            if len(new_particle)==0:
                continue
            # recalculate in the current frame
            particle[0] = new_particle[0][0] + fpos_x - wid1
            particle[1] = new_particle[0][1] + fpos_y - wid0
            t2 = t.time()
            # ensure a fast fitting process
            if self._fast == False:
                # perform a fit, diameter is now fixed
                fitted_particle = self._FIT.perform_opencl_fit(self.image, [particle], False)
            else:
                fitted_particle = np.array(np.round([particle], 2), dtype=float)
            t3 = t.time()
            if len(fitted_particle) > 0:
                tracked_particles.append(fitted_particle[0])

            # debug print
            if ('full' in self._debug_str or 'time' in self._debug_str):
                print('Estimate:\t', round((t2-t1)*1000,2), 'ms \t(tracking)')
                print('Fitting:\t', round((t3-t2)*1000,2), 'ms\t(tracking)')
                
        tracked_particles = np.round(tracked_particles, 2)
        return tracked_particles

    # update for new particles
    def _update_particles(self):
        # perform a two-grid symmetry transform to cover the whole image
        t1 = t.time()
        particles_all = self._LAT.determine_xy_positions(self.image)
        # check for new particles
        new_particles = []
        for particle in particles_all:
            # check for occurance
            occ = 0
            for f_particle in self.fitted_particles:
                dist = np.linalg.norm(particle-f_particle[:2])
                # assume its the same particle
                if dist <= self._track_width:
                    occ += 1
            if occ == 0:
                new_particles.append(particle)
        # estimate the z-positon with the theory
        new_particles = self._AXI.estimate_z_positions(self.image, np.array(new_particles))
        t2 = t.time()
        # determine between + and -
        new_particles = self._FIT.distinguish_pm_particles(self.image, new_particles)
        # final fit
        if self._fast == False:
            new_particles = self._FIT.perform_opencl_fit(self.image, new_particles, True)
        else:
            new_particles = np.array(np.round(new_particles, 2), dtype=float)
        t3 = t.time()

        # debug print
        if ('full' in self._debug_str or 'time' in self._debug_str):
            print('Estimate:\t', round((t2-t1)*1000,2), 'ms \t(update)')
            print('Fitting:\t', round((t3-t2)*1000,2), 'ms\t(update)')

        # track old particles
        old_particles = self._track_particles()
        # combine the lists
        if (len(old_particles)==0 and len(new_particles)==0):
            self.fitted_particles = np.array([])
            return self.fitted_particles

        # if at least one of the two lists contains elements, continue
        if len(old_particles)==0:
            self.fitted_particles = new_particles
        elif len(new_particles)==0:
            self.fitted_particles = old_particles
        else:
            self.fitted_particles = np.append(old_particles, new_particles, axis=0)
        return self.fitted_particles

    # ---------- main functions to execute ---------- #
    
    # get a full image 3D estimate for all possible particles
    def perform_initial_search(self, img, output="px"):
        # set image in class
        outcome = self._check_image(img)
        if outcome == 0:
            print('No image set.')
            return np.array([])

        # set counter to zero to ensure correct refresh rate
        self.counter = 0
        # perform a two-grid symmetry transform to cover the whole image
        t1 = t.time()
        particles = self._LAT.determine_xy_positions(img)
        # estimate the z-positon with the theory
        particles = self._AXI.estimate_z_positions(img, particles)
        t2 = t.time()
        # determine between + and -
        particles = self._FIT.distinguish_pm_particles(img, particles)
        # final fit
        if self._fast == False:
            self.fitted_particles = self._FIT.perform_opencl_fit(img, particles, True)
        else:
            self.fitted_particles = np.array(np.round(particles, 2), dtype=float)
        t3 = t.time()

        transformed = np.array(self.fitted_particles)
        if len(self.fitted_particles) != 0:
            transformed = self.fitted_particles[::, :4]

        if ('full' in self._debug_str or 'time' in self._debug_str):
                print('Estimate:\t', round((t2-t1)*1000,2), 'ms \t(initial)')
                print('Fitting:\t', round((t3-t2)*1000,2), 'ms\t(initial)')

        # debug plot
        if ('full' in self._debug_str or 'loc' in self._debug_str):
            self._LAT._debug.debug_stage(self.fitted_particles, "localized particles",
                                        self.image, self._debug_str)

        if output == 'px':
            return np.sort(transformed, axis=0)
        else:
            return np.sort(self._transform_into_um(transformed, img), axis=0)

    # track particles and update them on a certain refreshrate
    def track_particles(self, img, output="px"):
        # set image in class
        outcome = self._check_image(img)
        if outcome == 0:
            print('No image set.')
            return np.array([])

        # check if the counter exists, if not set it to zero
        if not hasattr(self, 'counter'):
            self.counter = 0
        # check if the particle list exists, if not create an empty one
        if not hasattr(self, 'fitted_particles'):
            self.fitted_particles = np.array([])
        # based on the counter value decide wheter they are updated or tracked
        if self.counter == 0:
            self.fitted_particles = self._track_particles()
        else:
            if self.counter%(self._refreshrate-1) == 0:
                self.fitted_particles = self._update_particles()
            else:
                self.fitted_particles = self._track_particles()
        # increase the counter
        self.counter += 1

        transformed = self.fitted_particles
        if len(self.fitted_particles) != 0:
            transformed = self.fitted_particles[::, :4]

        # debug plot
        if ('full' in self._debug_str or 'loc' in self._debug_str):
            self._LAT._debug.debug_stage(self.fitted_particles, "localized particles",
                                        self.image, self._debug_str)

        if output == 'px':
            return np.sort(transformed, axis=0)
        else:
            return np.sort(self._transform_into_um(transformed, img), axis=0)
