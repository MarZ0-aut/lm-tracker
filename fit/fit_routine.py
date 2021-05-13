# import necessary modules and functions
import numpy as np
np.set_printoptions(suppress=True)

from .debug_plots_fit import Debugging_Fit

# ---------- opencl fitting ---------- #
class OpenCLFit(object):
    # init definiton and tuning parameters
    def __init__(self, tuning, theory, holo, debug_str):
        self._holo = holo
        self._debug_str = debug_str
        self._debug = Debugging_Fit(tuning, theory, self._holo)

        self._dx = theory["dx"]
        self._n_p = theory["n_p"]
        self._np = theory["n_p"]
        self._phi = theory["phi"]
        self._theta = theory["theta"]
        self._psi = theory["psi"]

        self._plus_minus_width = tuning["pm_width"]
        self._computation_mode = tuning["comp_mode"]
        self._components = tuning["components"]
        self._particle_group_distance = tuning["group_distance"]
        self._NA_offset = tuning["NA_offset"]
        self._opencl_width = tuning["opencl_width"]
        self._alpha_fit = tuning["alpha_fit"]

    # calculate maximum width for cut out image
    def _calc_max_width(self, img, pos, wi):
        wid = [wi, wi]
        # check if it has to be reduced
        for i in range(2):
            vals = [wi, wi]
            if pos[i]-wid[i] <= 0:
                vals[0] = pos[i]
            if pos[i]+wid[i] >= img.shape[i]-1:
                vals[1] = img.shape[i]-1-pos[i]
            wid[i] = min(vals)
        return np.array(wid, dtype=int)

    # preparation for opencl calculation (to prevent duplicate code)
    def _prep_opencl(self, particle, wid):
        width = wid/2
        # restrict image to position, only small image around center
        px, py = particle[:2]
        fpos_x = int(round(px, 0))
        fpos_y = int(round(py, 0))
        # recalculate in holo frame
        frem_x = (px-fpos_x)*self._dx
        frem_y = -(py-fpos_y)*self._dx
        # retreive image that is plugged into fitting class
        wid0, wid1 = self._calc_max_width(self._image, [fpos_y, fpos_x], wi=width)
        fimage = self._image[fpos_y-wid0:fpos_y+wid0, fpos_x-wid1:fpos_x+wid1]
        return fimage, (fpos_x, fpos_y), (frem_x, frem_y), [wid0, wid1]

    # comparison value
    def _comp_value(self, input_img, reference_img, array):
        return np.sum(np.square((reference_img-input_img)*array))

    # determine wheter it is + or -
    def _distinguish_pm_particle(self, particle, holo):
        px, py, pz, pd, mask_rad = particle
        # preparation function
        fit_image, pf, pr, wid = self._prep_opencl(particle, self._plus_minus_width)
        xr, yr = pr
        # pass shape to holo class
        holo.set_shape(fit_image.shape, False, False, self._computation_mode)
        # create fitting mask
        x = np.linspace(-wid[1]-0.5, wid[1]-0.5, 2*wid[1])
        y = np.linspace(-wid[0]-0.5, wid[0]-0.5, 2*wid[0])
        X, Y = np.meshgrid(x, y, sparse=True)
        R2 = X*X + Y*Y
        mask = R2 >= mask_rad*mask_rad

        # + image
        sol_p = pz+pd
        plus = holo.calcholo([[pd, xr, yr, sol_p, self._n_p, 1,
                               self._phi[0], self._theta[0], self._psi[0], 0]],
                             components=self._components)
        pmean = self._comp_value(plus, fit_image, mask)
        # - image
        sol_m = -pz+pd
        minus = holo.calcholo([[pd, xr, yr, sol_m, self._n_p, 1,
                                self._phi[0], self._theta[0], self._psi[0], 0]],
                              components=self._components)
        mmean = self._comp_value(minus, fit_image, mask)

        # debug plot
        if ('full' in self._debug_str or 'pm' in self._debug_str or 'save' in self._debug_str):
            self._debug.debug_pm(fit_image, plus, minus, pmean, mmean, mask, np.sign(pz),
                                 self._debug_str, [px, py])

        # check which one is more similar to real data
        if pmean < mmean:
            return [px, py, sol_p, pd, 1, mask_rad]
        else:
            return [px, py, sol_m, pd, 1, mask_rad]

    # determine wheter it is + or -
    def distinguish_pm_particles(self, image, particles):
        self._image = image
        len_particles = len(particles)
        if len_particles == 0:
            return []
        if len_particles == 1:
            distinguished = self._distinguish_pm_particle(particles[0], self._holo)
            return [distinguished]
        # distinguish individual particles
        distinguished = []
        for particle in particles:
            data = self._distinguish_pm_particle(particle, self._holo)
            distinguished.append(data)
        return distinguished

    # perform a matrix multiplication to create an overlap
    def _single_overlap(self, array):
        M0 = np.dot(array, array.T)
        M1 = np.not_equal(M0, self._Zero_overlap)
        return M0, np.array(M1, dtype=int)

    # full overlap function, iterates until there is no change
    def _overlap(self, array):
        # set zero matrix
        self._Zero_overlap = np.zeros(array.shape)
        # create starting point of recursion
        M00, M01 = self._single_overlap(array)
        # check for first difference
        M10, M11 = self._single_overlap(M00)
        while np.sum(M11-M01)>0:
            # update matrix
            M01 = M11
            M10, M11 = self._single_overlap(M10)
        return np.not_equal(M10, self._Zero_overlap)

    # find particles that are close together
    def _find_particle_vicinity(self, particles):
        len_parts = len(particles)

        # create distance matrix, 1 for close together, 0 for not
        M0 = np.diag(np.arange(1, len_parts+1))
        for i in range(len_parts):
            for j in range(len_parts):
                dist = np.linalg.norm(particles[i][:2]-particles[j][:2])
                if dist < self._particle_group_distance:
                    M0[i][j] = 1

        # create an overlap (eg. 2 <-> 4, 2 <-> 3 therefore 3 <-> 4)
        O = self._overlap(M0)

        # calculate a mask array
        group_mask = np.unique(O, axis=0)
        return group_mask

    # check wheter NA correction of LC correction is necessary
    def _check_switches(self, axials, diameters, wids):
        """
        Dependent on the diameter, the axial position and the size of
        the image, a typical correction is necessary. Parameters used are rough
        estimates received by empirical testing.
        """
        effects = []
        # iterate over all particles
        N = len(axials)
        for i in range(N):
            z_pos, diam = axials[i], diameters[i]
            # positive relation
            if np.sign(z_pos) == 1:
                value = 1.0350494075*z_pos - diam/2
            # negative relation
            else:
                value = -1.0228782275000001*z_pos + diam
            # relate effect to image size
            effects.append(value/self._dx)
        # NA effect occurrance
        if np.amin(effects) < np.amax(wids)+self._NA_offset:
            NA_corr = True
        else:
            NA_corr = False
        return NA_corr, True

    # single particle fit
    def _perform_single_particle_fit(self, particle, bool_diam):
        part = particle[0]
        # setup fit parameters
        fit_image, pf, pr, wid = self._prep_opencl(part, self._opencl_width)
        xr, yr = pr
        px, py = pf
        # pass shape to holo class
        switch_NA, switch_LC = self._check_switches([part[2]], [part[3]], wid)
        self._holo.set_shape(fit_image.shape, switch_NA, switch_LC, self._computation_mode)

        # perform the fit
        parfit = self._holo.fit_sel(fit_image, [[part[3], xr, yr, part[2], self._n_p,
                                           part[4], self._phi[0], self._theta[0], self._psi[0], 0]],
            np.array([bool_diam, True, True, True, False, self._alpha_fit,
                      self._phi[1], self._theta[1], self._psi[1], False], dtype=np.bool), [part[5]],
            components=self._components)
        d, x, y, z, n2, alpha, phi, theta, psi, prop_dist = parfit[0]
        # recalculate in original frame
        posx = px + x/self._dx
        posy = py - y/self._dx
        posz = z
        diam = d

        # debug plot
        if ('full' in self._debug_str or 'fit' in self._debug_str
            or 'save' in self._debug_str or 'resid' in self._debug_str):
            self._debug.debug_opencl(fit_image, diam, x, y, posz, alpha, phi, theta, psi, posx, posy,
                                     switch_NA, switch_LC, self._debug_str)
        return [posx, posy, posz, diam, alpha, part[5]]

    # multi particle fit
    def _perform_multi_particle_fit(self, group, bool_diam):
        group_len = len(group)

        left, right, top, bottom = [], [], [], []
        centerx, centery = 0, 0
        z_positions = []
        diameters = []
        for i in range(group_len):
            # unpack particle parameters
            px, py, pz, pd, alp, rad = group[i]
            # sum up to calculate center position
            centerx += px
            centery += py
            # retreive calculated distance to edge
            p0, p1 = int(round(px, 0)), int(round(py, 0))
            wid = self._calc_max_width(self._image, [p1, p0], wi=self._opencl_width)
            # add edge-distances to list
            left.append(p0-wid[1]//2)
            right.append(p0+wid[1]//2)
            top.append(p1-wid[0]//2)
            bottom.append(p1+wid[0]//2)
            z_positions.append(pz)
            diameters.append(pd)
        # calculate center position
        centerx /= group_len
        centery /= group_len
        centerx = int(round(centerx, 0))
        centery = int(round(centery, 0))

        fl, fr, ft, fb = [], [], [], []
        pars = []
        rads = []
        for i in range(group_len):
            # calculation for center widths
            fl.append(centerx-left[i])
            fr.append(right[i]-centerx)
            ft.append(centery-top[i])
            fb.append(bottom[i]-centery)
            # unpack particle parameters
            px, py, pz, pd, alp, rad = group[i]
            # set radii and parameters
            rads.append(rad)
            pars.append([pd, (px-centerx)*self._dx, (py-centery)*self._dx, pz,
                         self._n_p, alp,
                         self._phi[0], self._theta[0], self._psi[0], 0])

        # create fit image
        fwid0, fwid1 = max(max(ft), max(fb)), max(max(fr), max(fl))
        fit_image = self._image[centery-fwid0:centery+fwid0, centerx-fwid1:centerx+fwid1]
        # set the shape and perform the N particle fit
        switch_NA, switch_LC = self._check_switches(z_positions, diameters, [fwid0, fwid1])
        self._holo.set_shape(fit_image.shape, switch_NA, switch_LC, self._computation_mode)
        parfit = self._holo.fit_sel(fit_image, pars,
            np.array([bool_diam, True, True, True, False, self._alpha_fit,
                      self._phi[1], self._theta[1], self._psi[1], False], dtype=np.bool), rads,
            components=self._components)

        # retreive solution and recalculate in original pixel frame
        particles_temp = []
        for i in range(group_len):
            d, x, y, z, n2, alpha, phi, theta, psi, prop_dist = parfit[i]
            posx = centerx + x/self._dx
            posy = centery + y/self._dx
            posz = z
            diam = d
            particles_temp.append([posx, posy, posz, diam, alpha, group[i][5]])

            # debug plot
            if ('full' in self._debug_str or 'fit' in self._debug_str
                or 'save' in self._debug_str or 'resid' in self._debug_str):
                self._debug.debug_opencl(fit_image, diam, x, y, posz, alpha, phi, theta, psi, posx, posy,
                                   switch_NA, switch_LC, self._debug_str, x/self._dx, y/self._dx)
        return particles_temp

    # perform the pyopencl fit for all found particles
    def perform_opencl_fit(self, image, particles, bool_diam):
        self._image = image
        particles = np.array(particles)
        # create particle groups that are close together
        groups = []
        if len(particles) != 0:
            # group mask
            group_mask = self._find_particle_vicinity(particles)

            # retreive the groups being close together
            for i in range(len(group_mask)):
                groups.append(particles[group_mask[i]])
        # split into singles and groups
        singles = []
        multiples = []
        for group in groups:
            if len(group) == 1:
                singles.append(group)
            else:
                multiples.append(group)
        len_singles = len(singles)
        len_multiples = len(multiples)

        particles_fitted = []
        # single particle fit
        if len_singles > 0:
            # if only a single particle
            if len_singles == 1:
                particle_s = self._perform_single_particle_fit(singles[0], bool_diam)
                particles_fitted.append(particle_s)
            else:
                # if multiple single particles
                for single in singles:
                    data = self._perform_single_particle_fit(single, bool_diam)
                    particles_fitted.append(data)

        # multi particle fit
        if len_multiples > 0:
            # if only 1 particle group
            if len_multiples == 1:
                particles_m = self._perform_multi_particle_fit(multiples[0], bool_diam)
                for particle_m in particles_m:
                    particles_fitted.append(particle_m)
            else:
                # if multiple particle groups
                for multiple in multiples:
                    data = self._perform_multi_particle_fit(multiple, bool_diam)
                    for dat in data:
                        particles_fitted.append(dat)
        return np.array(np.round(particles_fitted, 2), dtype=float)