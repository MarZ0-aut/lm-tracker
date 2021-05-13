# import necessary modules and functions
import numpy as np
import itertools as it

from scipy.signal import find_peaks
from concurrent.futures import as_completed
from numpy.polynomial.polynomial import polyfit, polyroots

from .processing import Processing
from .debug_plots_lat import Debugging_Lateral
from .symmetry_transformation import sym_trafo_Bowman

class Lateral(object):
    # init definition and tuning parameters
    def __init__(self, executor, tuning, debug_str=""):
        # carry over the objects
        self._debug = Debugging_Lateral()
        self._Proc = Processing(tuning)

        self._debug_str = debug_str
        self._executor = executor

        self._candidate_contrast = tuning["candidate_contrast"]
        self._candidate_threshold = tuning["candidate_threshold"]
        self._candidate_distance = tuning["candidate_distance"]
        self._candidate_shapes = tuning["candidate_shapes"]
        self._candidate_decentering = tuning["candidate_decentering"]
        self._candidate_fit_width = tuning["candidate_fit_width"]

    # perform a full 2D symmetry transform, separate dimensions
    def _perform_1D_symmetry_transform(self, img, pos, string=""):
        # subtract mean to ensure proper work of ST
        norm_img = np.subtract(img, np.mean(img))
        # symmetry transform for axis 0
        st1 = sym_trafo_Bowman(norm_img, 0)
        # break condition
        if (np.amax(st1)/np.mean(st1)/len(st1) < self._candidate_contrast):
            return np.array([]), np.array([])
        # symmetry transform for axis 1
        st0 = sym_trafo_Bowman(norm_img, 1)
        # break condition
        if (np.amax(st0)/np.mean(st0)/len(st0) < self._candidate_contrast):
            return np.array([]), st1

        # debug plot
        if ('full' in self._debug_str or 'ST' in self._debug_str or 'save' in self._debug_str):
            self._debug.debug_symmetry_trafo(st0, st1, img, pos, string, self._debug_str)
        return st0, st1

    # find possible candidates (first search in one square)
    def _find_possible_candidates(self, img, corner):
        # perform a symmetry transform on the image
        ST0, ST1 = self._perform_1D_symmetry_transform(img, corner, "corner ")
        if ((len(ST0)==0) or (len(ST1)==0)):
            return np.array([])
        # find possible positions for x and y direction, divide by 2 because of convolution
        cand0 = find_peaks(ST0, height=self._candidate_threshold,
                           distance=self._candidate_distance)[0]/2
        if len(cand0)==0:
            return np.array([])
        cand1 = find_peaks(ST1, height=self._candidate_threshold,
                           distance=self._candidate_distance)[0]/2
        if len(cand1)==0:
            return np.array([])
        # get possible candidate positions
        cand = np.array(list(it.product(*[cand0, cand1])))
        return cand

    # perform the symmetry transform search on an image
    def _perform_single_image_symmetry_transform(self, pack, excess):
        img, corner = pack
        wid, hei = excess
        # perform a full symmetry transform to find all possible positions
        cy, cx = corner
        candidate_quadrant = self._find_possible_candidates(img, [cx, cy])
        candidates = []
        for particle in candidate_quadrant:
            px, py = particle
            # recalculate in the original size of the image
            posx = px+cx+wid
            posy = py+cy+hei
            candidates.append([posx, posy])
        return candidates

    # grid analysis for one grid
    def _do_single_grid_symmetry_transform(self, imgs, corners, excess):
        len_imgs = len(imgs)
        if len_imgs == 0:
            return []
        if len_imgs == 1:
            parts = self._perform_single_image_symmetry_transform((imgs[0], corners[0]), excess)
            return parts
        # create futures to pass to Pool
        futures = list(map(lambda X: self._executor.submit(
                self._perform_single_image_symmetry_transform,
                X, excess), zip(imgs, corners)))
        # pick up results
        parts = []
        for res in as_completed(futures):
            data = res.result()
            for dat in data:
                parts.append(dat)
        return parts

    # check for positions occuring twice
    def _combine_found_positions(self, posA, posB):
        posA = np.array(posA)
        posB = np.array(posB)
        posA_shape = len(posA)
        posB_shape = len(posB)
        if (posA_shape==0 and posB_shape==0):
            return np.array([])
        # if at least one of the two lists contains elements, continue
        if posA_shape==0:
            return posB
        if posB_shape==0:
            return posA
        # the case that both lists contain positions
        # get tuples for all positions
        combs = list(it.product(*[posA, posB]))
        indices = list(it.product(*[np.arange(posA_shape), np.arange(posB_shape)]))
        com_pos = []
        delA = []
        delB = []
        for i in range(len(combs)):
            # calculate distance between the tuples
            dist = np.linalg.norm(combs[i][0]-combs[i][1])
            # if the two found positions are close together, one can
            # assume that it is the same particle
            if dist <= self._candidate_distance/2:
                posx = (combs[i][0][0]+0.5*abs(combs[i][0][0]-combs[i][1][0]))
                posy = (combs[i][0][1]+0.5*abs(combs[i][0][1]-combs[i][1][1]))
                com_pos.append([posx, posy])
                delA.append(indices[i][0])
                delB.append(indices[i][1])
        # remove twice occuring particles
        posA = np.delete(posA, delA, axis=0)
        posB = np.delete(posB, delB, axis=0)
        for i in range(posA.shape[0]):
            com_pos.append(posA[i])
        for i in range(posB.shape[0]):
            com_pos.append(posB[i])
        # ensure that there are no solutions occuring twice
        com_pos = np.round(np.unique(np.array(com_pos), axis=0), 0)
        return com_pos

    # symmetry transform for an image that is not a square image but large enough
    # to be cut into smaller squares, evaluate on two displaced grids for better
    # particle detection
    def _do_two_grid_symmetry_transform(self):
        # receive data for lattice A
        imgs_A, corners_A, displacement_A, excess_A = self._Proc.prepare_image(self._image, [])

        """ _________________           _________________
            |               |           | 0 | 3 | 6 | 9 |
            |               |           |___|___|___|___|
            |               |     ->    | 1 | 4 | 7 | 10|
            |               |           |___|___|___|___|
            |               |           | 2 | 5 | 8 | 11|
            |_______________|           |___|___|___|___| """
        # and displaced lattice B
        displ = displacement_A//2
        image_B = self._image[displ:-displ,displ:-displ]
        """ _________________           _________________
            |               |           | _____________ |
            |               |           | | 12| 14| 16| |
            |               |     ->    | |___|___|___| |
            |               |           | | 13| 15| 17| |
            |               |           | |___|___|___| |
            |_______________|           |_______________| """
        imgs_B, corners_B, displacement_B, excess_B = self._Proc.prepare_image(image_B, [excess_A, displacement_A])
        corners_B += displ
        # debug plot
        if ('full' in self._debug_str or 'grid' in self._debug_str or 'save' in self._debug_str):
            self._debug.debug_grid(imgs_A, self._debug_str, "cut_grid")
            self._debug.debug_grid(imgs_B, self._debug_str, "overlap_grid")

        # analysis on lattice A
        pos_A = self._do_single_grid_symmetry_transform(imgs_A, corners_A, excess_A)
        # analysis on lattice B
        pos_B = self._do_single_grid_symmetry_transform(imgs_B, corners_B, excess_B)
        # get the combined positions
        positions = self._combine_found_positions(pos_A, pos_B)

        # debug plot
        if ('full' in self._debug_str or 'search' in self._debug_str or 'save' in self._debug_str):
            self._debug.debug_stage(positions, 'grid search', self._image, self._debug_str)
        return np.array(positions, dtype=np.int)

    # ---------- sorting candidates and fitting in 2D ---------- #

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

    # single function for sorting out candidates
    def _sort_out_candidate(self, img, pack):
        width_small, width_big = self._candidate_shapes
        p0, p1 = pack[0]
        i = pack[1]
        # small image
        w00, w01 = self._calc_max_width(img, [p1, p0], wi=width_small/2)
        img_mod0 = img[p1-w00:p1+w00, p0-w01:p0+w01]
        ST00, ST01 = self._perform_1D_symmetry_transform(img_mod0, [p0, p1], "pos ")
        # check for small image if threshold and centering are ok
        small, big = False, False
        if not (len(ST00)==0 or len(ST01)==0):
            if ((np.amax(ST00)>self._candidate_threshold) and (np.amax(ST01)>self._candidate_threshold)):
                if ((np.abs(np.argmax(ST00)-width_small)<self._candidate_decentering) and
                    (np.abs(np.argmax(ST01)-width_small)<self._candidate_decentering)):
                        small = True
        # if small image fails, try big image
        if small == False:
            # big image
            w10, w11 = self._calc_max_width(img, [p1, p0], wi=width_big/2)
            img_mod1 = img[p1-w10:p1+w10, p0-w11:p0+w11]
            ST10, ST11 = self._perform_1D_symmetry_transform(img_mod1, [p0, p1], "pos ")
            # check for big image if threshold and centering are ok
            if not (len(ST10)==0 or len(ST11)==0):
                if ((np.amax(ST10)>self._candidate_threshold) and (np.amax(ST11)>self._candidate_threshold)):
                    if ((np.abs(np.argmax(ST10)-width_big)<self._candidate_decentering) and
                        (np.abs(np.argmax(ST11)-width_big)<self._candidate_decentering)):
                            big = True
        # continue if either small and/or big image is valid
        if (small==True or big==True):
            self._index_sorting[i] = 999999
            if small==True:
                self._STs_sorting.append((ST00, ST01))
                self._shapes_sorting.append(img_mod0.shape)
            else:
                self._STs_sorting.append((ST10, ST11))
                self._shapes_sorting.append(img_mod1.shape)
        else:
            self._index_sorting[i] = i
            self._shapes_sorting.append(img_mod0.shape)
            self._STs_sorting.append((ST00, ST01))

    # sort out candidates
    def _sort_out_candidates(self, img, cands):
        nr_cands = len(cands)
        if nr_cands == 0:
            return np.array([], dtype='int'), [], []
        # initialize arrays that data is written to
        self._index_sorting = np.arange(nr_cands)
        self._shapes_sorting = []
        self._STs_sorting = []
        # create the futures
        futures = list(map(lambda X: self._executor.submit(
                self._sort_out_candidate,
                img, X), zip(cands, self._index_sorting)))
        # run in parallel
        for res in as_completed(futures):
            res.result()
        # remove candidates that are not centered or just are false detections
        shapes = np.array(self._shapes_sorting)
        STs = np.array(self._STs_sorting)
        index = np.delete(self._index_sorting, np.where(self._index_sorting==999999))
        cands = np.delete(cands, index, axis=0)
        shapes = np.delete(shapes, index, axis=0)
        STs = np.delete(STs, index, axis=0)

        # debug plot
        if ('full' in self._debug_str or 'sorted' in self._debug_str or 'save' in self._debug_str):
            self._debug.debug_stage(cands, 'sorted candidates', img, self._debug_str)
        return cands, shapes, STs

    # symmetry transform: sub-pixel fit
    def _perform_particle_fit(self, st):
        # take the derivative of S
        if len(st) == 0:
            return np.array([])
        dS = np.gradient(st)
        # find the boundary values for fitting
        argmax = np.argmax(st)
        dS_min = argmax-self._candidate_fit_width
        dS_max = argmax+self._candidate_fit_width
        pfit = np.arange(dS_min, dS_max, 1)
        Sfit = dS[dS_min:dS_max]
        try:
            # fit to the derivative
            coeff = polyfit(pfit, Sfit, 3)
            # find the root of the fit function
            root = np.array([np.abs(polyroots(coeff)[1])])/2
        except:
            root = np.array([])
        finally:
            return root

    # single function for a 2D estima
    def _estimate_2D_position(self, pack):
        shape, cand, STs = pack
        cand0, cand1 = cand
        shape0, shape1 = shape
        ST0, ST1 = STs
        # fit to the particle position
        pos0 = self._perform_particle_fit(ST0)
        if len(pos0) == 1:
            pos1 = self._perform_particle_fit(ST1)
            # if both roots are found, take as particle
            if len(pos1) == 1:
                posx = pos0[0]-shape1/2+cand0
                posy = pos1[0]-shape0/2+cand1
                return [posx, posy]

    # perform a fit on the found candidates
    def _estimate_2D_positions(self, imgs, cands, sts):
        len_cands = len(cands)
        if len_cands == 0:
            return []
        elif len_cands == 1:
            parts = self._estimate_2D_position((imgs[0], cands[0], sts[0]))
            return [parts]
        else:
            parts = []
            for i in range(len(cands)):
                data = self._estimate_2D_position((imgs[i], cands[i], sts[i]))
                if data != None:
                    parts.append(data)
        return parts

    # get a full image 3D estimate for all possible particles
    def determine_xy_positions(self, image):
        # initialize image
        self._image = image
        # find possible candidates
        candidates = self._do_two_grid_symmetry_transform()
        # sort the possible candidates
        sorted_candidates, candidate_images, STs = self._sort_out_candidates(self._image, candidates)
        # fit the remaining candidates = determine them as particles
        particles = self._estimate_2D_positions(candidate_images, sorted_candidates, STs)

        # debug plot
        if ('full' in self._debug_str or '2D' in self._debug_str or 'save' in self._debug_str):
            self._debug.debug_stage(particles, '2D fitted candidates', self._image, self._debug_str)
        return particles