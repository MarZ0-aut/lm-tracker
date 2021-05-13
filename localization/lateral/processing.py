# import necessary modules and functions
import numpy as np
import itertools as it

class Processing(object):
    # init definition and tuning parameters
    def __init__(self, tuning):
        self._image_cut_percent = tuning["image_cut_percent"]

    # number of maximum sized squares that fit into a rectangle
    def _calc_number_squares(self, x, l, w):
        gcd = np.gcd(l-x[0], w-x[1])
        return int((l-x[0])*(w-x[1])/(gcd*gcd))

    # find the minimum cutoff and the biggest squares to fit in that rectangle
    def _calc_reduction_parameters(self, img):
        # length and width of the incoming image
        le, wi = img.shape
        perc = self._image_cut_percent
        # maximum amount of pixels you want to cut off the image
        l_max, w_max = int(le*perc), int(wi*perc)
        # safety mechanism, ensures that arrays are not empty
        if l_max == 0:
            l_max = 1
        if w_max == 0:
            w_max = 1
        l_ran, w_ran = np.multiply(np.arange(l_max),1)[:l_max], np.multiply(np.arange(w_max),1)[:w_max]
        # all possible pixel-cutoff combinations
        cuts = list(it.product(*[l_ran, w_ran]))
        factors = list(map(lambda X: self._calc_number_squares(X, le, wi), cuts))
        # get the minimum factor and its gcd
        min_index = np.argmin(factors)
        min_factor = factors[min_index]
        min_cuts = cuts[min_index]
        max_gcd = np.gcd(le-min_cuts[0], wi-min_cuts[1])
        return min_factor, min_cuts, max_gcd

    # slice image at specific position
    def _slice_img(self, index, img, gc):
        iy, ix = index
        return img[ix*gc:gc*(ix+1):,iy*gc:gc*(iy+1)]

    # get the left-upper corner of a sliced image
    def _get_corner_img(self, index, gc, le, wi):
        idy, idx = index
        return [idx*gc+le, idy*gc+wi]

    # prepare image, slice it and keep corner positions
    def prepare_image(self, img, params):
        # check if image length/width is a multiple of 2, if not cut one pixel off
        len0, len1 = img.shape
        l2 = len0%2
        w2 = len1%2
        cut_img = img[l2:,w2:]
        if len(params)==0:
            # find optimal width and height of image to splice
            nr_imgs, xy_cuts, gcd = self._calc_reduction_parameters(cut_img)
            # cutoff excess border of incoming image to ensure a high enough gcd
            hei_c, wid_c = xy_cuts
            hei_c //= 2
            wid_c //= 2
        else:
            # cutoff excess border of incoming image to ensure a high enough gcd
            wid_c, hei_c = params[0]
            # number of images in "image-grid"
            gcd = params[1]
        # number of images in "image-grid"
        gcd0, gcd1 = int(len0/gcd), int(len1/gcd)
        cut_img = cut_img[hei_c:len0-hei_c,wid_c:len1-wid_c]
        # slice image into smaller images and get their corner positions
        indices = list(it.product(*[np.arange(gcd1),np.arange(gcd0)]))
        square_imgs = list(map(lambda X: self._slice_img(X, cut_img, gcd), indices))
        corner_pos = list(map(lambda X: self._get_corner_img(X, gcd, l2, w2), indices))
        return square_imgs, corner_pos, gcd, (wid_c, hei_c)