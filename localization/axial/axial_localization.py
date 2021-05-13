# import necessary modules and functions
import numpy as np

from mkl_fft import fft, ifft
from numpy.fft import fftfreq
from PyAstronomy import pyaC
from scipy.stats import chisquare
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

from .debug_plots_ax import Debugging_Axial

class Axial(object):
    # init definiton and tuning parameters
    def __init__(self, tuning, theory, debug_str=""):
        # carry over the objects
        self._debug = Debugging_Axial(theory)
        self._debug_str = debug_str

        self._candidate_intensity_width = tuning["candidate_intensity_width"]
        self._intensity_distance = tuning["intensity_distance"]
        self._intensity_peak_height = tuning["intensity_peak_height"]
        self._intensity_first_peakpos = tuning["intensity_first_peakpos"]
        self._lower_frequency_factor = tuning["lower_frequency_factor"]
        self._intensity_number_peaks = tuning["intensity_number_peaks"]
        self._min_rad_backup = tuning["min_rad_backup"]
        self._signal_window_slope = tuning["signal_window_slope"]
        self._frequency_window_slope = tuning["frequency_window_slope"]
        self._higher_frequency = tuning["higher_frequency"]
        self._envelope_number_peaks = tuning["envelope_number_peaks"]
        self._envelope_chi = tuning["envelope_chi"]
        self._comparison_value = tuning["comparison_value"]

        self._dx = theory["dx"]
        self._lamda = theory["lamda"]
        self._n_m = theory["n_m"]
        self._d_p = theory["d_p"]

    # calculate maximum possible distance
    def _check_max_distances(self, position, positions):
        # define the 4 main directions
        left = []
        right = []
        top = []
        bottom = []
        for point in positions:
            # define the parameters
            x = position[0]-point[0]
            y = position[1]-point[1]
            rad = np.sqrt(x*x + y*y)
            d = dist0 = dist1 = self._intensity_distance
            # parameter correction
            if rad<dist0:
                dist0 = rad/2.
                dist1 = dist0*np.sqrt(2)
            if dist1>d:
                dist1 = d
            # check where the other particle sits
            if (abs(x)<1 and abs(y)<1):
                l, r, t, b = d, d, d, d
            elif (x>0 and y>0):
                l, r, t, b = dist1, d, dist1, d
            elif (x>0 and y<0):
                l, r, t, b = dist1, d, d, dist1
            elif (x<0 and y>0):
                l, r, t, b = d, dist1, dist1, d
            elif (x<0 and y<0):
                l, r, t, b = d, dist1, d, dist1
            elif (abs(x)<1 and y>0):
                l, r, t, b = d, d, dist0, d
            elif (abs(x)<1 and y<0):
                l, r, t, b = d, d, d, dist0
            elif (x>0 and abs(y<1)):
                l, r, t, b = dist0, d, d, d
            elif (x<0 and abs(y<1)):
                l, r, t, b = d, dist0, d, d
            left.append(l)
            right.append(r)
            top.append(t)
            bottom.append(b)
        # take the minimum of each list to ensure a non overlap
        le, ri = int(round(min(left), 0)), int(round(min(right), 0))
        to, bo = int(round(min(top), 0)), int(round(min(bottom), 0))
        return (le, ri, to, bo)

    # calculate new array with mean of two arrays
    def _calc_mean_two_arrs(self, arr1, arr2):
        len1 = arr1.shape[0]
        len2 = arr2.shape[0]
        if len1==len2:
            n_arr = np.mean([arr1, arr2], axis=0)
        elif len1>len2:
            n_arr = np.hstack((np.mean([arr1[:-(len1-len2)],arr2], axis=0),arr1[len2:]))
        else:
            n_arr = np.hstack((np.mean([arr2[:-(len2-len1)],arr1], axis=0),arr2[len1:]))
        return n_arr

    # calculate new array with mean from four arrays
    def _calc_mean_four_arrs(self, arr1, arr2, arr3, arr4):
        n_arr1 = self._calc_mean_two_arrs(arr1, arr2)
        n_arr2 = self._calc_mean_two_arrs(arr3, arr4)
        n_arr = self._calc_mean_two_arrs(n_arr1, n_arr2)
        return n_arr

    # comparison value if spherical symmetric
    def _compare_arrays(self, arr1, arr2, arr3, arr4, pos):
        # calculate lengths
        l1, l2, l3, l4 = len(arr1), len(arr2), len(arr3), len(arr4)
        
        # correct arrays to match lengths
        arr1 = np.flip(arr1, axis=0)
        arr3 = np.flip(arr3, axis=0)
        if l3 > l1:
            arr3 = arr3[(l3-l1):]
        elif l1 > l3:
            arr1 = arr1[(l1-l3):]
        if l4 > l2:
            arr4 = arr4[:-(l4-l2)]
        elif l2 > l4:
            arr2 = arr2[:-(l2-l4)]
        # create combined arrays
        horizontal = np.concatenate((arr1, arr2), axis=0)
        vertical = np.concatenate((arr3, arr4), axis=0)
        # create difference
        difference = np.subtract(horizontal, vertical)
        abs_diff = np.abs(difference)
        comp_value = np.mean(abs_diff)
        
        # debug plot
        if ('full' in self._debug_str or 'interf' in self._debug_str or 'save' in self._debug_str):
            self._debug.debug_interference(horizontal, vertical, np.round(comp_value, 3), self._debug_str, pos)
        
        return comp_value

    # combine intensity from 4 main directions
    def _combine_intensity(self, img, pos, dists):
        width = self._candidate_intensity_width
        p0, p1 = pos

        top = np.flip(np.mean(img[0:p1,p0-width:p0+width+1], axis=1))[:dists[2]]
        bottom = np.mean(img[p1:img.shape[0]-1,p0-width:p0+width+1], axis=1)[:dists[3]]
        left = np.flip(np.mean(img[p1-width:p1+width+1,0:p0], axis=0))[:dists[0]]
        right = np.mean(img[p1-width:p1+width+1,p0:img.shape[0]-1], axis=0)[:dists[1]]        
        # calculate maximum length of lists
        maxlist = np.amax([top.shape[0], bottom.shape[0], left.shape[0], right.shape[0]])
        # check for interference
        diff = self._compare_arrays(left, right, top, bottom, pos)
        # create maximum pixel list
        rad = np.arange(maxlist)
        # calculate into µm
        rad = np.multiply(rad, self._dx)
        # combine mean of all lists into one list
        sig = self._calc_mean_four_arrs(top, bottom, left, right)
        return rad, sig, diff

    # calculate low and high cutoff for intensity modification function
    def _calc_intensity_cutoff(self, rad, sig):
        len_s = len(sig)
        # find all peaks with given height
        peakpos, _ = find_peaks(sig, height=self._intensity_peak_height)
        if len(peakpos)==0:
            return [], peakpos, _, 0
        # retreive the radius for the masked fitting array, "remove center peak"
        min_rad = peakpos[0]
        # if first one is within amount of peaks it's the brightest spot in the middle, ignore
        if (peakpos[0] <= self._intensity_first_peakpos) and (np.argmax(_['peak_heights']) == 0):
            peakpos = np.delete(peakpos, [0])
        # number of peaks has to be greater or equal to 3 to succeed (minimum requirement)
        if len(peakpos) < 3:
            return [], peakpos, _, min_rad
        # check if there is a zero crossing between the first two peaks
        zerox = pyaC.zerocross1d(rad[peakpos[0]:peakpos[1]], sig[peakpos[0]:peakpos[1]])
        # if not, combine first two peaks into one
        if len(zerox) == 0:
            peakpos[1] = (peakpos[0]+peakpos[1])/2
            peakpos = np.delete(peakpos, [0])
        # lower frequency boundary is distance between first two peaks
        omega = (1/((peakpos[1] - peakpos[0])*self._dx))*self._lower_frequency_factor
        # find lower boundary condition for modification of signal
        stop = peakpos[1]
        xcl, xil = pyaC.zerocross1d(rad[:stop], sig[:stop], getIndices=True)
        if len(xcl)==0:
            lo = peakpos[0]
        else:
            lo = xil[-1]
        # find upper boundary condition for modification of signal
        len_p = len(peakpos)
        if len_p <= self._intensity_number_peaks:
            peaks = len_p
        else:
            peaks = self._intensity_number_peaks
        peaks -= 1
        go = peakpos[peaks]
        xch, xih = pyaC.zerocross1d(rad[go:], sig[go:], getIndices=True)
        if len(xch)==0:
            hi = len_s
        else:
            hi = xih[0] + peakpos[peaks]
        # additional mask correction
        if min_rad < self._min_rad_backup:
            min_rad = self._min_rad_backup
        return [omega, lo, hi], peakpos, _, min_rad

    # slow increasing and lowering function with constant in the middle
    def _get_dual_tanh(self, x, C2, C1, x2, x1):
        return 0.5*(np.tanh(-C1*(x-x1))+np.tanh(C2*(x-x2)))

    # modification function for the intensity
    def _modify_intensity(self, ran, left, right, slope):
        xvals = np.linspace(0, ran, ran)
        yvals = list(map(lambda x: self._get_dual_tanh(x, *[slope, slope, left, right]), xvals))
        yvals -= np.amin(yvals)
        return yvals

    # hilbert transformation with
    def _mod_hilbert_trafo(self, signal, data, pos):
        omega_low, lo, hi = data

        # remove mean value for adequate spectrum
        signal = np.subtract(signal, np.mean(signal))
        len_s = len(signal)
        # adequately pad signal
        power2 = np.power(2, np.arange(0, 15, 1))
        index = np.where(power2>2*len_s)[0][0]
        len_tot = power2[index]
        pad_width = len_tot-len_s
        padded = np.pad(signal, (0, pad_width), 'constant')

        # retreive indices for actual signal frequencies / µm
        f = fftfreq(len_tot)/self._dx
        min_freq = list(map(lambda x: abs(x-omega_low), f))
        max_freq = list(map(lambda x: abs(x-self._higher_frequency), f))
        min_index = np.where(min_freq==np.amin(min_freq))[0][0]
        max_index = np.where(max_freq==np.amin(max_freq))[0][0]

        # prepare modification function
        mod = self._modify_intensity(len_tot, min_index, max_index, self._frequency_window_slope)

        # perform fourier transform
        FT = fft(padded)
        # apply the band pass filter
        FT_corr = np.multiply(FT, mod)
        IFT = ifft(FT_corr)[:len_s]
        # calculate the phase
        phase = np.unwrap(np.angle(IFT))[lo:hi]

        # debug plot
        if ('full' in self._debug_str or 'filter' in self._debug_str or 'save' in self._debug_str):
            self._debug.debug_spectrum(f[:len_tot//2], np.abs(FT[:len_tot//2]),
                                       np.abs(FT_corr[:len_tot//2]), signal,
                                       IFT.real, phase, (lo, hi), self._debug_str, pos)
        return phase, IFT.real

    # fit function for phase
    def _get_fit_function(self, x, A, B):
        return np.sqrt((x*x)+(A*A))-B

    # jacobi matrix for curve_fit
    def _get_jacobian(self, x, A, B):
          dA = A/(np.sqrt(x*x+A*A))
          dB = -np.ones_like(x)
          return np.vstack((dA, dB)).T

    # fit function for envelope function
    def _get_envelope_function(self, x, A, B):
        return A*np.exp(-B*x)

    # single particle z estimate
    def _estimate_z_position(self, part, factor, img_mean, positions):
        px, py = part
        # round position to nearest integer
        position = [int(round(px,0)), int(round(py,0))]
        distances = self._check_max_distances(position, positions)
        # grab intensity from 4 main directions
        initial_radius, initial_signal, comp_value = self._combine_intensity(self._image, position, distances)
        if comp_value > self._comparison_value:
            return None
        # center signal around 0 if necessary
        if img_mean > 0.05:
            norm_signal = np.subtract(initial_signal, img_mean)
        else:
            norm_signal = initial_signal
        # calculate lower frequency cutoff
        cutoffs, peaks, _, min_rad = self._calc_intensity_cutoff(initial_radius, norm_signal)
        if len(cutoffs) == 0:
            return None
        # signal modification
        modification = self._modify_intensity(initial_signal.shape[0], cutoffs[1], cutoffs[2], self._signal_window_slope)
        signal = np.multiply(norm_signal, modification)
        # calculate the instantaneous phase
        inst_phase, mod_signal = self._mod_hilbert_trafo(signal, cutoffs, part)
        if len(inst_phase)==0:
            return None

        # retreive the overall shape of the peaks and estimate the preferred sign
        heights = _['peak_heights']
        index = np.where(heights==np.amax(heights))[0][0]
        peaks = peaks[index:index+self._envelope_number_peaks]
        try:
            fitvals = peaks - peaks[0]
            popt0, pcov0 = curve_fit(self._get_envelope_function,
                                       fitvals, norm_signal[peaks],
                                       p0=[norm_signal[peaks[0]], 0.05])
            success = 1
        except:
            success = 0
        if success == 1:
            fit = list(map(lambda x: self._get_envelope_function(x, *popt0), fitvals))
            chi, p = chisquare(norm_signal[peaks], fit)
            if chi > self._envelope_chi:
                sign = +1
            else:
                sign = -1

            # debug plot
            if ('full' in self._debug_str or 'envelope' in self._debug_str or 'save' in self._debug_str):
                self._debug.debug_envelope(peaks, fitvals, fit, chi, sign, norm_signal, self._debug_str, part)
        else:
            sign = +1

        low, high = cutoffs[1], cutoffs[2]
        # get list of x values for fit
        fit_radius = initial_radius[low:high]
        # rescale the phase for accordance to the theory
        inst_phase = np.divide(inst_phase, factor)

        # try the phase fit
        try:
            popt1, pcov1 = curve_fit(self._get_fit_function, fit_radius, inst_phase,
                                   p0=[1, factor*inst_phase[0]], method='lm', jac=self._get_jacobian)
        except:
            return None

        # debug plot
        if ('full' in self._debug_str or 'phase' in self._debug_str or 'save' in self._debug_str):
            fit = list(map(lambda x: self._get_fit_function(x, *popt1), fit_radius))
            self._debug.debug_intensity(fit_radius, initial_signal, mod_signal, position,
                                        inst_phase, fit, self._image, self._debug_str)
        return [px, py, sign*abs(popt1[0]), self._d_p, min_rad]

    # determine the z position of the found particles
    def estimate_z_positions(self, image, parts):
        self._image = image

        factor = (2*np.pi/(self._lamda/self._n_m))
        img_mean = np.mean(self._image)
        len_parts = len(parts)
        if len_parts == 0:
            return []
        if len_parts == 1:
            estimates = self._estimate_z_position(parts[0], factor, img_mean, parts)
            return [estimates]

        parts.sort()
        estimates = []
        for part in parts:
            res = self._estimate_z_position(part, factor, img_mean, parts)
            if res != None:
                estimates.append(res)

        # debug plot
        if ('full' in self._debug_str or '3D' in self._debug_str or 'save' in self._debug_str):
            self._debug.debug_stage(estimates, '3D estimated candidates', self._image, self._debug_str)

        return estimates