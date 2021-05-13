# import necessary modules and functions
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Debugging_Axial(object):
    # init definiton and theory parameters
    def __init__(self, theory):
        self._dx = theory["dx"]

    # debug folder
    def _debug_folder(self):
        # create timestring
        timestr = str(datetime.datetime.now())
        time_str = timestr[11:13]+'-'+timestr[14:16]+'-'+timestr[17:19]
        date_str = timestr[:10]
        # create folder to safe images:
        if "plots" not in os.listdir():
            os.mkdir("plots")
        if date_str not in os.listdir("plots"):
            os.mkdir("plots\\"+date_str)
            print("Folder", "plots\\"+date_str, "created.")
        return time_str, date_str
        
    # debug help function
    def _debug_help(self, img):
        ny, nx = img.shape
        ny /= 2
        nx /= 2
        stepx, stepy = nx/2, ny/2

        ticklistx = np.arange(-nx, nx+stepx, stepx)
        ticksx = np.array(np.round(ticklistx*self._dx,0), dtype=int)
        ticklisty = np.arange(-ny, ny+stepy, stepy)
        ticksy = np.array(np.round(ticklisty*self._dx,0), dtype=int)
        extent = [-nx, nx, -ny, ny]
        return extent, (ticksx, ticksy, ticklistx, ticklisty)

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
        
    # debug function for spectral correction
    def debug_spectrum(self, fs, spec, corr, orig, sig, phi, tupl, debug_str, pos):
        lo, hi = tupl
        ran1 = (np.arange(0, len(orig), 1))*self._dx
        ran2 = (np.arange(0, len(sig), 1)[lo:hi])*self._dx

        fig = plt.figure(figsize=(12, 3))
        ax0 = fig.add_subplot(131)
        ax1 = fig.add_subplot(132)
        ax2 = fig.add_subplot(133)

        ax0.set_title("Signal spectrum")
        ax0.set_xlabel("Spatial frequency / µm")
        ax1.set_title("Radial signal")
        ax1.set_xlabel("Position / µm")
        ax2.set_title("Reconstructed phase")
        ax2.set_xlabel("Position / µm")

        ax0.plot(fs, spec, label="in")
        ax0.plot(fs, corr, label="out")
        ax1.plot(ran1, orig, label='in')
        ax1.plot(ran1, sig, label='out')
        ax2.plot(ran2, phi)

        ax0.legend()
        ax1.legend()
        if "save" in debug_str:
            timestr, datestr = self._debug_folder()
            name = "plots\\"+datestr+"\\"+timestr+"_"+str(np.array(np.round(pos,0), dtype=int))+"_spec"+".png"
            plt.savefig(name, dpi=400)
            print("Figure", name, "created.")
        if "full" in debug_str or "filter" in debug_str:
            plt.show()
            plt.close()
        else:
            plt.close();
        
    # debug plot for envelope function
    def debug_envelope(self, peaks, rad, fit, chi, sign, sig, debug_str, pos):
        ran = peaks*self._dx

        fig = plt.figure()
        ax0 = fig.add_subplot(111)
        ax0.plot(ran, sig[peaks], "kx")
        ax0.plot(np.arange(len(sig[:peaks[-1]]))*self._dx, sig[:peaks[-1]])
        ax0.plot(ran, fit, "r", label="Chi: "+str(round(chi, 2))+", Sign: "+str(sign))
        ax0.set_xlabel("Radial distance / µm")
        ax0.set_title("Radial peak shape")
        plt.legend()
        if "save" in debug_str:
            timestr, datestr = self._debug_folder()
            name = "plots\\"+datestr+"\\"+timestr+"_"+str(np.array(np.round(pos,0), dtype=int))+"_env"+".png"
            plt.savefig(name, dpi=400)
            print("Figure", name, "created.")
        if "full" in debug_str or "envelope" in debug_str:
            plt.show()
            plt.close()
        else:
            plt.close();

    # debug plot for envelope function
    def debug_interference(self, arr1, arr2, value, debug_str, pos):
        fig = plt.figure()
        ax0 = fig.add_subplot(111)
        ax0.plot(arr1, label="horizontal")
        ax0.plot(arr2, label="vertical")
        ax0.set_title("Horizontal/Vertical comparison "+str(value))
        ax0.legend()
        if "save" in debug_str:
            timestr, datestr = self._debug_folder()
            name = "plots\\"+datestr+"\\"+timestr+"_"+str(np.array(np.round(pos,0), dtype=int))+"_IF-check"+".png"
            plt.savefig(name, dpi=400)
            print("Figure", name, "created.")
        if "full" in debug_str or "interf" in debug_str:
            plt.show()
            plt.close()
        else:
            plt.close();

    # debug plot for instantaneous phase and signal
    def debug_intensity(self, rad, sig, out, pos, phi, fit, image, debug_str):
        width = self._calc_max_width(image, [pos[1], pos[0]], wi=len(sig)/2)
        img = image[pos[1]-width[0]:pos[1]+width[0], pos[0]-width[1]:pos[0]+width[1]]

        extent, plotdata = self._debug_help(img)
        ran = np.arange(0, len(sig), 1)*self._dx

        fig = plt.figure(figsize=(10,3))
        ax0 = fig.add_subplot(131)
        ax1 = fig.add_subplot(132)
        ax2 = fig.add_subplot(133)
        ax3 = ax2.twinx()

        ax0.set_xticks(plotdata[2])
        ax0.set_xticklabels(plotdata[0])
        ax0.set_yticks(plotdata[3])
        ax0.set_yticklabels(plotdata[1])

        ax0.set_title("Evaluated image "+str(pos))
        ax0.set_xlabel("Position X / µm")
        ax0.set_ylabel("Position Y / µm")
        ax1.set_title("Reconstructed phase")
        ax1.set_xlabel("Position / µm")
        ax3.set_title("Radial signal")
        ax3.set_xlabel("Position / µm")

        ax0.imshow(img, extent=extent)
        ax1.plot(rad, phi, label="experimental")
        ax1.plot(rad, fit, label="fit")
        ax3.plot(ran, sig, label="experimental")
        ax3.plot(ran, out, label="corrected")

        ax1.legend()
        ax3.legend()
        if "save" in debug_str:
            timestr, datestr = self._debug_folder()
            name = "plots\\"+datestr+"\\"+timestr+"_"+str(np.array(np.round(pos,0), dtype=int))+"_phase"+".png"
            plt.savefig(name, dpi=400)
            print("Figure", name, "created.")
        if "full" in debug_str or "phase" in debug_str:
            plt.show()
            plt.close()
        else:
            plt.close();

    # found candidates debug
    def debug_stage(self, pos, string, img, debug_str):
        fig = plt.figure(dpi=100)
        ax = fig.add_subplot(111)
        ax.set_title("Stage: "+string)
        ax.set_xlabel("Position X / px")
        ax.set_ylabel("Position Y / px")
        ax.imshow(img)
        if len(pos)>0:
            for i in range(len(pos)):
                # Create a circular patch
                wid= 30
                circ = patches.Circle((pos[i][0],pos[i][1]),wid/4,linewidth=1,edgecolor='r',facecolor='none')
                # Add the patch to the Axes
                ax.add_patch(circ)
        if "save" in debug_str:
            timestr, datestr = self._debug_folder()
            name = "plots\\"+datestr+"\\"+timestr+"_"+string+".png"
            plt.savefig(name, dpi=400)
            print("Figure", name, "created.")
        if "full" in debug_str or "3D" in debug_str:
            plt.show()
            plt.close()
        else:
            plt.close();