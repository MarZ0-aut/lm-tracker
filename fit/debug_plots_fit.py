# import necessary modules and functions
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Debugging_Fit(object):
    
    # init definiton and theory parameters
    def __init__(self, tuning, theory, holo):
        self._dx = theory["dx"]
        self._n_p = theory["n_p"]
        self._d_p = theory["d_p"]
        self._computation_mode = tuning["comp_mode"]
        self._components = tuning["components"]
        
        self._holo = holo
        self._parameters = tuning

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
        
    # debug for plus-minus determination
    def debug_pm(self, img, pi, mi, pm, mm, mask, sign, debug_str, pos):
        extent, plotdata = self._debug_help(img)
        xt, yt, tx, ty = plotdata

        fig = plt.figure(figsize=(6,2))
        ax0 = fig.add_subplot(131)
        ax1 = fig.add_subplot(132)
        ax2 = fig.add_subplot(133)

        ax0.set_xticklabels(xt)
        ax1.set_xticklabels(xt)
        ax2.set_xticklabels(xt)
        ax0.set_yticklabels(yt)
        ax1.set_yticklabels(yt)
        ax2.set_yticklabels(yt)
        ax0.set_xticks(tx)
        ax1.set_xticks(tx)
        ax2.set_xticks(tx)
        ax0.set_yticks(ty)
        ax1.set_yticks(ty)
        ax2.set_yticks(ty)

        ax0.set_title(str(round(mm, 2))+", "+str(-1*sign))
        ax1.set_title("Real data")
        ax2.set_title(str(round(pm, 2))+", "+str(sign))

        ax0.imshow(mi*mask, extent=extent)
        ax1.imshow(img*mask, extent=extent)
        ax2.imshow(pi*mask, extent=extent)
        if "save" in debug_str:
            timestr, datestr = self._debug_folder()
            name = "plots\\"+datestr+"\\"+timestr+"_"+str(np.array(np.round(pos,0), dtype=int))+"_pm"+".png"
            plt.savefig(name, dpi=400)
            print("Figure", name, "created.")
        if "full" in debug_str or "pm" in debug_str:
            plt.show()
            plt.close()
        else:
            plt.close();
        
    # debug print for opencl fit
    def debug_opencl(self, mimg, pd, px, py, pz, alpha, phi, theta, psi, X, Y, NA, LC,
                     debug_str, offsetx=0, offsety=0):
        wid = 7
        offsetx = int(round(offsetx, 0))
        offsety = int(round(offsety, 0))
        
        if "full" in debug_str or "fit" in debug_str or "resid" in debug_str:
            print("d:%5.2f  x:%5.2f  y:%5.2f  z:%5.2f  alpha:%5.2f "%(pd, px, py,pz, alpha))
            print("phi:%6.2f theta:%6.2f psi:%6.2f"%(phi, theta, psi))
        # pass shape to holo class
        self._holo.set_shape(mimg.shape, False, False, self._computation_mode)
        # calculate fitted image
        fit_image = self._holo.calcholo([[pd, px, py, pz, self._n_p, alpha, phi, theta, psi, 0]],
                                  components=self._components)
        # calculate a combined image

        ly, lx = mimg.shape
        comb_image = np.zeros((ly,lx))
        comb_image[0:ly//2, 0:lx//2] = fit_image[0:ly//2, 0:lx//2]
        comb_image[ly//2:, lx//2:] = fit_image[ly//2:, lx//2:]
        comb_image[0:ly//2, lx//2:] = mimg[0:ly//2, lx//2:]
        comb_image[ly//2:, 0:lx//2] = mimg[ly//2:, 0:lx//2]

        extent, plotdata = self._debug_help(comb_image)
        ranx = np.arange(-lx/2,lx/2, 1)*self._dx
        rany = np.arange(-ly/2,ly/2, 1)*self._dx

        # show the image
        fig = plt.figure(figsize=(12, 4))
        ax11 = fig.add_subplot(221)
        ax12 = fig.add_subplot(223, sharex=ax11)
        ax13 = fig.add_subplot(122)

        ax13.set_xticks(plotdata[2])
        ax13.set_xticklabels(plotdata[0])
        ax13.set_yticks(plotdata[3])
        ax13.set_yticklabels(plotdata[1])
        ax13.set_title("Combined image ["+str(int(X))+", "+str(int(Y))+"]")
        ax13.set_xlabel("Position X / µm")
        ax13.set_ylabel("Position Y / µm")
        ax11.set_xlabel("Position / µm")
        ax12.set_xlabel("Position / µm")
        ax11.set_title("Cross section X=0")
        ax12.set_title("Cross section Y=0")

        radius = self._d_p/self._dx
        ax13.imshow(comb_image, extent=extent, cmap='gray', vmin=0, vmax=2.5)
        if offsetx == 0:
            circ = patches.Circle((px/self._dx, -py/self._dx), radius, linewidth=1,edgecolor='r',facecolor='none')
            ax13.add_patch(circ)
        ax11.plot(ranx, np.mean(mimg[ly//2+offsety-wid:ly//2+offsety+wid,:], axis=0), label="observed")
        ax11.plot(ranx, np.mean(fit_image[ly//2+offsety-wid:ly//2+offsety+wid,:], axis=0), label="fitted")
        ax12.plot(rany, np.mean(mimg[:, lx//2+offsetx-wid:lx//2+offsetx+wid], axis=1)[::-1], label="observed")
        ax12.plot(rany, np.mean(fit_image[:, lx//2+offsetx-wid:lx//2+offsetx+wid], axis=1)[::-1], label="fitted")
        if offsetx == 0:
            ax12.plot([radius*self._dx, radius*self._dx], [0, 2], "r-", lw=2)
            ax12.plot([-radius*self._dx, -radius*self._dx], [0, 2], "r-", lw=2)
            ax11.plot([radius*self._dx, radius*self._dx], [0, 2], "r-", lw=2)
            ax11.plot([-radius*self._dx, -radius*self._dx], [0, 2], "r-", lw=2)
        ax11.legend(loc="lower right")
        ax12.legend(loc="lower right")
        fig.tight_layout(pad=0.5)
            
        if "save" in debug_str:
            timestr, datestr = self._debug_folder()
            name = "plots\\"+datestr+"\\"+timestr+"_"+str(np.array(np.round([X, Y],0), dtype=int))+"_fit"+".png"
            plt.savefig(name, dpi=400)
            print("Figure", name, "created.")
        if 'full' in debug_str or 'fit' in debug_str:
            plt.show()
            plt.close()
        else:   
            plt.close();

        # residual image
        fig1, ax_1 = plt.subplots(1, 1)
        ax_1.set_xticks(plotdata[2])
        ax_1.set_xticklabels(plotdata[0])
        ax_1.set_yticks(plotdata[3])
        ax_1.set_yticklabels(plotdata[1])
        ax_1.set_title("Residual image")
        ax_1.set_xlabel("Position X / µm")
        ax_1.set_ylabel("Position Y / µm")
        h1 = ax_1.imshow(mimg/fit_image, extent=extent, cmap='gray')
        plt.colorbar(h1, ax=ax_1)
        if "save" in debug_str:
            timestr, datestr = self._debug_folder()
            name = "plots\\"+datestr+"\\"+timestr+"_"+str(np.array(np.round([X, Y],0), dtype=int))+"_resid"+".png"
            plt.savefig(name, dpi=400)
            print("Figure", name, "created.")
        if 'full' in debug_str or 'resid' in debug_str:
            plt.show()
            plt.close()
        else:
            plt.close(); 