# import necessary modules and functions
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt

class Debugging_Simulation(object):
    
    # init definiton and theory parameters
    def __init__(self, theory):
        self._dx = theory["dx"]
        
    # create timestring
    def debug_fit(self, img, mask):
        timestr = str(datetime.datetime.now())
        time_str = timestr[11:13]+'-'+timestr[14:16]+'-'+timestr[17:22]
        date_str = timestr[:10]
        # create folder to safe images:
        if "plots" not in os.listdir():
            os.mkdir("plots")
        if date_str not in os.listdir("plots"):
            os.mkdir("plots\\"+date_str)
        
        ny, nx = img.shape
        ny /= 2
        nx /= 2
        stepx, stepy = nx/2, ny/2
    
        ticklistx = np.arange(-nx, nx+stepx, stepx)
        ticksx = np.array(np.round(ticklistx*self._dx,0), dtype=int)
        ticklisty = np.arange(-ny, ny+stepy, stepy)
        ticksy = np.array(np.round(ticklisty*self._dx,0), dtype=int)
        extent = [-nx, nx, -ny, ny]
    
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xticks(ticklistx)
        ax.set_xticklabels(ticksx)
        ax.set_yticks(ticklisty)
        ax.set_yticklabels(ticksy)
        ax.set_title("Image (to be fitted) + mask")
        ax.set_xlabel("X position / µm")
        ax.set_ylabel("Y position / µm")
        h = ax.imshow(img*mask, extent=extent, cmap="gray")
        plt.colorbar(h)
        plt.savefig("plots\\"+date_str+"\\"+time_str+".png", dpi=400)
        plt.show()
        
        plt.close()