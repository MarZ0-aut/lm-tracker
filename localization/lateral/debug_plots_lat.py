# import necessary modules and functions
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Debugging_Lateral(object):
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

    # debug symmetry transform
    def debug_symmetry_trafo(self, ST0, ST1, img, pos, string, debug_str):
        fig = plt.figure(figsize=(9, 4))
        ax0 = fig.add_subplot(121)
        ax1 = fig.add_subplot(122)

        ax0.set_title("Symmetry transformation")
        ax0.set_xlabel("Position / µm")
        ax0.set_ylabel("Autocorrellation value")
        ax1.set_title("Evaluated image "+string+str(pos))
        ax1.set_xlabel("Position X / µm")
        ax1.set_ylabel("Position Y / µm")

        ContrX = round(np.amax(ST0)/np.mean(ST0)/len(ST0), 3)
        ContrY = round(np.amax(ST1)/np.mean(ST1)/len(ST1), 3)
        ax0.plot(ST0, label="X, ContrX: "+str(ContrX))
        ax0.plot(ST1[::-1], label="Y, ContrY: "+str(ContrY))
        ax1.imshow(img)
        ax0.legend(loc="upper right")
        
        if "save" in debug_str:
            timestr, datestr = self._debug_folder()
            name = "plots\\"+datestr+"\\"+timestr+"_"+string+str(pos)+".png"
            plt.savefig(name, dpi=400)
            print("Figure", name, "created.")
        if "full" in debug_str or "ST" in debug_str:
            plt.show()
            plt.close()
        else:
            plt.close();
        
        # debug grid
    def debug_grid(self, imgs, debug_str, string):
        fig = plt.figure(dpi=75)
        if len(imgs)==0:
            plt.close()
            pass
        elif len(imgs)==1:
            ax = fig.add_subplot(111)
            ax.set_title("Single image from grid")
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.imshow(imgs[0])
        else:
            # definitions
            tf = len(imgs)
            fpr = int(tf**(1/2))
            dx = 1/fpr
            exr = tf%fpr
            if not exr==0:
                exr=1
            tnr = (tf//fpr)+exr
            dy = 1/tnr
            # set all axes
            axs = []
            for j in range(tf//fpr):
                for i in range(fpr):
                    ax = fig.add_axes([(j*dx), (1-i*dy), dx, dy], label=str(i)+str(j)+string)
                    ax.axes.get_xaxis().set_visible(False)
                    ax.axes.get_yaxis().set_visible(False)
                    axs.append(ax)
            for i in range(tf%fpr):
                ax = fig.add_axes([(i*dx), (1-dy), dx, dy], label=str(i)+str(j)+string)
                ax.axes.get_xaxis().set_visible(False)
                ax.axes.get_yaxis().set_visible(False)
                axs.append(ax)
            # write images into axes
            for i in range(tf):
                axs[i].imshow(imgs[i])
                
        if "save" in debug_str:
            timestr, datestr = self._debug_folder()
            name = "plots\\"+datestr+"\\"+timestr+"_"+string+".png"
            plt.savefig(name, dpi=400)
            print("Figure", name, "created.")
        if "full" in debug_str or "grid" in debug_str:
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
        if ("full" in debug_str or "search" in debug_str or "sorted" in debug_str or "2D" in debug_str
            or "loc" in debug_str):
            plt.show()
            plt.close()
        else:
            plt.close();