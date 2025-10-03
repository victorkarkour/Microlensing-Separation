# import astropy.constants as ac
import numpy as np
# import scipy.signal as signal
# import scipy.optimize as sc
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import time
from multiprocessing import Pool
import gc
import sys
# import math
# from collections import Counter
from scipy.stats import gamma
from itertools import repeat
from Sep_gen import Sep_gen
import click
import os
import matplotlib.gridspec as gridspec

matplotlib.use("Agg")
class Sep_plot(Sep_gen):

    def __init__(self, numestep = 10, numdiv = 2):# which = "Log", wnum = 10, inum = 10):
        self.numestep = numestep
        self.numdiv = numdiv
        # self.which = which
        # self.wnum = wnum
        # self.inum = inum

    def DataProj(self, w = 0, start = 0.5, end = 20, step = 0.5):
        """
        
        """
        # List initialization
        listt = []
        totlist = []
        totparam = []
        # Parameter list for MultiPlot
        param = [
        # Row 1
        (0. , 0., w, end, step, start), (0., np.pi/6, w, end, step, start), (0., np.pi/3, w, end, step, start), (0., np.pi/2, w, end, step, start),
        # Row 2
        (0.5, 0., w, end, step, start), (0.5, np.pi/6, w, end, step, start), (0.5, np.pi/3, w, end, step, start), (0.5, np.pi/2, w, end, step, start),
        # Row 3
        (0.9, 0., w, end, step, start), (0.9, np.pi/6, w, end, step, start), (0.9, np.pi/3, w, end, step, start), (0.9, np.pi/2, w, end, step, start)
            
        ]
        
        start = time.perf_counter()
        
        with Pool(processes = 3) as pool:
            result = pool.map(self.WorkProj, param)
        
        
        # Test source
        # totlist, totparam, listt = WorkProj(param[0])
        
        # Splits the total result into different results
        for i in range(len(result)):
            totlist.append(result[i][0])
            totparam.append(result[i][1])
            listt.append(result[i][2])
        
        end_time = time.perf_counter()
        totaltime = end_time - start
        print(f"Time to Compute was {totaltime:.4f} seconds.")
                
        return totlist, totparam, listt

    def WorkProj(self, param):
        """
        """
        list1 = []
        listt = []
        paramlist = []
        
        # Takes data from DataProj
        e, i, w, end, step, start = param
        
        stepthrough = np.arange(start, end + step, step)
        
        # For each value in the stepthrough, calculates orbit
        for val in stepthrough:
                x, y, t = Sep_gen.OrbGeoAlt(a = val, e = e, i = i ,w = w)
                
                # Appends results of orbital projection
                list1.append((x,y))
                listt.append(t)
        
        return list1, param, listt

    @classmethod
    def DataHist(cls, w = 0, step = 0.002, end = 10, which = "Linear", inclination = False, istep = None, estep = []):
        """
        """
        # Goes from both Linear and Log calculations to just Linear
        Linear = which
        
        if inclination and len(estep) == 0:
            param = [
                # Row 1
                (0.1, istep, w, end, step, Linear, inclination), (0.40, istep, w, end, step, Linear, inclination), (0.7, istep, w, end, step, Linear, inclination),
                # Row 2     
                (0.2, istep, w, end, step, Linear, inclination), (0.50, istep, w, end, step, Linear, inclination), (0.8, istep, w, end, step, Linear, inclination),     
                # Row 3     
                (0.3, istep, w, end, step, Linear, inclination), (0.6, istep, w, end, step, Linear, inclination), (0.9, istep, w, end, step, Linear, inclination)
                
                     ]
        elif inclination:
            param = [estep, istep, w, end, step, Linear, inclination]
        else:
            # Parameters for Paralellization
            param = [
            # Row 1
            (0. , 0., w, end, step, Linear, inclination), (0., np.pi/6, w, end, step, Linear, inclination), (0., np.pi/3, w, end, step, Linear, inclination), (0., np.pi/2, w, end, step, Linear, inclination),
            # Row 2
            (0.5, 0., w, end, step, Linear, inclination), (0.5, np.pi/6, w, end, step, Linear, inclination), (0.5, np.pi/3, w, end, step, Linear, inclination), (0.5, np.pi/2, w, end, step, Linear, inclination),
            # Row 3
            (0.9, 0., w, end, step, Linear, inclination), (0.9, np.pi/6, w, end, step, Linear, inclination), (0.9, np.pi/3, w, end, step, Linear, inclination), (0.9, np.pi/2, w, end, step, Linear, inclination)
                
            ]
        
        
        start = time.perf_counter()
        # Multi Processing
        if inclination == True and len(estep) != 0:
            totlist = Sep_gen.Rchange(param = param)
        else:
            with Pool(processes = 6) as pool:
                totlist = pool.map(Sep_gen.Rchange, param)
        # else:
        #     totlist = Rchange(param)
        end_time = time.perf_counter()
        totaltime = end_time - start
        print(f"Time to Compute was {totaltime:.4f} seconds.")
        


        # totlist = [
        #     list1,list4,list7, list10,
        #     list2,list5,list8, list11,
        #     list3,list6,list9, list12
        # ]
        
        return totlist, param

    def MultiPlotProj(self, w = 0, start = 0.5, end = 20, step = 0.5):
        """
        Creates a 3 by 4 plot of 20 planetary orbits each with varying semimajor axes
        according to differing parameters.
        
        --------
        ### Parameters
        
        t0 : float <br>
            time at periapsis of planet's orbit
            
        a : integer <br>
            semimajor axis 
        
        w : float (in radians) <br>
            argument of periapsis
            
        W : float (in radians) <br>
            longitude of ascending node
        
        i : float (in radians) <br>
            inclination angle
            
        e : integer <br>
            eccentricity
            
        n : integer <br>
            number of figures in the plot
            (will likely remove)
        -------
        ### Returns 
        
        3 by 4 Plot of Planetary Orbits
        
        """
        
        # Initialize Lists
        listt = []
        rlist = []
        totlist = []
        
        # Gets everything ready for multiprocessing of orbital projections
        list, totparam, listt = self.DataProj(w = w, start = start, end = end, step = step)
        rect = dict(boxstyle = "round", alpha = 0.5, facecolor = "white")        
        
        fig, axs = plt.subplots(3,4, figsize = (13,9), gridspec_kw = {"hspace" : 0, "wspace" : 0}, sharex = False, sharey = False)
        # Makes the 2 x 2 figure for the 3 x 4 figures to go into
        # fig.suptitle("Orbital Projection with Alterations in e, i, and "r"From $\omega$ 0 to $\ \frac{\pi}{2}$", x = 0.49, y = 0.99)
    
        # Iterates through each subplot in the 3x4 figure
        for j, ax  in enumerate(axs.flatten()):
            
            # Takes the first data set in the list
            # These contain 12 other data sets  
            iterlist = list[j]
            param = totparam[j]
            
            # Finds points <= 0.01 for each projection
            rtemp, xchange, ychange, rtemp_log, temp = Sep_gen.Rchange(param, coords = True)
            
            vel = Sep_gen.Velocity(param)
            # Iterates through the data clump to access
            # the data set 
            for g in range(len(iterlist)):
                # This contains each data set in the data clump
                initialx, initialy = iterlist[g]
                # Calculates the velocity of each data point in the data set
                
                rlist.append(rtemp)
                # print("Last Velocity Output: ", vel[-1])
                velmax = np.max(vel[g])
                # # IMPORTANT!!!!!
                velmin = 0.1
                
                dot = Sep_gen.DotSize(vel[g],velmax,velmin)
                # print("Last Dot Size Output: ", dot[-1])
                
                # Determines the colors of each data set according
                # to its positioning
                colorlist = ["forestgreen", "tomato", "mediumblue", "orange", "purple",
                                "pink", "blue", "red", "green", "cyan"]
                if end == 1.25:
                    if g == 0:
                        label = "a = 0.75"
                        color = colorlist[g]
                    elif g == 1:
                        label = "a = 1.0"
                        color = colorlist[g]
                    else:
                        label = "a = 1.25"
                        color = colorlist[g]
                elif end == 1.5:
                    if g == 0:
                        label = "a = 0.5"
                        color = colorlist[g]
                    elif g == 1:
                        label = "a = 1.0"
                        color = colorlist[g]
                    else:
                        label = "a = 1.5"
                        color = colorlist[g]
                else:
                    rangelist = np.arange(0.5,end+0.5,0.5)
                    alpha = rangelist[g]
                    label = f"a = {alpha}"
                    color = colorlist[g % 10]   
                
                # Plots the data set, including the dot size according to velocity        
                dataproj = ax.scatter(initialx, initialy, s=dot, color= color, label=label)
                # Also includes points at which |r-r0| <= 0.01
                data = ax.scatter(xchange[g], ychange[g], s = 8, color = "yellow")
                
            # Creates the grid for each plot
            ax.grid(True,color = "grey", linestyle="--", linewidth="0.55", axis = "both", which = "both")
                
            # Plots an Einstein Ring Radius of 1 around each plot
            Circ2 = patches.Circle((0,0), 1, ec="k", fill=False, linestyle = ":", linewidth = 1)
            
            # Adds the Circle to the plot
            ax.add_patch(Circ2)
            
            # # Just grabs the labels for each plot just before it iterates through again
            # if j == 0:
            #     handles, labels = ax.get_legend_handles_labels()
                
            # Limits
            ax.set_xlim(-2,2)
            ax.set_ylim(-2,2)
                
            textstr = "\n".join((f'e = {param[0]}', f'i = {round(param[1],2)}'))
            ax.text(0.80, 0.95, textstr, transform = ax.transAxes, fontsize = 10, verticalalignment = "top", bbox = rect)
                
            if j % 4 == 0:
                ax.tick_params(axis = "both", labelbottom = False, labelleft = False)
            elif j < 8 or j >= 9:
                # ax.set_yticks([])
                # ax.set_xticks([])
                ax.tick_params(axis = "both", labelbottom = False, labelleft = False)
            if j == 11:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles[0:9],labels[0:9], loc = "upper left", fontsize = "x-small",prop = {"size":8}, borderpad = 0.5, labelspacing = 0.75, handlelength = 2, framealpha = 0.5)
        
        # Shows the legend of each data point
        # fig.legend(handles[0:5],labels[0:5],fontsize="small", prop = {"size":8}, borderpad = 0.5, labelspacing = 0.75, handlelength = 2, framealpha = 0.5)
        # fig.tight_layout(pad=0.05, rect = (0.02, 0.02, 0.95, 0.95))
        fig.tight_layout()
        
        # Saves to Figure Folder
        plt.savefig(f"/College Projects/Microlensing Separation/Figures/Multi_a05_{end}_omega_0.png")
        # plt.savefig(f"C:/Users/victo/College Projects/Microlensing Separation/Figures/Multi_a05_{end}_omega_0.png")
        # plt.show()
        
        return rlist

    def MultiPlotHist(self, w = 0, step = 0.002, end = 10, which = "Log"):
        """
        """
        
        # totlinlist = [[] for _ in range(12)]
        # totloglist = [[] for _ in range(12)]
        # totlinsemilist = [[] for _ in range(12)]
        
        if which == "Linear":
            colorlist = ["black", "green"]
        else:
            colorlist = ["black", "red", "blue", "green"]
        
        # Data
        rlist, param = self.DataHist(w = w, step = step, end = end, which = which) 
        rect = dict(boxstyle = "round", alpha = 0.5, facecolor = "white")
        
        
        fig, axs = plt.subplots(3,4, figsize = (13,9), gridspec_kw = {"hspace" : 0, "wspace" : 0}, sharex = False, sharey = False) 
        # if which == "Linear":
        #     fig.suptitle("Detections of $R_E$ with Alterations in e, i, and "r"$\omega$ = 0" f" ({which})", x = 0.49, y = 0.99)
        # else:
        #     fig.suptitle("Detections of $R_E$ with Alterations in e, i, and "r"$\omega$ = 0" f"\n (For Linear, Log, & Power Law)", x = 0.49, y = 0.99)
        
        # Iterates through each subplot in the 3x4 figure 
        for j, ax  in enumerate(axs.flatten()):
            # xticks = ax.xaxis.get_major_ticks()
            # xticks.lable1.set_visible(False)
            
            steplindict, x, y, steplogdict, steplinsemidict = rlist[j]
            
            iterparam = param[j]
            totliniter = steplindict
            totlinitersemi = steplinsemidict
            totlogiter = steplogdict
            
            totlinlist = [key for key, val in totliniter.items() for _ in range(val)]
            totlinsemilist = [key for key, val in totlinitersemi.items() for _ in range(val)]
            totloglist = [key for key, val in totlogiter.items() for _ in range(val)] 
            
            # Create variables for bin sizes
            nbin = 200
            amin = 0.5
            amax = 21
            # Make logbinsizes for all
            logbinsize = (np.log10(amin)-np.log10(amax))/nbin
            
            weights_lin = np.abs(np.ones_like(totlinlist) / (len(totlinlist) * logbinsize))
            weights_linsemi = np.abs(np.ones_like(totlinsemilist) / (len(totlinsemilist) * logbinsize))    
            weights_log = np.abs(np.ones_like(totloglist) / (len(totloglist) * logbinsize))
            
            # Creates the log spaced bins for our data
            # Stupid fix to stupid problems :)
            if j == 0 and which == "Linear":
                logbins_lin = np.linspace(amin,amax,1000+1)
            else:
                if j == 0:
                    logbins_lin = np.linspace(amin,amax,nbin)
                    logbins_log = np.geomspace(amin,amax,nbin)
                else:    
                    logbins_lin = np.geomspace(amin,amax, nbin)
                    logbins_log = np.geomspace(amin,amax,nbin)
            if which == "Linear":
                
                datahist_lin, bins, patches_lin = ax.hist(
                    totlinlist, bins=logbins_lin, range=(0.5, end+0.5),
                    stacked=True, histtype="step",
                    weights=weights_lin, label = "Linear"
                )
            else:
                datahist_lin, bins, patches_lin = ax.hist(
                    totlinlist, bins=logbins_lin, range=(0.5, end+0.5),
                    stacked=True, histtype="step", alpha = 0.75, edgecolor = "black",
                    weights=weights_lin, fc = "none", label = "Linear"
                )
                datahist_log, bins, patches_log = ax.hist(
                    totloglist, bins=logbins_log, range=(0.5, end+0.5),
                    stacked=True, histtype="step", alpha = 0.75, edgecolor = "red",
                    weights=weights_log, fc = "none", label = "Log"
                )
                datahist_linsemi , bins, patches_linsemi = ax.hist(
                    totlinsemilist, bins=logbins_lin, range=(0.5, end+0.5),
                    stacked=True, histtype="step", alpha = 0.40, edgecolor = "blue",
                    weights=weights_linsemi, fc = "none", label = "Linear / a"
                )
            if which == "Linear":
                for patch in patches_lin:
                    patch.set_edgecolor("k")
            else:
                for patch in patches_lin:
                    patch.set_edgecolor("k")
                for patch in patches_log:
                    patch.set_edgecolor("r")
                for patch in patches_linsemi:
                    patch.set_edgecolor("b")
            ax.set_xlim(0.5,20.5)
            ax.set_ylim(0,10)
            ax.set_xscale("log")
            
            textstr = "\n".join((f'e = {iterparam[0]}', f'i = {round(iterparam[1],2)}'))
            ax.text(0.63, 0.95, textstr, transform = ax.transAxes, fontsize = 10, verticalalignment = "top", bbox = rect)
            # ax.text(3e0, 8.5, f"$e = {iterparam[0]}$") 
            # ax.text(3e0, 7.5, f"$i = {round(iterparam[1],2)}$")
            ax.grid(True,color = "grey", linestyle="--", linewidth="0.25", axis = "x", which = "both")
            ax.vlines(1/(1-iterparam[0]), 0, 1, transform = ax.get_xaxis_transform(), colors = 'green', alpha = 0.75, label = "Peak Eccentricity")
            if j % 4 == 0:
                ax.tick_params(axis = "both", labelbottom = False, labelleft = False)
            elif j < 8 or j >= 9:
                ax.set_yticks([])
                ax.set_xticks([])
                ax.tick_params(axis = "x", labelbottom = False)
            
            if j == 11:
                handles = [patches.Rectangle((0,0),1,1,color = c, ec = "w") for c in colorlist]
                if which == "Linear":
                    labels = ["Linear", "Peak Eccentricity"]
                else:
                    labels = ["Linear", "Log", r"Power Law: $\alpha = 2$", "Peak Eccentricity"]
                    
                ax.legend(handles = handles, labels = labels, loc = "best", fontsize = "small")
                
        # fig.tight_layout(pad=1.25,h_pad=0, w_pad=0, rect = (0.08, 0.0, 0.95, 0.95))
        fig.tight_layout()
        if which == "Linear":
            plt.savefig(f'/College Projects/Microlensing Separation/Figures/MultiHist_omega_0_0002_{which}.png')
        else:
            plt.savefig(f'/College Projects/Microlensing Separation/Figures/MultiHist_omega_0_0002_{which}.png')
        # plt.show()
        return rlist

    @staticmethod
    def CompletePlotHist(param):
        """
        
        """
        step, end, inclination, which, estep_outer, inum, wnum = param
        
        if inclination == False:
            colorlist = ["black", "red"]
            labels = ["Linear", "Log"]
        else:
            if which == "Log":
                colorlist = ["red"]
                labels = ["Log"]
            elif which == "Linear":
                colorlist = ["black"]
                labels = ["Linear"]
            else:
                colorlist = ["blue"]
                labels = ["Linear / a"]
        # Dictionary for storing Rchange results
        totlinlist = []
        totloglist = []
        if len(estep_outer) == 0:
             tothistlist =[[] for _ in range(9)]
        else:
            tothistlist = []
        evalhistlist = []
        
        # Create variables for bin sizes
        nbin = 200
        amin = 0.5
        amax = 21
        # Make logbinsizes for all
        logbinsize = (np.log10(amin)-np.log10(amax))/nbin
        logbins = np.geomspace(amin,amax, nbin)
        
        print(f"start time {time.time()}, {estep_outer}")
        
        # For making the stepthrough of omega
        wstep = np.linspace(0,np.pi/2,wnum)
        if inclination:
            # REMEMBER TO REMOVE IF STATMENTS FOR LINEAR (will eventually want linear in both)
            cosstep = np.linspace(0,1,inum)
            istep = np.arccos(cosstep)
            # Only works if estep_outer has values in the list
            if len(estep_outer) != 0:
                estep = estep_outer
        for k in wstep:
            print("Value of omega currently: ", k, " and current position in array: ", np.where(wstep == k))
            # Each omega calculates its own data groups
            if inclination and len(estep_outer) != 0:
                steptotlist, param = Sep_plot.DataHist(w = k, step = step, end = end, which = which, inclination = inclination, istep = istep, estep = estep)
            elif inclination:
                steptotlist, param = Sep_plot.DataHist(w = k, step = step, end = end, which = which, inclination = inclination, istep = istep)
            else:
                steptotlist, param = Sep_plot.DataHist(w = k, step = step, end = end, which = which)
            # Once complete, takes the data through each set
            if inclination == False:
                for j in range(len(steptotlist)):
                    # steplinlist, x, y, steploglist = steptotlist[j]
                    if inclination:
                        steplindict, x, y, steplogdict, blank = steptotlist
                    else:
                        steplindict, x, y, steplogdict = steptotlist[j]
                    # Adds new counts onto previous counts
                    totliniter = steplindict
                    totlogiter = steplogdict
                
                    totlinlist[j] = [key for key, val in totliniter.items() for _ in range(val)]
                    totloglist[j] = [key for key, val in totlogiter.items() for _ in range(val)]
                    
                    weights_log = np.abs(np.ones_like(totloglist[j]) / (len(totloglist[j]) * logbinsize))  
                
                    hist_log, histbins_log = np.histogram(totloglist[j],bins = logbins_log, range=(0.5, end+0.5))
                
                    # histlist.append((hist_log, histbins_log))
            else:
                    if len(estep_outer) != 0:
                        steplindict, x, y, steplogdict, evalcirc = steptotlist
                        histlist = tothistlist
                        evallist = evalcirc
                        if which == "Log":
                            # Log histogram
                            totlogiter = steplogdict
                            totloglist = [key for key, val in totlogiter.items() for _ in range(val)]
                            hist_log, histbins_log = np.histogram(totloglist,bins = logbins, range=(0.5, end+0.5))
                            histlist.append((hist_log, histbins_log))
                        elif which == "Linear":
                            # Linear histogram
                            totliniter = steplindict
                            totlinlist = [key for key, val in totliniter.items() for _ in range(val)]
                            hist_lin, histbins_lin = np.histogram(totlinlist,bins = logbins, range=(0.5, end+0.5))
                            histlist.append((hist_lin, histbins_lin))
                        elif which == "Linear / a":
                            # Linear / a histogram
                            totliniter = steplindict
                            totlinlist = [key for key, val in totliniter.items() for _ in range(val)]
                            hist_lin, histbins_lin = np.histogram(totlinlist,bins = logbins, range=(0.5, end+0.5))
                            histlist.append((hist_lin, histbins_lin))  
                        else:
                            return(print(f"Warning: {which} is not a valid point. Please use (Log) or (Linear) as your options"))
                        
                        # E = 0 histogram
                        totcirclist = [key for key, val in evallist.items() for _ in range(val)]
                        histcirc, binscirc = np.histogram(totcirclist, bins = logbins, range=(0.5, end+0.5))
                        evalhistlist = [(histcirc, binscirc)]
                    else:
                      for j in range(len(steptotlist)):
                        steplindict, x, y, steplogdict, blank = steptotlist[j]
                        
                        histlist = tothistlist[j]
                        if which == "Log":
                            # Log histogram
                            totlogiter = steplogdict
                            totloglist = [key for key, val in totlogiter.items() for _ in range(val)]
                            hist_log, histbins_log = np.histogram(totloglist,bins = logbins, range=(0.5, end+0.5))
                            histlist.append((hist_log, histbins_log))
                        elif which == "Linear":
                            # Linear histogram
                            totliniter = steplindict
                            totlinlist = [key for key, val in totliniter.items() for _ in range(val)]
                            hist_lin, histbins_lin = np.histogram(totlinlist,bins = logbins, range=(0.5, end+0.5))
                            histlist.append((hist_lin, histbins_lin))
                        elif which == "Linear / a":
                            # Linear / a histogram
                            totliniter = steplindict
                            totlinlist = [key for key, val in totliniter.items() for _ in range(val)]
                            hist_lin, histbins_lin = np.histogram(totlinlist,bins = logbins, range=(0.5, end+0.5))
                            histlist.append((hist_lin, histbins_lin))  
                        else:
                            return(print(f"Warning: {which} is not a valid point. Please use (Log) or (Linear) as your options"))
                        tothistlist[j] = histlist
            gc.collect()
        
                    
        if inclination and len(estep_outer) == 0:
            fig, axs = plt.subplots(3,3,figsize = (9,9), sharex=True,sharey=True,gridspec_kw=dict(hspace=0,wspace=0))
            fig.suptitle("Detections of $R_E$ with marginalizations for "r"$\cos{i} = 0$ to 1 , and " r"$\omega$ = $0$ to $\frac{\pi}{2}$" f"\n ({which})")
        elif len(estep_outer) == 0:
            fig, axs = plt.subplots(3,4, figsize = (9,9), sharex=True,sharey=True,gridspec_kw=dict(hspace=0,wspace=0))               
            fig.suptitle("Detections of $R_E$ with marginalizations for "r"$\cos{i} = 0$ to 1 , and " r"$\omega$ = $0$ to $\frac{\pi}{2}$" f"\n ({which})")
        # Iterates through each subplot in the 3x4 figure
        if inclination == False:
            for j, ax  in enumerate(axs.flatten()):
            # Takes newly made lists for data collection
                iterlin = totlinlist[j]
                iterlog = totloglist[j]
                # Create variables for bin sizes
                nbin = 200
                amin = 0.5
                amax = 21
                # Make logbinsizes for all
                logbinsize = (np.log10(amin)-np.log10(amax))/nbin
            
                # Calculate weights for normalization
                # Check if this is wrong or not, cause of the np.ones_like()
                weights_lin = np.abs(np.ones_like(iterlin) / (len(iterlin) * logbinsize))    
                weights_log = np.abs(np.ones_like(iterlog) / (len(iterlog) * logbinsize))  
                # Creates the log spaced bins for our data
                # Stupid fix to stupid problems :)
                if j == 0:
                    logbins_lin = np.linspace(amin,amax,nbin)
                    logbins_log = np.geomspace(amin,amax,nbin)
                else:    
                    logbins_lin = np.geomspace(amin,amax, nbin)
                    logbins_log = np.geomspace(amin,amax,nbin)
            
                if inclination == False:
                        
                    datahist_lin, bins, patches_lin = ax.hist(
                        iterlin, bins=logbins_lin, range=(0.5, end+0.5),
                        stacked=True, histtype="step", alpha = 0.75, edgecolor = "black",
                        weights=weights_lin, fc = "none",
                    )
                    datahist_log, bins, patches_log = ax.hist(
                        iterlog, bins=logbins_log, range=(0.5, end+0.5),
                        stacked=True, histtype="step", alpha = 0.75, edgecolor = "red",
                        weights=weights_log, fc = "none",
                    )
                
                
            # Turns histogram bars into respective colors
            for patch in patches_lin:
                patch.set_edgecolor("k")
            for patch in patches_log:
                patch.set_edgecolor("r")
            ax.set_xlim(0.5,20.5)
            ax.set_ylim(0,10)
            ax.set_xscale("log")
        elif len(estep_outer) == 0:
            for j, ax  in enumerate(axs.flatten()):
                histlist = tothistlist[j]
                iterparam = param[j]
                for val in range(len(histlist)):
                    hist, bins = histlist[val]
                    if val == 0:
                        tothist = np.zeros_like(hist)
                        tothist = tothist + hist
                    elif val == len(histlist)-1:
                        tothist = tothist + hist
                        norm = np.abs(1 / (logbinsize * np.sum(tothist)))
                        StepPatch = ax.stairs(tothist * norm, bins, edgecolor = colorlist[0], fill = False)
                    else:
                        tothist = tothist + hist
                
                ax.grid(True,color = "grey", linestyle="--", linewidth="0.25", axis = "x", which = "both")
                ax.set_xlim(0.5,20.5)
                ax.set_xscale("log")
                ax.text(8, 4, f"e = {iterparam[0]}")
            
        # Text for understanding positions of each figure 
        
        if inclination == False:
            plt.text(1.25e-6,25,"e=0")
            plt.text(1.25e-6,15,"e=0.5")
            plt.text(1.25e-6,5,"e=0.9")
            plt.text(1.25e-6,5,"e=0.9")
            plt.text(4e-5,30.5,"i=0")
            plt.text(1.5e-3,30.5,"i=30")
            plt.text(5.5e-2,30.5,"i=60")
            plt.text(2.75, 30.5,"i=90")
        
            
        handles = [patches.Rectangle((0,0),1,1,color = c, ec = "w") for c in colorlist]
            
        print(f"finich time {time.time()}, {estep_outer}")   
        
        # Saves plot
        if inclination == False:
            fig.legend(handles, labels)
            plt.savefig(f'/College Projects/Microlensing Separation/Figures/CompleteHist_0002_LinLog.png')
        elif len(estep_outer) == 0:
            fig.legend(handles, labels)
            plt.savefig(f'/College Projects/Microlensing Separation/Figures/CompleteHist_9plots_incline_{inum}_0002_{which}.png')
            plt.show()
        return tothistlist, evalhistlist

    def UnityPlotHist(self, which, wnum, inum, unity = False):
        """
        """
        totlist = []
        evalcirc = []
        
        # Create variables for bin sizes
        nbin = 200
        amin = 0.5
        amax = 21
        # Make logbinsizes for all
        logbinsize = np.abs((np.log10(amin)-np.log10(amax))/nbin)
        
        
        # Initialize Lists
        estep = np.linspace(0,0.98, self.numestep)
        labels = ["Uniform", "Gamma", "Circular"]
        colorlist = ["blue", "red", "black"]
        # Gamma Portion
        alpha = 1.35 # Shape (Alpha)
        theta = 1/5.05 # Scale (Beta = 1 / Scale)
        x = np.linspace(0,0.98, wnum)
        gammastep = gamma.pdf(x, a = alpha, scale = theta)
        
        esteplist = []*self.numdiv
        param = []
        
        # Slices estep into parts for parallelization
        slices = int(self.numestep / self.numdiv)
        for i in range(self.numdiv):
            esteplist.append(estep[i*slices:(i+1)*slices])
            
            obj = Sep_gen()
        
            # Step, end, inclincation, which, estep
            param.append((0.002, 20, True, which, esteplist[i], wnum, inum, repeat(obj)))
            
        # Processing using parallelization     
        with Pool(processes = self.numdiv) as pool:
                tothistlist = pool.map(Sep_gen.HistGen, param)
        print("pool finished")
        for j in range(len(tothistlist)):
            totlist.append(tothistlist[j][0])
            evalcirc.append(tothistlist[j][1])
        

        circhist = evalcirc[0]
        
        # FIGURE FOR SEMIMAJOR AXIS
        fig, ax = plt.subplots(figsize = (9,9), sharex=True,sharey=True,gridspec_kw=dict(hspace=0,wspace=0))
        # fig.suptitle("Detections of $R_E$ with marginalizations for e = 0.0-0.9, "r"$\cos{i} = 0$ to 1 , and " r"$\omega$ = $0$ to $\frac{\pi}{2}$" f"\n ({which})")        
        # Combine Histogram Calculations
        for i in range(len(totlist)):
            histlist = totlist[i]
            circiter = circhist[0]
            for val in range(len(histlist)):
                    hist, bins = histlist[val]
                    ecirchist, circbins = circiter
                    # For Gamma weighting
                    gammanorm = gammastep[val]
                    gammahist = gammanorm * hist
                    if val == 0:
                        if i == 0:
                            tothist = np.zeros_like(hist)
                            totgammahist = tothist
                        tothist = tothist + hist
                        totgammahist = totgammahist + gammahist
                    elif val == len(histlist)-1 and i == len(totlist)-1:
                        tothist = tothist + hist
                        totgammahist = totgammahist + gammahist
                        norm = np.abs(1 / (np.sum(tothist)*logbinsize))
                        ecircnorm = np.abs(1 / (np.sum(ecirchist) * logbinsize))
                        gammanorm_final = np.abs(1/ (np.sum(totgammahist) * logbinsize))
                        total = np.sum(tothist)
                        print(total)
                        StepPatch = ax.stairs(tothist * norm, bins, edgecolor = colorlist[0], fill = False, label = "Uniform Dist.") # Uniform Dist
                        # May or may not need norm for gamma
                        StepPatch = ax.stairs(totgammahist * gammanorm_final, bins, edgecolor = colorlist[1], fill = False, label = "Gamma Dist.") # Gamma Dist
                        StepPatch = ax.stairs(ecirchist * ecircnorm, bins, edgecolor = colorlist[2], fill = False, label = "Circular Dist.") # Circular Dist
                    else:
                        tothist = tothist + hist
                        totgammahist = totgammahist + gammahist
        ax.grid(True,color = "grey", linestyle="--", linewidth="0.25", axis = "x", which = "both")
        ax.set_xlim(0.5,20.5)
        ax.set_xscale("log")
        ax.set_xlabel(r"Semimajor Axis [$\log{a/R_e}$]")    
        ax.set_ylabel(r"Counts")
        
        handles = [patches.Rectangle((0,0),1,1,color = c, ec = "w") for c in colorlist]    
        
        fig.legend(handles, labels)
        fig.tight_layout()
        # try:
        #     #### os.getcwd IS FOR DESKTOP, IF FOR UNITY, CHANGE TO os.get_cwd
        #     if not unity:
        #         plt.savefig(f'/College Projects/Microlensing Separation/Figures/UnityHist_eccent_incline_{self.numestep}_0002_{which}.png')
        #         print(os.getcwd(), os.path.abspath(f"/College Projects/Microlensing Separation/Figures/UnityHist_eccent_incline_{self.numestep}_0002_{which}.png"))
        #     else:
        #         plt.savefig(f"~/Figures/UnityHist_eccent_incline_{self.numestep}_0002_{which}.png")
        #         print(os.getcwd(), os.path.abspath(f"~/Figures/UnityHist_eccent_incline_{self.numestep}_0002_{which}.png"))
        #         print("It works!")
        # except:
        #     print("Did not save figure, something must be wrong....")
        #     print(os.getcwd())
        
        plt.savefig(f'/College Projects/Microlensing Separation/Figures/UnityHist_eccent_incline_{self.numestep}_0002_{which}_norm.png')
        # plt.savefig(f"C:/Users/victo/College Projects/Microlensing Separation/Figures/UnityHist_eccent_incline_{self.numestep}_0002_{which}.png")
        return tothist


if __name__ == "__main__":
    # @click.command()
    # @click.argument("numestep", required=False, type=int)#, help = "number of e values stepping through")
    # @click.argument("numdiv", required=False, type=int)#, help = "divisors for e values")
    # @click.argument("which", required=False)#, help = "type of step through (Linear, Log, Linear/a)")
    # @click.argument("wnum", required=False, type=int,)# help = "number of omegas to step through")
    # @click.argument("inum", required=False, type=int,)# help = "number of i's to step through")
    # @click.option("--unity/--no-unity", default=False, show_default=True,)# help="Toggle unity output save path.")
    # def cli(numestep, numdiv, which, wnum, inum, unity):
    #     # Defaults if not provided positionally
    #     if numestep is None: numestep = 10 # Usually 80
    #     if numdiv is None: numdiv = 5 # Usually 10
    #     if which is None: which = "Linear" # Usually Linear
    #     if wnum is None: wnum = 3 # Usually 75
    #     if inum is None: inum = 3 # Usually 75
    #     which = which.capitalize() if which.lower() == "linear" else which
    #     plotter = Sep_plot(numestep=numestep, numdiv=numdiv)
    #     plotter.UnityPlotHist(which=which, wnum=wnum, inum=inum, unity=unity)
    # cli()
    numestep = 4
    numdiv = 2
    wnum = 5 # THIS DETERMINES HOW MANY POSITIONS IN THE ARRAY THERE ARE
    inum = wnum
    which = "Linear"
    unity = "False"
    tothist = Sep_plot(numestep=numestep, numdiv=numdiv)
    rlist = tothist.MultiPlotProj(w = 0, start = 0.5, end = 20, step = 0.5)
    # rtemp = tothist.MultiPlotHist(w = 0, step = 0.002, end = 20, which = which)
    
    #step, end, inclination, which, estep_outer, inum, wnum
    # clist = tothist.CompletePlotHist([0.002, 20, True,"Linear", [], 75, 75])
    # tothist.UnityPlotHist(which = which, wnum = wnum, inum = inum, unity = unity)
    
    



