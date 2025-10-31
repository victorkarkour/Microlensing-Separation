# import astropy.constants as ac
import numpy as np
# import scipy.signal as signal
# import scipy.optimize as sc
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import time
from multiprocessing import Pool
import pandas as pd
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

    def __init__(self, numestep = 10, numdiv = 2, wnum = 10):# which = "Log":
        self.numestep = numestep
        self.numdiv = numdiv
        # self.which = which
        self.wnum = wnum
        self.inum = wnum

    def DataProj(self, w = 0, start = 0.5, end = 20, step = 0.5, specify = []):
        """
        
        """
        # List initialization
        listt = []
        totlist = []
        totparam = []
        # Parameter list for MultiPlot
        if len(specify) == 0:
            param = [
            # Row 1
            (0. , 0., w, end, step, start), (0., np.pi/6, w, end, step, start), (0., np.pi/3, w, end, step, start), (0., np.pi/2, w, end, step, start),
            # Row 2
            (0.5, 0., w, end, step, start), (0.5, np.pi/6, w, end, step, start), (0.5, np.pi/3, w, end, step, start), (0.5, np.pi/2, w, end, step, start),
            # Row 3
            (0.9, 0., w, end, step, start), (0.9, np.pi/6, w, end, step, start), (0.9, np.pi/3, w, end, step, start), (0.9, np.pi/2, w, end, step, start)
                
            ]
        else:
            param = [specify[0],specify[1], w, end, step, start]
        
        start = time.perf_counter()
        
        if len(specify) == 0:
            with Pool(processes = 3) as pool:
                result = pool.map(self.WorkProj, param)
        else:
            result = self.WorkProj(param)
        
        
        # Test source
        # totlist, totparam, listt = WorkProj(param[0])
        
        # Splits the total result into different results
        if len(specify) == 0:
            for i in range(len(result)):
                totlist.append(result[i][0])
                totparam.append(result[i][1])
                listt.append(result[i][2])
        else:
            totlist.append(result[0])
            totparam.append(result[1])
            listt.append(result[2])
            
        
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
    def DataHist(cls, w = 0, step = 0.002, end = 10, which = "Linear", inclination = False, istep = None, estep = [], specify = []):
        """
        """
        # Goes from both Linear and Log calculations to just Linear
        Linear = which
        
        if inclination and len(estep) == 0:
            param = [
                # Row 1
                (0, istep, w, end, step, Linear, inclination), (0.27, istep, w, end, step, Linear, inclination), (0.53, istep, w, end, step, Linear, inclination), (0.80, istep, w, end, step, Linear, inclination),
                # Row 2     
                (0.09, istep, w, end, step, Linear, inclination), (0.36, istep, w, end, step, Linear, inclination), (0.62, istep, w, end, step, Linear, inclination), (0.89, istep, w, end, step, Linear, inclination),    
                # Row 3     
                (0.18, istep, w, end, step, Linear, inclination), (0.45, istep, w, end, step, Linear, inclination), (0.71, istep, w, end, step, Linear, inclination), (0.98, istep, w, end, step, Linear, inclination)
                
                     ]
        elif inclination:
            param = [estep, istep, w, end, step, Linear, inclination]
        else:
            # Parameters for Paralellization
            if len(specify) == 0:
                param = [
                # Row 1
                (0. , 0., w, end, step, Linear, inclination), (0., np.pi/6, w, end, step, Linear, inclination), (0., np.pi/3, w, end, step, Linear, inclination), (0., np.pi/2, w, end, step, Linear, inclination),
                # Row 2
                (0.5, 0., w, end, step, Linear, inclination), (0.5, np.pi/6, w, end, step, Linear, inclination), (0.5, np.pi/3, w, end, step, Linear, inclination), (0.5, np.pi/2, w, end, step, Linear, inclination),
                # Row 3
                (0.9, 0., w, end, step, Linear, inclination), (0.9, np.pi/6, w, end, step, Linear, inclination), (0.9, np.pi/3, w, end, step, Linear, inclination), (0.9, np.pi/2, w, end, step, Linear, inclination)
                    
                ]
            else:
                # eccentricity and inclination specified
                param = [specify[0], specify[1], w, end, step, Linear, inclination]
        
        start = time.perf_counter()
        # Multi Processing
        if (inclination == True and len(estep) != 0) or len(specify) != 0:
            totlist = Sep_gen.Rchange(param = param)
        elif len(specify) == 0:
            with Pool(processes = 15) as pool:
                totlist = pool.map(Sep_gen.Rchange, param)
        end_time = time.perf_counter()
        totaltime = end_time - start
        print(f"Time to Compute was {totaltime:.4f} seconds.")
        


        # totlist = [
        #     list1,list4,list7, list10,
        #     list2,list5,list8, list11,
        #     list3,list6,list9, list12
        # ]
        
        return totlist, param

    def MultiPlotProj(self, w = 0, start = 0.5, end = 20, step = 0.5, specify = []):
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
        if len(specify) == 0:
            # 3 by 4 Plot
            list, totparam, listt = self.DataProj(w = w, start = start, end = end, step = step)
            fig, axs = plt.subplots(3,4, figsize = (13,9), gridspec_kw = {"hspace" : 0, "wspace" : 0}, sharex = False, sharey = False)
        else:
            # 1 by 1 plot
            list, totparam, listt = self.DataProj(w = w, start = start, end = end, step = step, specify = specify)
            fig, axs = plt.subplots(figsize = (9,9), gridspec_kw = {"hspace" : 0, "wspace" : 0}, sharex = False, sharey = False)
        
        rect = dict(boxstyle = "round", alpha = 0.5, facecolor = "white")        
        
        
        # fig.suptitle("Orbital Projection with Alterations in e, i, and "r"From $\omega$ 0 to $\ \frac{\pi}{2}$", x = 0.49, y = 0.99)
    
        # Iterates through each subplot in the 3x4 figure
        if len(specify) == 0:
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
                ax.text(0.80, 0.95, textstr, transform = ax.transAxes, fontsize = 20, verticalalignment = "top", bbox = rect)
                
                if j == 8:
                        ax.tick_params(axis = "both", labelbottom = True, labelleft = True, labelsize = 12)
                else:
                    ax.tick_params(axis = "both", labelbottom = False, labelleft = False, labelsize = 12)
                if j == 11:
                    handles, labels = ax.get_legend_handles_labels()
                    # MAY HAVE TO REMOVE BELOW LINE
                    ax.tick_params(labelsize = 12)


                    ax.legend(handles[0:9],labels[0:9], loc = "upper left", fontsize = 12, borderpad = 0.5, labelspacing = 0.50, handlelength = 2, framealpha = 0.75)
        else:
            iterlist = list[0]
            param = totparam[0]
            
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
                rangelist = np.arange(0.5,end+0.5,0.5)
                alpha = rangelist[g]
                label = f"a = {alpha}"
                color = colorlist[g % 10]   
                
                # Plots the data set, including the dot size according to velocity        
                dataproj = axs.scatter(initialx, initialy, s=dot, color= color, label=label)
                # Also includes points at which |r-r0| <= 0.01
                data = axs.scatter(xchange[g], ychange[g], s = 8, color = "yellow")
                
            # Creates the grid for each plot
            axs.grid(True,color = "grey", linestyle="--", linewidth="0.55", axis = "both", which = "both")
                
            # Plots an Einstein Ring Radius of 1 around each plot
            Circ2 = patches.Circle((0,0), 1, ec="k", fill=False, linestyle = ":", linewidth = 1)
            
            # Adds the Circle to the plot
            axs.add_patch(Circ2)
            
            # # Just grabs the labels for each plot just before it iterates through again
            # if j == 0:
            #     handles, labels = ax.get_legend_handles_labels()
                
            # Limits
            axs.set_xlim(-2,2)
            axs.set_ylim(-2,2)
            # Decorations    
            textstr = "\n".join((f'e = {param[0]}', f'i = {round(param[1],2)}'))
            axs.text(0.80, 0.95, textstr, transform = axs.transAxes, fontsize = 20, verticalalignment = "top", bbox = rect)
            handles, labels = axs.get_legend_handles_labels()
            axs.tick_params(labelsize = 12)
            axs.legend(handles[0:9],labels[0:9], loc = "upper left", fontsize = 12, borderpad = 0.5, labelspacing = 0.50, handlelength = 2, framealpha = 0.75)
        fig.tight_layout()
        
        # Saves to Figure Folder
        if len(specify) == 0:
            plt.savefig(f"/College_Projects/Microlensing Separation/Figures/MultiProj_omega_0.png")
        else:
            plt.savefig(f"/College_Projects/Microlensing Separation/Figures/MultiProj_omega_0_specified.png")
        # plt.savefig(f"C:/Users/victo/College_Projects/Microlensing Separation/Figures/Multi_a05_{end}_omega_0.png")
        # plt.show()
        
        
        return rlist

    def MultiPlotHist(self, w = 0, step = 0.002, end = 10, which = "Log", specify = []):
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
        rlist, param = self.DataHist(w = w, step = step, end = end, which = which, specify = specify) 
        rect = dict(boxstyle = "round", alpha = 0.5, facecolor = "white")
        
        if len(specify) == 0:
            fig, axs = plt.subplots(figsize = (13,9), gridspec_kw = {"hspace" : 0, "wspace" : 0}, sharex = False, sharey = False)
        else:
            fig, axs = plt.subplots(figsize = (13,9), gridspec_kw = {"hspace" : 0, "wspace" : 0}, sharex = False, sharey = False)
        # if which == "Linear":
        #     fig.suptitle("Detections of $R_E$ with Alterations in e, i, and "r"$\omega$ = 0" f" ({which})", x = 0.49, y = 0.99)
        # else:
        #     fig.suptitle("Detections of $R_E$ with Alterations in e, i, and "r"$\omega$ = 0" f"\n (For Linear, Log, & Power Law)", x = 0.49, y = 0.99)
        
        # Iterates through each subplot in the 3x4 figure
        if len(specify) == 0:
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
                ax.text(0.63, 0.95, textstr, transform = ax.transAxes, fontsize = 20, verticalalignment = "top", bbox = rect)
                # ax.text(3e0, 8.5, f"$e = {iterparam[0]}$") 
                # ax.text(3e0, 7.5, f"$i = {round(iterparam[1],2)}$")
                ax.grid(True,color = "grey", linestyle="--", linewidth="0.25", axis = "x", which = "both")
                ax.vlines(1/(1-iterparam[0]), 0, 1, transform = ax.get_xaxis_transform(), colors = 'green', alpha = 0.75, label = r"Expected Peak $e$")
                if j == 0 or j == 4 or j == 8:
                    if j == 8:
                        ax.tick_params(axis = "both", labelbottom = True, labelleft = True, labelsize = 12)
                    else:
                        ax.tick_params(axis = "both", labelbottom = True, labelleft = False, labelsize = 12)
                else:
                    ax.set_yticks([])
                    ax.set_xticks([])
                    ax.tick_params(axis = "x", labelbottom = False)
                if j == 11:
                    handles = [patches.Rectangle((0,0),1,1,color = c, ec = "w") for c in colorlist]
                    ax.tick_params(labelsize = 12)
                    if which == "Linear":
                        labels = ["Linear", "Peak Eccentricity"]
                    else:
                        labels = ["Linear", "Log", r"Power Law: $\alpha = 2$", r"Expected Peak $e$"]
                        
                    ax.legend(handles = handles, labels = labels, loc = "best", fontsize = 12)
        else:
            steplindict, x, y, steplogdict, steplinsemidict = rlist
            iterparam = param
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
            if which == "Linear":
                logbins_lin = np.linspace(amin,amax,1000+1)
            else: 
                logbins_lin = np.geomspace(amin,amax, nbin)
                logbins_log = np.geomspace(amin,amax,nbin)
            if which == "Linear":
                datahist_lin, bins, patches_lin = axs.hist(
                    totlinlist, bins=logbins_lin, range=(0.5, end+0.5),
                    stacked=True, histtype="step",
                    weights=weights_lin, label = "Linear"
                )
            else:
                datahist_lin, bins, patches_lin = axs.hist(
                    totlinlist, bins=logbins_lin, range=(0.5, end+0.5),
                    stacked=True, histtype="step", alpha = 0.75, edgecolor = "black",
                    weights=weights_lin, fc = "none", label = "Linear"
                )
                datahist_log, bins, patches_log = axs.hist(
                    totloglist, bins=logbins_log, range=(0.5, end+0.5),
                    stacked=True, histtype="step", alpha = 0.75, edgecolor = "red",
                    weights=weights_log, fc = "none", label = "Log"
                )
                datahist_linsemi , bins, patches_linsemi = axs.hist(
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
            axs.set_xlim(0.5,20.5)
            axs.set_ylim(0,10)
            axs.set_xscale("log")
            
            textstr = "\n".join((f'e = {iterparam[0]}', f'i = {round(iterparam[1],2)}'))
            axs.text(0.63, 0.95, textstr, transform = axs.transAxes, fontsize = 20, verticalalignment = "top", bbox = rect)
            # ax.text(3e0, 8.5, f"$e = {iterparam[0]}$") 
            # ax.text(3e0, 7.5, f"$i = {round(iterparam[1],2)}$")
            axs.grid(True,color = "grey", linestyle="--", linewidth="0.25", axis = "x", which = "both")
            axs.vlines(1/(1-iterparam[0]), 0, 1, transform = axs.get_xaxis_transform(), colors = 'green', alpha = 0.75, label = r"Expected Peak $e$")
            axs.tick_params(labelsize = 12)
            handles = [patches.Rectangle((0,0),1,1,color = c, ec = "w") for c in colorlist]
            if which == "Linear":
                labels = ["Linear", "Peak Eccentricity"]
            else:
                labels = ["Linear", "Log", r"Power Law: $\alpha = 2$", r"Expected Peak $e$"]
                
            axs.legend(handles = handles, labels = labels, loc = "best", fontsize = 12)
                
        # fig.tight_layout(pad=1.25,h_pad=0, w_pad=0, rect = (0.08, 0.0, 0.95, 0.95))
        fig.tight_layout()
        if len(specify) == 0:
            plt.savefig(f'/College_Projects/Microlensing Separation/Figures/MultiHist_omega_0_0002_{which}.png')
        else:
            plt.savefig(f'/College_Projects/Microlensing Separation/Figures/MultiHist_omega_0_specified.png')
        # plt.show()
        return rlist

    @staticmethod
    def CompletePlotHist(param):
        """
        
        """
        step, end, inclination, which, estep_outer, inum, wnum, unity = param
        
        if not inclination:
            colorlist = ["black", "red", "blue", "green"]
            # labels = ["Linear", "Log", r"Power Law = $\alpha = 2$"]
        elif inclination:
            if which == "Log":
                colorlist = ["red", "green"]
            elif which == "Linear":
                colorlist = ["black", "green"]
            else:
                colorlist = ["blue", "green"]
        # Dictionary for storing Rchange results
        totlinlist = []
        totloglist = []
        totsemilist = []
        histlistlin = []
        histlistlog = []
        histlistsemi = []
        if len(estep_outer) == 0:
             tothistlist =[[] for _ in range(12)]
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
                # Eccentricity, i, and omega marginalized
                steptotlist, param = Sep_plot.DataHist(w = k, step = step, end = end, which = which, inclination = inclination, istep = istep, estep = estep)
            elif inclination:
                # i and omega marginalized
                steptotlist, param = Sep_plot.DataHist(w = k, step = step, end = end, which = which, inclination = inclination, istep = istep)
            else:
                # omega marginalized
                steptotlist, param = Sep_plot.DataHist(w = k, step = step, end = end, which = which, inclination = False)
            # Once complete, takes the data through each set
            if not inclination:
                # For omega marginalization
                for j in range(len(steptotlist)):
                        steplindict, x, y, steplogdict, stepsemidict = steptotlist[j]
                        histlist = tothistlist[j]
                        # Log histogram
                        totlogiter = steplogdict
                        totloglist = [key for key, val in totlogiter.items() for _ in range(val)]
                        hist_log, histbins_log = np.histogram(totloglist,bins = logbins, range=(0.5, end+0.5))
                        if k != 0.0:
                            histlistiter = histlistlog[j]
                            hist_log_iter, bins_log_iter = histlistiter
                            hist_log_iter = hist_log_iter + hist_log
                            histlistlog[j] = (hist_log_iter, bins_log_iter)
                        else:
                            histlistlog.append((hist_log, histbins_log))
                        # Linear histogram
                        totliniter = steplindict
                        totlinlist = [key for key, val in totliniter.items() for _ in range(val)]
                        hist_lin, histbins_lin = np.histogram(totlinlist,bins = logbins, range=(0.5, end+0.5))
                        if k != 0.0:
                            histlistiter = histlistlin[j]
                            hist_lin_iter, bins_lin_iter = histlistiter
                            hist_lin_iter = hist_lin_iter + hist_lin
                            histlistlin[j] = (hist_lin_iter, bins_lin_iter)
                        else:
                            histlistlin.append((hist_lin, histbins_lin))
                        # Linear / a histogram
                        totsemiiter = stepsemidict
                        totsemilist = [key for key, val in totsemiiter.items() for _ in range(val)]
                        hist_semi, histbins_semi = np.histogram(totsemilist,bins = logbins, range=(0.5, end+0.5))
                        if k != 0.0:
                            histlistiter = histlistsemi[j]
                            hist_semi_iter, bins_semi_iter = histlistiter
                            hist_semi_iter = hist_semi_iter + hist_semi
                            histlistsemi[j] = (hist_semi_iter, bins_semi_iter)
                        else:
                            histlistsemi.append((hist_semi, histbins_semi))
                        
                        # Shouldn't be needed, all histlist values are in individual histograms
                        # histlist.append((histlistlin, histlistlog, histlistsemi))
                        # tothistlist[j] = histlist
            else:
                    if len(estep_outer) != 0:
                        # For eccentricity, inclination, and omega marginalization
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
                        # For inclination and omega marginalization
                        for j in range(len(steptotlist)):
                            steplindict, stepsemidict, y, steplogdict, blank = steptotlist[j]
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
                            elif which == "Linear_a":
                                # Linear / a histogram
                                totsemiiter = stepsemidict
                                totsemilist = [key for key, val in totsemiiter.items() for _ in range(val)]
                                hist_semi, histbins_semi = np.histogram(totsemilist,bins = logbins, range=(0.5, end+0.5))
                                histlist.append((hist_semi, histbins_semi))  
                            else:
                                return(print(f"Warning: {which} is not a valid point. Please use (Log) or (Linear) as your options"))
                            tothistlist[j] = histlist
            gc.collect()
        
                    
        fig, axs = plt.subplots(3,4, figsize = (13,9), sharex=True,sharey=True,gridspec_kw=dict(hspace=0,wspace=0))
        # fig.suptitle("Detections of $R_E$ with marginalizations for "r"$\cos{i} = 0$ to 1 , and " r"$\omega$ = $0$ to $\frac{\pi}{2}$" f"\n ({which})")
                       
        # Iterates through each subplot in the 3x4 figure
        if not inclination :
            for j, ax  in enumerate(axs.flatten()):
            # Takes newly made lists for data collection 
                # histlist = tothistlist[j]
                histiter_lin, bins_lin = histlistlin[j]
                histiter_log, bins_log = histlistlog[j]
                histiter_semi, bins_semi = histlistsemi[j]
                iterparam = param[j]
                # for val in range(len(histiter_lin)):
                #     hist_lin, bins_lin = histiter_lin[val]
                #     hist_log, bins_log = histiter_log[val]
                #     hist_semi, bins_semi = histiter_semi[val]
                #     if val == 0:
                #         # Initializes
                #         tothist = np.zeros_like(hist_lin)
                #         tothist_lin = tothist + hist_lin
                #         tothist_log = tothist + hist_log
                #         tothist_semi = tothist + hist_semi
                #     elif val == len(histiter_lin)-1:
                #         # Takes final count and normalizes, then plots
                #         tothist_lin = tothist + hist_lin
                #         tothist_log = tothist + hist_log
                #         tothist_semi = tothist + hist_semi
                norm_lin = np.abs(1 / (logbinsize * np.sum(histiter_lin)))
                norm_log = np.abs(1 / (logbinsize * np.sum(histiter_log)))
                norm_semi = np.abs(1 / (logbinsize * np.sum(histiter_semi)))
                StepPatch_lin = ax.stairs(histiter_lin * norm_lin, bins_lin, edgecolor = colorlist[0], fill = False, alpha = 0.5)
                StepPatch_log = ax.stairs(histiter_log * norm_log, bins_log, edgecolor = colorlist[1], fill = False, alpha = 0.5)
                StepPatch_semi = ax.stairs(histiter_semi * norm_semi, bins_semi, edgecolor = colorlist[2], fill = False, alpha = 0.5)
                    # else:
                    #     # Keeps counting
                    #     tothist_lin = tothist + hist_lin
                    #     tothist_log = tothist + hist_log
                    #     tothist_semi = tothist + hist_semi
                    
                rect = dict(boxstyle = "round", alpha = 0.5, facecolor = "white")
                textstr = "\n".join((f'e = {iterparam[0]}', f'i = {round(iterparam[1],2)}'))
                ax.text(0.63, 0.95, textstr, transform = ax.transAxes, fontsize = 20, verticalalignment = "top", bbox = rect)
                ax.grid(True,color = "grey", linestyle="--", linewidth="0.25", axis = "x", which = "both")
                # Lines and organizing labels to be cleaner
                ax.vlines(1/(1-iterparam[0]), 0, 1, transform = ax.get_xaxis_transform(), colors = 'green', alpha = 0.75, label = r"Expected Peak $e$")
                if j == 0 or j == 4 or j == 8:
                    if j == 8:
                        ax.tick_params(axis = "both", labelbottom = True, labelleft = True, labelsize = 12)
                    else:
                        ax.tick_params(axis = "both", labelbottom = True, labelleft = False, labelsize = 12)
                else:
                    # ax.set_yticks([])
                    ax.set_xticks([])
                    ax.tick_params(axis = "x", labelbottom = False, labelsize = 12)
                
                if j == 11:
                    handles = [patches.Rectangle((0,0),1,1,color = c, ec = "w") for c in colorlist]
                    labels = ["Linear", "Log", r"Power Law: $\alpha = 2$", r"Expected Peak $e$"]
                    ax.tick_params(labelsize = 12)
                    ax.legend(handles = handles, labels = labels, loc = "best", fontsize = 20) 
            # Initial Params for Plot
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
                # Decoration
                rect = dict(boxstyle = "round", alpha = 0.5, facecolor = "white")
                textstr = f'e = {iterparam[0]}'
                ax.text(0.63, 0.95, textstr, transform = ax.transAxes, fontsize = 20, verticalalignment = "top", bbox = rect)
                ax.grid(True,color = "grey", linestyle="--", linewidth="0.25", axis = "x", which = "both")
                # Lines and organizing labels to be cleaner
                ax.vlines(1/(1-iterparam[0]), 0, 1, transform = ax.get_xaxis_transform(), colors = 'green', alpha = 0.75, label = r"Expected Peak $e$")
                if j == 0 or j == 4 or j == 8:
                    if j == 8:
                        ax.tick_params(axis = "both", labelbottom = True, labelleft = True, labelsize = 12)
                    else:
                        ax.tick_params(axis = "both", labelbottom = True, labelleft = False, labelsize = 12)
                else:
                    # ax.set_yticks([])
                    ax.set_xticks([])
                    ax.tick_params(axis = "x", labelbottom = False, labelsize = 12)
                
                if j == 11:
                    handles = [patches.Rectangle((0,0),1,1,color = c, ec = "w") for c in colorlist]
                    labels = [f"{which}", r"Expected Peak $e$"]
                    ax.tick_params(labelsize = 12)
                    ax.legend(handles = handles, labels = labels, loc = "best", fontsize = "small") 
            # Initial Params for Plot
                ax.set_xlim(0.5,20.5)
                ax.set_ylim(0,10)
                ax.set_xscale("log")
            
        print(f"finish time {time.time()}, {estep_outer}")   
        
        fig.tight_layout()
        # Saves plot
        if inclination == False:
            # fig.legend(handles, labels)
            plt.savefig(f'/College_Projects/Microlensing Separation/Figures/CompleteHist_0002_{wnum}_LinLogSemi.png')
        elif len(estep_outer) == 0:
            # fig.legend(handles, labels)
            if unity:
                plt.savefig(f"/home/karkour.2/Figures/CompleteHist_{wnum}_{inum}_{which}_unity.png")
            else:
                plt.savefig(f'/College_Projects/Microlensing Separation/Figures/CompleteHist_{wnum}_{inum}_{which}.png')
        return tothistlist, evalhistlist

    def UnityPlotHistGen(self, which, unity = False):
        """
        """
        totlist = []
        evalcirc = []

        estep = np.linspace(0,0.98, self.numestep)
        x = np.linspace(0,0.98, self.wnum)
        esteplist = []*self.numdiv
        param = []
        
        # Slices estep into parts for parallelization
        slices = int(self.numestep / self.numdiv)
        for i in range(self.numdiv):
            esteplist.append(estep[i*slices:(i+1)*slices])
            
            obj = Sep_gen()
        
            # Step, end, inclincation, which, estep
            param.append((0.002, 20, True, which, esteplist[i], self.wnum, self.inum, repeat(obj)))
            
        # Processing using parallelization     
        with Pool(processes = self.numdiv) as pool:
                tothistlist = pool.map(Sep_gen.HistGen, param)
        # print("pool finished")
        for j in range(len(tothistlist)):
            totlist.append(tothistlist[j][0])
            evalcirc.append(tothistlist[j][1])

        # Process for CSV File
        circhist = evalcirc[0]
        for i in range(len(totlist)):
            histlist = totlist[i]
            circiter = circhist[0]
            for val in range(len(histlist)):
                    hist, bins = histlist[val]
                    ecirchist, circbins = circiter
                    if val == 0:
                        if i == 0:
                            tothist = np.zeros_like(hist)
                        tothist = tothist + hist
                    elif val == len(histlist)-1 and i == len(totlist)-1:
                        tothist = tothist + hist
                        total = np.sum(tothist)
                        print("Total Number of Points: ", total)
                    else:
                        tothist = tothist + hist
        # Save to CSV
        unity_data = {
                    "final list": tothist,
                    "circular list": ecirchist}
        
        df_unity = pd.DataFrame(unity_data)
        if unity:
            file_name = f'/home/karkour.2/Results/UnityHist_eccent_incline_{self.numestep}_0002_{which}.csv'
        else:
            file_name = f'/College_Projects/Microlensing Separation/Results/UnityHist_eccent_incline_{self.numestep}_0002_{which}.csv'

        df_unity.to_csv(file_name, index = False)

        print("File saved successfully")

        return x

    def UnityPlotHistLoad(self, which, unity = False):
        """
        """
        if unity:
            file_name = f'/home/karkour.2/Results/UnityHist_eccent_incline_{self.numestep}_0002_{which}.csv'
        else:
            file_name = f'/College_Projects/Microlensing Separation/Results/UnityHist_eccent_incline_{self.numestep}_0002_{which}.csv'

        df_unity = pd.read_csv(file_name)


        # Create variables for bin sizes
        nbin = 200
        amin = 0.5
        amax = 21
        # Make logbinsizes for all
        logbinsize = np.abs((np.log10(amin)-np.log10(amax))/nbin)
        
        
        # Initialize Lists
        
        labels = ["Uniform", "Gamma", "Circular"]
        colorlist = ["blue", "red", "black"]
        # Gamma Portion
        alpha = 1.35 # Shape (Alpha)
        theta = 1/5.05 # Scale (Beta = 1 / Scale)
        x = np.linspace(0,0.98, nbin-1)
        gammastep = gamma.pdf(x, a = alpha, scale = theta)
        # Make bins
        bins = np.geomspace(amin,amax, nbin)

        
        # FIGURE FOR SEMIMAJOR AXIS
        fig, ax = plt.subplots(figsize = (9,9), sharex=True,sharey=True,gridspec_kw=dict(hspace=0,wspace=0))
        # fig.suptitle("Detections of $R_E$ with marginalizations for e = 0.0-0.9, "r"$\cos{i} = 0$ to 1 , and " r"$\omega$ = $0$ to $\frac{\pi}{2}$" f"\n ({which})")        
        # Combine Histogram Calculations
        uniformhist = df_unity["final list"].to_numpy()
        hist = uniformhist.copy()
        circhist = df_unity["circular list"].to_numpy()
        # Make Gamma Calculation
        # for val in range(len(hist)):
        #     # For Gamma weighting
        #     gammanorm = gammastep[val]
        #     gammahist = gammanorm * hist[val]
        #     if val == 0:
        #         totgammahist = np.zeros_like(hist)
        #         totgammahist[val] = gammahist
        #     elif val == len(hist)-1:
        #         totgammahist[val] = gammahist
        #     else:
        #         totgammahist[val] = gammahist
        totgammahist = gammastep * hist

        norm = np.abs(1 / (np.sum(uniformhist) * logbinsize))
        ecircnorm = np.abs(1 / (np.sum(circhist) * logbinsize))
        gammanorm_final = np.abs(1/ (np.sum(totgammahist) * logbinsize))
        result = sum(uniformhist)
        print(result, sum(totgammahist), sum(circhist))

        StepPatch = ax.stairs(uniformhist * norm, bins, edgecolor = colorlist[0], fill = False, label = "Uniform Dist.") # Uniform Dist
        # May or may not need norm for gamma
        StepPatch = ax.stairs(totgammahist * gammanorm_final, bins, edgecolor = colorlist[1], fill = False, label = "Gamma Dist.") # Gamma Dist
        StepPatch = ax.stairs(circhist * ecircnorm, bins, edgecolor = colorlist[2], fill = False, label = "Circular Dist.") # Circular Dist


        ax.grid(True,color = "grey", linestyle="--", linewidth="0.25", axis = "x", which = "both")
        ax.set_xlim(0.5,20.5)
        ax.set_xscale("log")
        ax.set_xlabel(r"Semimajor Axis [$\log{a/R_e}$]")    
        ax.set_ylabel(r"Counts")
        
        handles = [patches.Rectangle((0,0),1,1,color = c, ec = "w") for c in colorlist]    
        
        ax.legend(handles, labels)
        fig.tight_layout()
        # try:
        #     #### os.getcwd IS FOR DESKTOP, IF FOR UNITY, CHANGE TO os.get_cwd
        #     if not unity:
        #         plt.savefig(f'/College_Projects/Microlensing Separation/Figures/UnityHist_eccent_incline_{self.numestep}_0002_{which}.png')
        #         print(os.getcwd(), os.path.abspath(f"/College_Projects/Microlensing Separation/Figures/UnityHist_eccent_incline_{self.numestep}_0002_{which}.png"))
        #     else:
        #         plt.savefig(f"~/Figures/UnityHist_eccent_incline_{self.numestep}_0002_{which}.png")
        #         print(os.getcwd(), os.path.abspath(f"~/Figures/UnityHist_eccent_incline_{self.numestep}_0002_{which}.png"))
        #         print("It works!")
        # except:
        #     print("Did not save figure, something must be wrong....")
        #     print(os.getcwd())
        
        plt.savefig(f'/College_Projects/Microlensing Separation/Figures/UnityHist_eccent_incline_{self.numestep}_0002_{which}_norm_load.png')
        # plt.savefig(f"C:/Users/victo/College_Projects/Microlensing Separation/Figures/UnityHist_eccent_incline_{self.numestep}_0002_{which}.png")
        return tothist
    def statistics(self, which):
        """
        """
         # Create variables for bin sizes
        nbin = 200
        amin = 0.5
        amax = 21
        # Make logbinsizes for all
        logbinsize = np.abs((np.log10(amin)-np.log10(amax))/nbin)

        # filename = f'/College_Projects/Microlensing Separation/Results/UnityHist_eccent_incline_{self.numestep}_0002_{which}.csv'
        filename_2 = f'/Users/victo/College_Projects/Microlensing Separation/Results/UnityHist_eccent_incline_{self.numestep}_0002_{which}.csv'
        df_stats = pd.read_csv(filename_2)
        df_stats["cumulative"] = 0
        cumulative = 0
        cumul_norm = np.abs(1 / (df_stats["final list"].max()))
        for i in range(len(df_stats["final list"])):
            cumulative = cumulative + df_stats.loc[i, "final list"]
            df_stats.loc[i, "cumulative"] = cumulative
        c = np.cumsum(df_stats["final list"])
        fig, ax = plt.subplots(figsize = (9,9), sharex=True,sharey=True,gridspec_kw=dict(hspace=0,wspace=0))
        fig.suptitle(f"Cumulative Distribution Function \n ({which})")
        ax.plot(df_stats["cumulative"]*cumul_norm, color = "black")
        ax.plot(c*cumul_norm, color = "red", alpha = 0.5)
        ax.set_xlim(0.5,20.5)
        ax.set_ylim(0,1)
        ax.set_xscale("log")
        ax.set_xlabel(r"Semimajor Axis [$\log{a/R_e}$]")
        ax.set_ylabel(r"CDF")
        # plt.savefig(f'/College_Projects/Microlensing Separation/Figures/CDF_{self.numestep}_0002_{which}.png')
        plt.savefig(f'C:/Users/victo/College_Projects/Microlensing Separation/Figures/CDF_{self.numestep}_0002_{which}.png')
        
        mean = np.log10(df_stats["final list"].mean())
        std = np.log10(df_stats["final list"].std())

        upper_68 = np.abs(mean + std)
        lower_68 = np.abs(mean - std)
        upper_95 = np.abs(mean + 2*std)
        lower_95 = np.abs(mean - 2*std)

        stats = [mean, std, upper_68, lower_68, upper_95, lower_95]
        print("Mean: ", mean, " Std Dev: ", std, "68% CI: [", lower_68, ", ", upper_68, "] 95% CI: [", lower_95, ", ", upper_95, "]")
        return stats

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
    numestep = 80
    numdiv = 80
    wnum = 75 # THIS DETERMINES HOW MANY POSITIONS IN THE ARRAY THERE ARE
    inum = wnum
    which = "Linear"
    unity = False
    specify = [0., np.pi/3]
    tothist = Sep_plot(numestep=numestep, numdiv=numdiv, wnum = wnum)
    # rlist = tothist.MultiPlotProj(w = 0, start = 0.5, end = 20, step = 0.5, specify = specify)
    # specify = [eccentricity, inclination]
    # rtemp = tothist.MultiPlotHist(w = 0, step = 0.002, end = 20, which = which , specify = specify)
    
    #step, end, inclination, which, estep_outer, inum, wnum
    # tothist.CompletePlotHist([0.002, 20, True, which, [], inum, wnum, unity])
    # folder = tothist.UnityPlotHistGen(which = which, unity = unity)
    # load = tothist.UnityPlotHistLoad(which = which, unity = unity)
    cdf = tothist.statistics(which = which)



