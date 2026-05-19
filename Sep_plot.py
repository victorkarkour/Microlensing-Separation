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
# import click
import os
import matplotlib.gridspec as gridspec
import statistics as stats

matplotlib.use("Agg")
matplotlib.rcParams["axes.labelsize"] = 18
matplotlib.rcParams["font.size"] = 18
matplotlib.rcParams["xtick.major.size"] = 12
matplotlib.rcParams["xtick.minor.size"] = 8
matplotlib.rcParams["ytick.major.size"] = 12
matplotlib.rcParams["ytick.minor.size"] = 8
class Sep_plot(Sep_gen):

    def __init__(self, numestep = 10, numdiv = 2, wnum = 10):# which = "Log":
        self.numestep = numestep
        self.numdiv = numdiv
        # self.which = which
        self.wnum = wnum
        self.wstep = np.arange(0, np.pi/2, wnum)

        self.inum = wnum
        cosstep = np.linspace(0,1,inum)
        self.istep = np.arccos(cosstep)

        
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
            with Pool(processes = 9) as pool:
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
        
        # Parameters for Paralellization
        if len(specify) == 0:
            param = [
            # Row 1
            (0. , 0., w, end, step, Linear, inclination, 0), (0., np.pi/6, w, end, step, Linear, inclination, 0), (0., np.pi/3, w, end, step, Linear, inclination, 0), (0., np.pi/2, w, end, step, Linear, inclination, 0),
            # Row 2
            (0.5, 0., w, end, step, Linear, inclination, 0), (0.5, np.pi/6, w, end, step, Linear, inclination, 0), (0.5, np.pi/3, w, end, step, Linear, inclination , 0), (0.5,np.pi/2,w,end ,step ,Linear,inclination , 0),
            # Row 3
            (0.9 , 0., w, end ,step , Linear, inclination ,0), (0.9, np.pi/6, w, end ,step , Linear, inclination ,0), (0.9, np.pi/3, w, end ,step , Linear, inclination , 0), (0.9, np.pi/2, w, end , step , Linear, inclination , 0)
                
            ]
        else:
            # eccentricity and inclination specified
            param = [specify[0], specify[1], w, end, step, Linear, inclination, 0]
        
        start = time.perf_counter()
        # Multi Processing
        if len(specify) != 0:
            totlist = Sep_gen.Rchange(param = param)
        else:
            with Pool(processes = 6) as pool:
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
        
        omega = round(w)
        
        # Gets everything ready for multiprocessing of orbital projections
        if len(specify) == 0:
            # 3 by 4 Plot
            list, totparam, listt = self.DataProj(w = w, start = start, end = end, step = step)
            fig, axs = plt.subplots(3,4, figsize = (13,9), gridspec_kw = {"hspace" : 0, "wspace" : 0}, sharex = True, sharey = True)
        else:
            # 1 by 1 plot
            list, totparam, listt = self.DataProj(w = w, start = start, end = end, step = step, specify = specify)
            fig, axs = plt.subplots(figsize = (9,9), gridspec_kw = {"hspace" : 0, "wspace" : 0}, sharex = True, sharey = True)
        
        rect = dict(boxstyle = "round", alpha = 1, facecolor = "white")        
        colorlist = ["forestgreen", "tomato", "mediumblue", "orange", "purple",
                                    "pink", "blue", "red", "green", "cyan"]
        linelist = ["-", "--", "-.", ":"]
        
        # fig.suptitle("Orbital Projection with Alterations in e, i, and "r"From $\omega$ 0 to $\ \frac{\pi}{2}$", x = 0.49, y = 0.99)
    
        # Iterates through each subplot in the 3x4 figure
        if len(specify) == 0:
            for j, ax  in enumerate(axs.flatten()):
                
                # Takes the first data set in the list
                # These contain 12 other data sets  
                iterlist = list[j]
                param = totparam[j]
                
                # Finds points <= 0.01 for each projection
                rtemp, xchange, ychange = Sep_gen.Rchange(param, coords = True)
                
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
                        linestyle = linelist[g % 4]   
                    
                    # Plots the data set, including the dot size according to velocity        
                    dataproj = ax.scatter(initialx, initialy, s=dot, color= color, label=label, ls = linestyle)
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
                limit = [-2,-1, 0, 1, 2]
                ax.set_xlim(-2,2)
                ax.set_ylim(-2,2)
                ax.set_xticks(limit)
                ax.set_yticks(limit)
                    
                textstr = "\n".join((f'e = {param[0]}', rf'i = {round(np.degrees(param[1]))}$\degree$'))
                ax.text(0.05, 0.95, textstr, transform = ax.transAxes, fontsize = 20, verticalalignment = "top", bbox = rect)
                
                ax.tick_params(axis = "both", labelbottom = False, labelleft = False)
                # if j == 0 or j == 4 or j == 8:
                #     if j == 8:
                #         ax.tick_params(axis = "both", labelbottom = False, labelleft = False)
                #     else:
                #         ax.tick_params(axis = "both", labelbottom = False, labelleft = False)
                # else:
                #     ax.tick_params(axis = "both", labelbottom = False, labelleft = False)
                # if j == 11:
                    # handles, labels = ax.get_legend_handles_labels()
                    # ax.legend(handles[0:9],labels[0:9], loc = "upper right", fontsize = 10, borderpad = 0.5, labelspacing = 0.50, handlelength = 2, framealpha = 0.75)
                # print(ax.xaxis.get_ticklocs(minor = False))
            if w == 0:
                value = "0"
                text_string = "(a)"
            elif w == (np.pi)/6:
                value = "pi_6"
                text_string = "(b)"
            elif w == (np.pi)/3:
                value = "pi_3"
                text_string = "(c)"
            elif w == (np.pi)/2:
                value = "pi_2"
                text_string = "(d)"
            else:
                return("Warning: Input correct version of pi (idk just do it right man).")
                
            plt.figtext(0.94, 0.01, text_string, fontsize = 40)
            fig.tight_layout(pad = 2.0)
        else:
            iterlist = list[0]
            param = totparam[0]
            
            # Finds points <= 0.01 for each projection
            rtemp, xchange, ychange = Sep_gen.Rchange(param, coords = True)
            
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
                rangelist = np.arange(0.5,end+0.5,0.5)
                alpha = rangelist[g]
                label = f"a = {alpha}"
                color = colorlist[g % 10]
                linestyle = linelist[g % 4]    
                
                # Plots the data set, including the dot size according to velocity        
                dataproj = axs.scatter(initialx, initialy, s=dot, color= color, label=label, ls = linestyle)
                # Also includes points at which |r-r0| <= 0.01
                data = axs.scatter(xchange[g], ychange[g], s = 8, color = "yellow",)
                
            # Creates the grid for each plot
            axs.grid(True,color = "grey", linestyle="--", linewidth="0.55", axis = "both", which = "both")
                
            # Plots an Einstein Ring Radius of 1 around each plot
            Circ2 = patches.Circle((0,0), 1, ec="k", fill=False, linestyle = ":", linewidth = 1)
            
            # Adds the Circle to the plot
            axs.add_patch(Circ2)
                
            # Limits
            limit = [-2,-1, 0, 1, 2]
            axs.set_xlim(-2,2)
            axs.set_ylim(-2,2)
            axs.set_xticks(limit)
            axs.set_yticks(limit)
            axs.set_xlabel(r"$a_{\perp,x} [R_E]$")
            axs.set_ylabel(r"$a_{\perp,y} [R_E]$")
            # Decorations    
            textstr = "\n".join((f'e = {param[0]}', rf'i = {round(np.degrees(param[1]))}$\degree$'))
            axs.text(0.05, 0.95, textstr, transform = axs.transAxes, fontsize = 20, verticalalignment = "top", bbox = rect)
            handles, labels = axs.get_legend_handles_labels()
            axs.legend(handles[0:12],labels[0:12], loc = "upper right", fontsize = 20, borderpad = 0.5, labelspacing = 0.50, handlelength = 2, framealpha = 0.75)
            plt.figtext(0.93, 0.01, "(a)", fontsize = 30)
            fig.tight_layout()
        
        # Saves to Figure Folder
        if len(specify) == 0:
            try:
                plt.savefig(f"/College_Projects/Microlensing Separation/Figures/MultiProj_omega_{value}.png")
            except OSError:
                plt.savefig(f'C:/Users/victo/College_Projects/Microlensing Separation/Figures/MultiProj_omega_{value}.png')
        else:
            try:
                plt.savefig(f"/College_Projects/Microlensing Separation/Figures/MultiProj_omega_{omega}_specified.png")
            except OSError:
                plt.savefig(f"C:/Users/victo/College_Projects/Microlensing Separation/Figures/MultiProj_omega_{omega}_specified.png")
            
        return rlist

    def MultiPlotHist(self, w = 0, step = 0.002, end = 10, which = "Log", specify = []):
        """
        """
        
        # totlinlist = [[] for _ in range(12)]
        # totloglist = [[] for _ in range(12)]
        # totlinsemilist = [[] for _ in range(12)]
        
        if which == "Linear":
            colorlist = ["green", "black"]
        else:
            colorlist = ["green", "red", "blue", "black"]
        
        # Data
        rlist, param = self.DataHist(w = w, step = step, end = end, which = which, specify = specify) 
        rect = dict(boxstyle = "round", alpha = 1, facecolor = "white")
        
        omega = round(w)
        # Initialize plot
        if len(specify) == 0:
            # 3 by 4 Plot
            fig, axs = plt.subplots(3,4, figsize = (13,9), gridspec_kw = {"hspace" : 0, "wspace" : 0}, sharex = True, sharey = True)
        else:
            # 1 by 1 plot
            fig, axs = plt.subplots(figsize = (9,9), gridspec_kw = {"hspace" : 0, "wspace" : 0}, sharex = True, sharey = True)
        # if which == "Linear":
        #     fig.suptitle("Detections of $R_E$ with Alterations in e, i, and "r"$\omega$ = 0" f" ({which})", x = 0.49, y = 0.99)
        # else:
        #     fig.suptitle("Detections of $R_E$ with Alterations in e, i, and "r"$\omega$ = 0" f"\n (For Linear, Log, & Power Law)", x = 0.49, y = 0.99)
        
        # Iterates through each subplot in the 3x4 figure
        if len(specify) == 0:
            for j, ax  in enumerate(axs.flatten()):
                
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
                        patch.set_edgecolor("g")
                else:
                    for patch in patches_lin:
                        patch.set_edgecolor("g")
                    for patch in patches_log:
                        patch.set_edgecolor("r")
                    for patch in patches_linsemi:
                        patch.set_edgecolor("b")

                limit = [x for x in np.arange(0.5, end + 0.5 , 0.5)]
                
                ax.set_xlim(0.5,20.5)
                ax.set_ylim(0,10)
                ax.set_xticks(limit)
                # ax.set_yticks(limit)
                ax.set_xscale("log")
                
                textstr = "\n".join((f'e = {iterparam[0]}', rf'i = {round(np.degrees(iterparam[1]))}$\degree$'))
                ax.text(0.05, 0.95, textstr, transform = ax.transAxes, fontsize = 20, verticalalignment = "top", bbox = rect)
                # ax.text(3e0, 8.5, f"$e = {iterparam[0]}$") 
                # ax.text(3e0, 7.5, f"$i = {round(iterparam[1],2)}$")
                ax.grid(True,color = "grey", linestyle="-", linewidth="0.25", axis = "x", which = "both", alpha = 0.75)
                ax.vlines(1/(1-iterparam[0]), 0, 1, transform = ax.get_xaxis_transform(), colors = 'k', alpha = 0.5, label = r"Expected Peak $e$")
                # if j == 0 or j == 4 or j == 8:
                #     if j == 8:
                #         ax.tick_params(axis = "both", labelbottom = True, labelleft = True)
                #     else:
                #         ax.tick_params(axis = "both", labelbottom = False, labelleft = False)
                # else:
                #     ax.set_yticks([])
                #     ax.set_xticks([])
                #     ax.tick_params(axis = "both", labelbottom = False, labelleft = False)
                # if j == 11:
                #     handles = [patches.Rectangle((0,0),1,1,color = c, ec = "w") for c in colorlist]
                #     if which == "Linear":
                #         labels = ["Linear", "Peak Eccentricity"]
                #     else:
                #         labels = ["Linear", "Log", r"Power Law: $\alpha = 2$", r"Expected Peak $e$"]
                        
                #     ax.legend(handles = handles, labels = labels, loc = "best", fontsize = 10)

                ax.tick_params(axis = "both", labelbottom = False, labelleft = False)

                if w == 0:
                    value = "0"
                    text_string = "(a)"
                elif w == (np.pi)/6:
                    value = "pi_6"
                    text_string = "(b)"
                elif w == (np.pi)/3:
                    value = "pi_3"
                    text_string = "(c)"
                elif w == (np.pi)/2:
                    value = "pi_2"
                    text_string = "(d)"
                else:
                    return("Warning: Input correct version of pi (idk just do it right man).")
                    
                plt.figtext(0.94, 0.01, text_string, fontsize = 40)
                fig.tight_layout(pad = 2.0)

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
                    patch.set_edgecolor("g")
            else:
                for patch in patches_lin:
                    patch.set_edgecolor("g")
                for patch in patches_log:
                    patch.set_edgecolor("r")
                for patch in patches_linsemi:
                    patch.set_edgecolor("b")

            limit = [x for x in np.arange(0.5, end + 0.5 , 0.5)]
            axs.set_xlim(0.5,20.5)
            axs.set_ylim(0,10)
            axs.set_xticks(limit)
            # axs.set_yticks(limit)
            axs.set_xscale("log")
            
            axs.set_xlabel(r"Semimajor Axis [$\log{a/R_e}$]")    
            axs.set_ylabel(r"Counts")
            textstr = "\n".join((f'e = {iterparam[0]}', rf'i = {round(np.degrees(param[1]))}$\degree$'))
            axs.text(0.05, 0.95, textstr, transform = axs.transAxes, fontsize = 20, verticalalignment = "top", bbox = rect)
            axs.grid(True,color = "grey", linestyle="--", linewidth="0.25", axis = "x", which = "both")
            axs.vlines(1/(1-iterparam[0]), 0, 1, transform = axs.get_xaxis_transform(), colors = 'k', alpha = 0.75, label = r"Expected Peak $e$")
            handles = [patches.Rectangle((0,0),1,1,color = c, ec = "w") for c in colorlist]
            if which == "Linear":
                labels = ["Linear", "Peak Eccentricity"]
            else:
                labels = ["Linear", "Log", r"Power Law: $\alpha = 2$", r"Expected Peak $e$"]
                
            axs.legend(handles = handles, labels = labels, loc = "upper right", fontsize = 25)
            plt.figtext(0.93, 0.01, "(b)", fontsize = 30)
            fig.tight_layout()

        if len(specify) == 0:
            try:
                 plt.savefig(f'/College_Projects/Microlensing Separation/Figures/MultiHist_omega_{value}.png')
            except OSError:
                plt.savefig(f'C:/Users/victo/College_Projects/Microlensing Separation/Figures/MultiHist_omega_{value}.png')
        else:
            try:
                plt.savefig(f"/College_Projects/Microlensing Separation/Figures/MultiHist_omega_{omega}_specified.png")
            except OSError:
                plt.savefig(f"C:/Users/victo/College_Projects/Microlensing Separation/Figures/MultiHist_omega_{omega}_specified.png")
            
        return rlist

    def CompleteHistGen(self, which = "Log", unity = False):
        """
        """
        step = 0.002
        end = 20
        
        
        # Dictionary for storing Rchange results
        totlinlist = []
        totloglist = []
        totsemilist = []
        histlistlin = []
        histlistlog = []
        histlistsemi = []
        tothistlist =[[] for _ in range(12)]
        evalhistlist = []
        total_lin = 0
        
        unity_data = {}
        # Create variables for bin sizes
        nbin = 200
        amin = 0.5
        amax = 21
        # Make logbinsizes for all
        logbinsize = (np.log10(amin)-np.log10(amax))/nbin
        logbins = np.geomspace(amin,amax, nbin)
        
        # For making the stepthrough of omega
        wstep = np.linspace(0,np.pi/2,wnum)
        for k in wstep:
            print("Value of omega currently: ", k, " and current position in array: ", np.where(wstep == k))
            # Each omega calculates its own data groups
            # omega marginalized
            steptotlist, param = Sep_plot.DataHist(w = k, step = step, end = end, which = which)
            # Once complete, takes the data through each set
            # ONLY OMEGA
            for j in range(len(steptotlist)):
                steplindict, x, y, steplogdict, stepsemidict = steptotlist[j]
                iter_param = param[j]
                eccent, incl = iter_param[0], iter_param[1]
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
                hist_power, histbins_power = np.histogram(totsemilist,bins = logbins, range=(0.5, end+0.5))
                if k != 0.0:
                    histlistiter = histlistsemi[j]
                    hist_power_iter, bins_power_iter = histlistiter
                    hist_power_iter = hist_power_iter + hist_power
                    histlistsemi[j] = (hist_power_iter, bins_power_iter)
                else:
                    histlistsemi.append((hist_power, histbins_power))

                tothist_lin = np.zeros_like(hist_lin)
                tothist_log = np.zeros_like(hist_lin)
                tothist_power = np.zeros_like(hist_lin)
        
                gc.collect()
                # Collect iterated data
                for val in range(len(histlistsemi)):
                        hist_lin, bins = histlistlin[val]
                        hist_log, bins = histlistlog[val]
                        hist_power, bins = histlistsemi[val]
                        if val == 0:
                            tothist_lin = tothist_lin + hist_lin
                            tothist_log = tothist_log + hist_log
                            tothist_power = tothist_power + hist_power
                        elif val == len(histlistsemi)-1 and j == len(histlistsemi)-1:
                            if total_lin == 0:
                                total_lin = np.sum(tothist_lin)
                                total_log = np.sum(tothist_log)
                                total_power = np.sum(tothist_power)
                            else:
                                total_lin += np.sum(tothist_lin)
                                total_log += np.sum(tothist_log)
                                total_power += np.sum(tothist_power)
                        else:
                            tothist_lin = tothist_lin + hist_lin
                            tothist_log = tothist_log + hist_log
                            tothist_power = tothist_power + hist_power
                unity_data[f"final lin {eccent} {round(incl,3)}"] = tothist_lin
                unity_data[f"final log {eccent} {round(incl,3)}"] = tothist_log
                unity_data[f"final power {eccent} {round(incl,3)}"] = tothist_power
        print("Total Number of Linear Points: ", total_lin)
        print("Total Number of Log Points: ", total_log)
        print("Total Number of Power Points: ", total_power)
        # Save to CSV
        df_unity = pd.DataFrame(unity_data)
        
        if unity:
            file_name = f'/home/karkour.2/Results/UnityHist_omegas_{self.wnum}_{which}.csv'
        else:
            try:
                file_name = f'/College_Projects/Microlensing Separation/Results/UnityHist_omegas_{self.wnum}_{which}.csv'
                df_unity.to_csv(file_name, index = False)
            except OSError:
                file_name = f'/Users/victo/College_Projects/Microlensing Separation/Results/UnityHist_omegas_{self.wnum}_{which}.csv'
        df_unity.to_csv(file_name, index = False)
        return tothistlist, evalhistlist

    def CompleteHistLoad(self, which = "Log", inclination = False):
        """
        """
        if inclination:
            try:
                file_name = f'/College_Projects/Microlensing Separation/Results/UnityHist_inclines_{self.wnum}_{which}.csv'
                pd.read_csv(file_name)
            except FileNotFoundError:
                file_name = f'/Users/victo/College_Projects/Microlensing Separation/Results/UnityHist_inclines_{self.wnum}_{which}.csv'
        else:
            try:
                file_name = f'/College_Projects/Microlensing Separation/Results/UnityHist_omegas_{self.wnum}_{which}.csv'
                pd.read_csv(file_name)
            except FileNotFoundError:
                file_name = f'/Users/victo/College_Projects/Microlensing Separation/Results/UnityHist_omegas_{self.wnum}_{which}.csv'
        df = pd.read_csv(file_name)
        if inclination:
            eccent = np.linspace(0,0.99,12)
        else:
            multi_param = [
                # Row 1
                (0. , 0.,), (0., np.pi/6, ), (0., np.pi/3, ), (0., np.pi/2,),
                # Row 2
                (0.5, 0., ), (0.5, np.pi/6, ), (0.5, np.pi/3, ), (0.5,np.pi/2,),
                # Row 3
                (0.9 , 0.,), (0.9, np.pi/6,), (0.9, np.pi/3,), (0.9, np.pi/2,) 
            ]
                
        # Create variables for bin sizes
        nbin = 200
        amin = 0.5
        amax = 21
        # Make logbinsizes for all
        logbinsize = (np.log10(amin)-np.log10(amax))/nbin
        logbins = np.geomspace(amin,amax, nbin)
        if not inclination:
            colorlist = ["green", "red", "blue", "black"]
        else:
            if which == "Linear":
                colorlist = ["green", "black"]
            elif which == "Log":
                colorlist = ["red", "black"]
            else:
                colorlist = ["blue", "black"]
        fig, axs = plt.subplots(3,4, figsize = (13,9), sharex=False,sharey=False,gridspec_kw=dict(hspace=0,wspace=0))
        # fig.suptitle("Detections of $R_E$ with marginalizations for "r"$\cos{i} = 0$ to 1 , and " r"$\omega$ = $0$ to $\frac{\pi}{2}$" f"\n ({which})")
        for j, ax  in enumerate(axs.flatten()):
            if not inclination:
                iter_param = multi_param[j]
                hist_lin = df[f"final lin {iter_param[0]} {round(iter_param[1],3)}"]
                hist_log = df[f"final log {iter_param[0]} {round(iter_param[1],3)}"]
                hist_power = df[f"final power {iter_param[0]} {round(iter_param[1],3)}"]

                            
                norm_lin = np.abs(1 / (logbinsize * hist_lin.sum()))
                norm_log = np.abs(1 / (logbinsize * hist_log.sum()))
                norm_power = np.abs(1 / (logbinsize * hist_power.sum()))
                StepPatch_lin = ax.stairs(hist_lin * norm_lin, logbins, edgecolor = colorlist[0], fill = False, alpha = 0.5, label = "Linear")
                StepPatch_log = ax.stairs(hist_log * norm_log, logbins, edgecolor = colorlist[1], fill = False, alpha = 0.5, label = "Log")
                StepPatch_power = ax.stairs(hist_power * norm_power, logbins, edgecolor = colorlist[2], fill = False, alpha = 0.5, label = "Power")
            else:
                iter_param = eccent[j]
                hist = df[f"final lin {round(iter_param,3)}"]

                norm = np.abs(1 / (logbinsize * hist.sum()))
                StepPatch = ax.stairs(hist * norm, logbins, edgecolor = colorlist[0], fill = False, alpha = 0.5, label = f"{which}")
                
            # Limits
            limit = [x for x in np.arange(0.5, 20 + 0.5 , 0.5)]
            ax.set_xlim(0.5,20.5)
            ax.set_ylim(0,10)
            ax.set_xticks(limit)
            ax.set_xscale("log")
            ax.set_yticks([0,2,4,6,8,10])
            
            # Decoration
            rect = dict(boxstyle = "round", alpha = 1, facecolor = "white")
            if not inclination:
                textstr = "\n".join((f'e = {iter_param[0]}', rf'i = {round(np.degrees(iter_param[1]))}$\degree$'))
                ax.vlines(1/(1-iter_param[0]), 0, 1, transform = ax.get_xaxis_transform(), colors = 'black', alpha = 0.75, label = r"Expected Peak $e$")
            else:
                textstr = f"e = {round(iter_param,3)}"
                ax.vlines(1/(1-iter_param), 0, 1, transform = ax.get_xaxis_transform(), colors = 'black', alpha = 0.75, label = r"Expected Peak $e$")
            ax.text(0.05, 0.95, textstr, transform = ax.transAxes, fontsize = 20, verticalalignment = "top", bbox = rect)
            ax.grid(True,color = "grey", linestyle="--", linewidth="0.25", axis = "x", which = "both")
            # Lines and organizing labels to be cleaner
            
            if j == 8:
                ax.tick_params(axis = "both", labelbottom = True, labelleft = True) 
            else:
                ax.set_yticks([])
                ax.set_xticks([])
                ax.tick_params(axis = "both", labelbottom = False, labelleft = False)
            if j == 11:
                handles = [patches.Rectangle((0,0),1,1,color = c, ec = "w") for c in colorlist]
                if inclination:
                    labels = [f"{which}", r"Expected Peak $e$"]
                else:
                    labels = ["Linear", "Log", r"Power Law: $\alpha = 2$", r"Expected Peak $e$"]
                ax.legend(handles = handles, labels = labels, loc = "best", fontsize = 15)    

        fig.tight_layout(pad = 0.5)
        # Saves plot
        if not inclination:
            try:
                plt.savefig(f'/College_Projects/Microlensing Separation/Figures/CompleteHist_{wnum}_LinLogPower.png')
            except OSError:
                plt.savefig(f"C:/Users/victo/College_Projects/Microlensing Separation/Figures/CompleteHist_{wnum}_LinLogPower.png")
        else:
            try:
                plt.savefig(f'/College_Projects/Microlensing Separation/Figures/CompleteHist_incline_{inum}_{which}.png')
            except OSError:
                plt.savefig(f"C:/Users/victo/College_Projects/Microlensing Separation/Figures/CompleteHist_incline_{inum}_{which}.png")


        return df

    def UnityPlotHistGen(self, which, unity = False, circ = False, gamma_bool = False, inclination = False):
        """
        """
        totlist = []
        gammalist = []
        evalcirc = []
        # Initialize eccentricity marginalizaitons
        estep = np.linspace(0,0.99, self.numestep)
        x = np.linspace(0,0.98, self.wnum)
        esteplist = []*self.numdiv
        param = []
        unity_data = {}
        
        
        # Slices estep into parts for parallelization
        base = self.numestep // self.numdiv
        rem = self.numestep % self.numdiv
        sizes = []
        for i in range(self.numdiv):
            add = 1 if i < rem else 0 # DISTRIBUTES REMAINDER INTO FIRST COUPLE SLICES
            sizes.append(base + add) # SLICES OUT OF numdiv
        idx = 0
        for i, sz in enumerate(sizes):
            slices = estep[idx: idx + sz] # SLICES estep BASED ON sizes variable
            esteplist.append(slices) 
            idx += sz # INCREMENTS TO WHATEVER sz WAS INITIALLY
        if circ == False:
            for i in range(self.numdiv):
                obj = Sep_gen()
                # Step, end, inclincation, which, estep_iter, omega, incl, estep, class
                if gamma_bool:
                    param.append((0.002, 20, True, which, esteplist[i], self.wnum, self.inum, estep, repeat(obj)))
            if inclination and not gamma_bool:
                eccent = np.linspace(0,0.99,12)
                for val in eccent:
                    param.append((0.002, 20, True, which, val, self.wnum, self.inum, None ,repeat(obj)))
        else:
            obj = Sep_gen()
            param = [0.002, 20, True, which, estep[0], self.wnum, self.inum, repeat(obj)]
        # Processing using parallelization
        if not circ and gamma_bool:     
            with Pool(processes = self.numdiv) as pool:
                tothistlist = pool.map(Sep_gen.HistGen, param)
                # tothistlist = Sep_gen.HistGen(param[2])
        elif circ:
            tothistlist = Sep_gen.CircHistGen(param)
        elif inclination:
            with Pool(processes = 12) as pool:
                tothistlist = pool.map(Sep_gen.HistGen, param)
                # tothistlist = Sep_gen.HistGen(param[0])
        # print("pool finished")
        if gamma_bool:
            for j in range(len(tothistlist)):
                totlist.append(tothistlist[j][0])
                gammalist.append(tothistlist[j][1])
        else:
                for j in range(len(tothistlist)):
                    totlist.append(tothistlist[j][0])
        # Process for CSV File
        if circ == False and gamma_bool:
            for i in range(len(totlist)):
                histlist = totlist[i]
                gammahistlist = gammalist[i]
                for val in range(len(histlist)):
                        hist, bins = histlist[val]
                        gamma_val, bins_gamma = gammahistlist[val]
                        if val == 0:
                            if i == 0:
                                tothist = np.zeros_like(hist)
                                totgamma = np.zeros_like(hist)
                            tothist = tothist + hist
                            totgamma = totgamma + gamma_val
                        elif val == len(histlist)-1 and i == len(totlist)-1:
                            tothist = tothist + hist
                            totgamma = totgamma + gamma_val
                            total = np.sum(tothist)
                            tot_gamma = np.sum(totgamma)
                        else:
                            tothist = tothist + hist
                            totgamma = totgamma + gamma_val
            print("Total Number of Points: ", total)
            print("Total Gamma: ", tot_gamma)
            # Save to CSV
            unity_data = {
                        "final list": tothist,
                        "gamma list": totgamma
                        }
        elif circ:
            for i in range(len(totlist)):
                histlist = totlist[i]
                for val in range(len(histlist)):
                        hist, bins = histlist[val]
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
                "circular list": tothist
                }
        else:
            for i, k in enumerate(eccent):
                histlist = totlist[i]
                for j in range(len(histlist)):
                    histlist_iter = histlist[j]
                    for val in range(len(histlist_iter[0])):
                            hist, bins = histlist_iter
                            if val == 0:
                                if i == 0:
                                    tothist = np.zeros_like(hist)
                                tothist = tothist + hist
                            elif val == len(histlist_iter)-1 and i == len(totlist)-1:
                                tothist = tothist + hist
                                total = np.sum(tothist)
                            else:
                                tothist = tothist + hist
                unity_data[f"final lin {round(k,3)}"] = tothist
            print("Total Number of Points: ", total)
            # Save to CSV
        df_unity = pd.DataFrame(unity_data)
        if unity:
            if not circ and not inclination:
                file_name = f'/home/karkour.2/Results/UnityHist_eccent_incline_{self.numestep}_0002_{which}.csv'
            elif inclination:
                file_name = f'/home/karkour.2/Results/UnityHist_inclines_{self.inum}_{which}.csv'
            else:
                file_name = f'/home/karkour.2/Results/UnityHist_eccent_incline_{self.wnum}_0002_circular_{which}.csv'
        else:
            if circ == False and gamma_bool:
                try:
                    file_name = f'/College_Projects/Microlensing Separation/Results/UnityHist_eccent_incline_{self.numestep}_0002_{which}.csv'
                    df_unity.to_csv(file_name, index = False)
                except OSError:
                    file_name = f'/Users/victo/College_Projects/Microlensing Separation/Results/UnityHist_eccent_incline_{self.numestep}_0002_{which}.csv'
            elif inclination:
                try:
                    file_name = f'/College_Projects/Microlensing Separation/Results/UnityHist_inclines_{self.inum}_{which}.csv'
                    df_unity.to_csv(file_name, index = False)
                except OSError:
                    file_name = f'/Users/victo/College_Projects/Microlensing Separation/Results/UnityHist_inclines_{self.inum}_{which}.csv'
            else:
                try:
                    file_name = f'/College_Projects/Microlensing Separation/Results/UnityHist_eccent_incline_{self.wnum}_0002_circular_{which}.csv'
                    df_unity.to_csv(file_name, index = False)
                except OSError:
                    file_name = f'/Users/victo/College_Projects/Microlensing Separation/Results/UnityHist_eccent_incline_{self.wnum}_0002_circular_{which}.csv'
                
        df_unity.to_csv(file_name, index = False)

        print("File saved successfully")

        return x

    def UnityPlotHistLoad(self, which, alpha_step = 0, dist = "", circ = False, test = False):
        """
        Using various specifications, will load plots of histograms with marginalizations in omega, inclination, and eccentricity
        
        --------
        ### Parameters
        
        which: String <br>
            The type of step (Linear, Log, or Power) used for generation
            
            Used to find file names and to help with comparing with values from alpha_step
            
        alpha_step: integer <br>
            Type of steps done through alpha 
            
            Ex: Log = -1 ; Linear = 0, Power = 1
            
            Used to find values to compare with the which parameter
            
        dist : String <br>
            Specify what type of distribution you want
            
            Used with parameters alpha_step and which to compare alpha values
            
            Note: Can only be circular, uniform, or gamma distributions
            
        circ : Boolean <br>
            Checks if you want to specifically plot one circular distribution using the which parameter  
        
        test : Boolean <br>
            Checks if you want to compare certain values of projected alpha for specific distributions

            Used alongside alpha_step, dist, and which to compare values of alpha step
        """
        if dist == "circular" or dist == "uniform" or dist == "gamma":
            
            # Reads files for all alpha values of a given step type (soon to be implemented [WAITING ON LOG TO BE DONE])
            try:
                # FOR FUTURE TESTS, PUT BEST TYPE OF STEP (Linear, Log, [Soon to be Power]) AS THE WHICH FOR EACH FILENAME (Look at stepalpha func.)
                
                file_name_1 = f'/College_Projects/Microlensing Separation/Results/UnityHist_eccent_incline_{self.numestep}_0002_{which}_alpha_{-2}.csv'
                file_name_2 = f'/College_Projects/Microlensing Separation/Results/UnityHist_eccent_incline_{self.numestep}_0002_{which}_alpha_{-1}.csv'
                file_name_3 = f'/College_Projects/Microlensing Separation/Results/UnityHist_eccent_incline_{self.numestep}_0002_{which}_alpha_{0}.csv'
                file_name_4 = f'/College_Projects/Microlensing Separation/Results/UnityHist_eccent_incline_{self.numestep}_0002_{which}_alpha_{1}.csv'
                file_name_5 = f'/College_Projects/Microlensing Separation/Results/UnityHist_eccent_incline_{self.numestep}_0002_{which}_alpha_{2}.csv'   
                pd.read_csv(file_name_1)
                
                if test:
                    test_1 = f'/College_Projects/Microlensing Separation/Results/UnityHist_{self.numestep}_alpha_{alpha_step}_test.csv'
                    print("Test Flag Activated")
            except FileNotFoundError:
                    
                file_name_1 = f'/Users/victo/College_Projects/Microlensing Separation/Results/UnityHist_eccent_incline_{self.numestep}_0002_{which}_alpha_{-2}.csv'
                file_name_2 = f'/Users/victo/College_Projects/Microlensing Separation/Results/UnityHist_eccent_incline_{self.numestep}_0002_{which}_alpha_{-1}.csv'
                file_name_3 = f'/Users/victo/College_Projects/Microlensing Separation/Results/UnityHist_eccent_incline_{self.numestep}_0002_{which}_alpha_{0}.csv'
                file_name_4 = f'/Users/victo/College_Projects/Microlensing Separation/Results/UnityHist_eccent_incline_{self.numestep}_0002_{which}_alpha_{1}.csv'
                file_name_5 = f'/Users/victo/College_Projects/Microlensing Separation/Results/UnityHist_eccent_incline_{self.numestep}_0002_{which}_alpha_{2}.csv'
                
                if test:
                        test_1 = f'/Users/victo/College_Projects/Microlensing Separation/Results/UnityHist_{self.numestep}_alpha_{alpha_step}_test.csv'
                        print("Test Flag Activated")
                        
            # Dataframe creation
            if test:
                df_unity_test = pd.read_csv(test_1)
                df_unity_1 = pd.read_csv(file_name_1)
                df_unity_2 = pd.read_csv(file_name_2)
                df_unity_3 = pd.read_csv(file_name_3)
                df_unity_4 = pd.read_csv(file_name_4)
                df_unity_5 = pd.read_csv(file_name_5)
        else:
            if not circ:
                if test:
                    print("Test flag activated")
                    try:
                        file_name_1 = f'/College_Projects/Microlensing Separation/Results/UnityHist_eccent_incline_{self.numestep}_0002_{which}_alpha_{alpha_step}.csv'
                        file_name_2 = f'/College_Projects/Microlensing Separation/Results/UnityHist_eccent_incline_{self.numestep}_0002_Linear_alpha_{alpha_step}.csv'
                        pd.read_csv(file_name_1)
                    except FileNotFoundError:
                        file_name_1 = f'/Users/victo/College_Projects/Microlensing Separation/Results/UnityHist_eccent_incline_{self.numestep}_0002_{which}_alpha_{alpha_step}.csv'
                        file_name_2 = f'/Users/victo/College_Projects/Microlensing Separation/Results/UnityHist_eccent_incline_8_0002_Linear_alpha_{alpha_step}.csv'
                    df_log = pd.read_csv(file_name_1)
                    df_lin = pd.read_csv(file_name_2)
                    
                try:
                    file_name = f'/College_Projects/Microlensing Separation/Results/UnityHist_eccent_incline_{self.numestep}_0002_{which}_alpha_{alpha_step}.csv'
                    pd.read_csv(file_name)
                    df_unity = pd.read_csv(file_name)
                except FileNotFoundError:
                    file_name = f'/Users/victo/College_Projects/Microlensing Separation/Results/UnityHist_eccent_incline_{self.numestep}_0002_{which}_alpha_{alpha_step}.csv'
                    df_unity = pd.read_csv(file_name)
            else:
                try:
                    file_name = f'/College_Projects/Microlensing Separation/Results/UnityHist_eccent_incline_{self.wnum}_0002_circular_{which}.csv'
                    pd.read_csv(file_name)
                except FileNotFoundError:
                    file_name = f'/Users/victo/College_Projects/Microlensing Separation/Results/UnityHist_eccent_incline_{self.wnum}_0002_circular_{which}.csv'
                df_unity = pd.read_csv(file_name)

        # Create variables for bin sizes
        nbin = 200
        amin = 0.5
        amax = 21
        # Make logbinsizes for all
        logbinsize = np.abs((np.log10(amin)-np.log10(amax))/nbin)
        
        # Initialize Lists
        labels = ["Uniform", "Gamma", "Circular"]
        colorlist = ["green", "red", "blue"]
        # Make bins
        bins = np.geomspace(amin,amax, nbin)
        
        # FIGURE FOR SEMIMAJOR AXIS
        fig, ax = plt.subplots(figsize = (9,9), sharex=True,sharey=True,gridspec_kw=dict(hspace=0,wspace=0))        
        # Combine Histogram Calculations
        if len(dist) != 0:
            if dist == "circular":
                circhist_1 = df_unity_1["circular list"].to_numpy()
                circhist_2 = df_unity_2["circular list"].to_numpy()
                circhist_3 = df_unity_3["circular list"].to_numpy()
                circhist_4 = df_unity_4["circular list"].to_numpy()
                circhist_5 = df_unity_5["circular list"].to_numpy()
                
                ecircnorm_1 = np.abs(1 / (np.sum(circhist_1) * logbinsize))
                ecircnorm_2 = np.abs(1 / (np.sum(circhist_2) * logbinsize))
                ecircnorm_3 = np.abs(1 / (np.sum(circhist_3) * logbinsize))
                ecircnorm_4 = np.abs(1 / (np.sum(circhist_4) * logbinsize))
                ecircnorm_5 = np.abs(1 / (np.sum(circhist_5) * logbinsize))
                
                if test:
                    circhist_test_minus_2 = df_unity_test["alpha -2 circ"].to_numpy()
                    circhist_test_minus_1 = df_unity_test["alpha -1 circ"].to_numpy()
                    circhist_test_0 = df_unity_test["alpha 0 circ"].to_numpy()
                    circhist_test_plus_1 = df_unity_test["alpha 1 circ"].to_numpy()
                    circhist_test_plus_2 = df_unity_test["alpha 2 circ"].to_numpy()
                    
                    ecirc_test_minus_2 = np.abs(1 / (np.sum(circhist_test_minus_2) * logbinsize))
                    ecirc_test_minus_1 = np.abs(1 / (np.sum(circhist_test_minus_1) * logbinsize))
                    ecirc_test_0 = np.abs(1 / (np.sum(circhist_test_0) * logbinsize))
                    ecirc_test_plus_1 = np.abs(1 / (np.sum(circhist_test_plus_1) * logbinsize))
                    ecirc_test_plus_2 = np.abs(1 / (np.sum(circhist_test_plus_2) * logbinsize))
                    
                    StepPatch = ax.stairs(circhist_test_minus_2 * ecirc_test_minus_2, bins, linestyle = "--", fill = False, label = f"projected alpha = -2 from alpha = {alpha_step}") 
                    StepPatch = ax.stairs(circhist_test_minus_1 * ecirc_test_minus_1, bins, linestyle = "--", fill = False, label = f"projected alpha = -1 from alpha = {alpha_step}")
                    StepPatch = ax.stairs(circhist_test_0 * ecirc_test_0, bins, linestyle = "--", fill = False, label = f"projected alpha = 0 from alpha = {alpha_step}") 
                    StepPatch = ax.stairs(circhist_test_plus_1 * ecirc_test_plus_1, bins, linestyle = "--", fill = False, label = f"projected alpha = 1 from alpha = {alpha_step}")
                    StepPatch = ax.stairs(circhist_test_plus_2 * ecirc_test_plus_2, bins, linestyle = "--", fill = False, label = f"projected alpha = 2 from alpha = {alpha_step}")
                    
                StepPatch = ax.stairs(circhist_1 * ecircnorm_1, bins, fill = False, lw = 0.5, label = f"Circular dist. for alpha = -2 ({which})") 
                StepPatch = ax.stairs(circhist_2 * ecircnorm_2, bins, fill = False, lw = 0.5, label = f"alpha = -1 ({which})")
                StepPatch = ax.stairs(circhist_3 * ecircnorm_3, bins, fill = False, lw = 0.5, label = f"alpha = 0 ({which})") 
                StepPatch = ax.stairs(circhist_4 * ecircnorm_4, bins, fill = False, lw = 0.5, label = f"alpha = 1 ({which})")
                StepPatch = ax.stairs(circhist_5 * ecircnorm_5, bins, fill = False, lw = 0.5, label = f"alpha = 2 ({which})")
            elif dist == "uniform":
                uniformhist_1 = df_unity_1["final list"].to_numpy()
                uniformhist_2 = df_unity_2["final list"].to_numpy()
                uniformhist_3 = df_unity_3["final list"].to_numpy()
                uniformhist_4 = df_unity_4["final list"].to_numpy()
                uniformhist_5 = df_unity_5["final list"].to_numpy()
                
                norm_1 = np.abs(1 / (np.sum(uniformhist_1) * logbinsize))
                norm_2 = np.abs(1 / (np.sum(uniformhist_2) * logbinsize))
                norm_3 = np.abs(1 / (np.sum(uniformhist_3) * logbinsize))
                norm_4 = np.abs(1 / (np.sum(uniformhist_4) * logbinsize))
                norm_5 = np.abs(1 / (np.sum(uniformhist_5) * logbinsize))
                
                if test:
                    uniformhist_test_minus_2 = df_unity_test["alpha -2"].to_numpy()
                    uniformhist_test_minus_1 = df_unity_test["alpha -1"].to_numpy()
                    uniformhist_test_0 = df_unity_test["alpha 0"].to_numpy()
                    uniformhist_test_plus_1 = df_unity_test["alpha 1"].to_numpy()
                    uniformhist_test_plus_2 = df_unity_test["alpha 2"].to_numpy()
                    
                    norm_test_minus_2 = np.abs(1 / (np.sum(uniformhist_test_minus_2) * logbinsize))
                    norm_test_minus_1 = np.abs(1 / (np.sum(uniformhist_test_minus_1) * logbinsize))
                    norm_test_0 = np.abs(1 / (np.sum(uniformhist_test_0) * logbinsize))
                    norm_test_plus_1 = np.abs(1 / (np.sum(uniformhist_test_plus_1) * logbinsize))
                    norm_test_plus_2 = np.abs(1 / (np.sum(uniformhist_test_plus_2) * logbinsize))
                    
                    StepPatch = ax.stairs(uniformhist_test_minus_2 * norm_test_minus_2, bins, linestyle = "--", fill = False, label = f"projected alpha = -2 from alpha = {alpha_step}") 
                    StepPatch = ax.stairs(uniformhist_test_minus_1 * norm_test_minus_1, bins, linestyle = "--", fill = False, label = f"projected alpha = -1 from alpha = {alpha_step}")
                    StepPatch = ax.stairs(uniformhist_test_0 * norm_test_0, bins, linestyle = "--", fill = False, label = f"projected alpha = 0 from alpha = {alpha_step}") 
                    StepPatch = ax.stairs(uniformhist_test_plus_1 * norm_test_plus_1, bins, linestyle = "--", fill = False, label = f"projected alpha = 1 from alpha = {alpha_step}")
                    StepPatch = ax.stairs(uniformhist_test_plus_2 * norm_test_plus_2, bins, linestyle = "--", fill = False, label = f"projected alpha = 2 from alpha = {alpha_step}")
                
                StepPatch = ax.stairs(uniformhist_1 * norm_1, bins, fill = False, lw = 0.5, label = f"Uniform dist. for alpha = -2 ({which})")
                StepPatch = ax.stairs(uniformhist_2 * norm_2, bins, fill = False, lw = 0.5, label = f"alpha = -1 ({which})")
                StepPatch = ax.stairs(uniformhist_3 * norm_3, bins, fill = False, lw = 0.5, label = f"alpha = 0 ({which})")
                StepPatch = ax.stairs(uniformhist_4 * norm_4, bins, fill = False, lw = 0.5, label = f"alpha = 1 ({which})")
                StepPatch = ax.stairs(uniformhist_5 * norm_5, bins, fill = False, lw = 0.5, label = f"alpha = 2 ({which})")
            elif dist == "gamma":
                gammahist_1 = df_unity_1["gamma list"].to_numpy()
                gammahist_2 = df_unity_2["gamma list"].to_numpy()
                gammahist_3 = df_unity_3["gamma list"].to_numpy()
                gammahist_4 = df_unity_4["gamma list"].to_numpy()
                gammahist_5 = df_unity_5["gamma list"].to_numpy()
                
                gammanorm_final_1 = np.abs(1/ (np.sum(gammahist_1) * logbinsize))
                gammanorm_final_2 = np.abs(1/ (np.sum(gammahist_2) * logbinsize))
                gammanorm_final_3 = np.abs(1/ (np.sum(gammahist_3) * logbinsize))
                gammanorm_final_4 = np.abs(1/ (np.sum(gammahist_4) * logbinsize))
                gammanorm_final_5 = np.abs(1/ (np.sum(gammahist_5) * logbinsize))
                
                if test:
                    gammahist_test_minus_2 = df_unity_test["alpha -2 gamma"].to_numpy()
                    gammahist_test_minus_1 = df_unity_test["alpha -1 gamma"].to_numpy()
                    gammahist_test_0 = df_unity_test["alpha 0 gamma"].to_numpy()
                    gammahist_test_plus_1 = df_unity_test["alpha 1 gamma"].to_numpy()
                    gammahist_test_plus_2 = df_unity_test["alpha 2 gamma"].to_numpy()
                                        
                    gammanorm_test_minus_2 = np.abs(1 / (np.sum(gammahist_test_minus_2) * logbinsize))
                    gammanorm_test_minus_1 = np.abs(1 / (np.sum(gammahist_test_minus_1) * logbinsize))
                    gammanorm_test_0 = np.abs(1 / (np.sum(gammahist_test_0) * logbinsize))
                    gammanorm_test_plus_1 = np.abs(1 / (np.sum(gammahist_test_plus_1) * logbinsize))
                    gammanorm_test_plus_2 = np.abs(1 / (np.sum(gammahist_test_plus_2) * logbinsize))
                    
                    StepPatch = ax.stairs(gammahist_test_minus_2 * gammanorm_test_minus_2, bins, linestyle = "--", fill = False, label = f"projected alpha = -2 from alpha = {alpha_step}") 
                    StepPatch = ax.stairs(gammahist_test_minus_1 * gammanorm_test_minus_1, bins, linestyle = "--", fill = False, label = f"projected alpha = -1 from alpha = {alpha_step}")
                    StepPatch = ax.stairs(gammahist_test_0 * gammanorm_test_0, bins, linestyle = "--", fill = False, label = f"projected alpha = 0 from alpha = {alpha_step}") 
                    StepPatch = ax.stairs(gammahist_test_plus_1 * gammanorm_test_plus_1, bins, linestyle = "--", fill = False, label = f"projected alpha = 1 from alpha = {alpha_step}")
                    StepPatch = ax.stairs(gammahist_test_plus_2 * gammanorm_test_plus_2, bins, linestyle = "--", fill = False, label = f"projected alpha = 2 from alpha = {alpha_step}")
                
                StepPatch = ax.stairs(gammahist_1 * gammanorm_final_1, bins, fill = False, lw = 0.5, label = f"Gamma dist. for alpha = -2 ({which})")
                StepPatch = ax.stairs(gammahist_2 * gammanorm_final_2, bins, fill = False, lw = 0.5, label = f"alpha = -1 ({which})")
                StepPatch = ax.stairs(gammahist_3 * gammanorm_final_3, bins, fill = False, lw = 0.5, label = f"alpha = 0 ({which})")
                StepPatch = ax.stairs(gammahist_4 * gammanorm_final_4, bins, fill = False, lw = 0.5, label = f"alpha = 1 ({which})")
                StepPatch = ax.stairs(gammahist_5 * gammanorm_final_5, bins, fill = False, lw = 0.5, label = f"alpha = 2 ({which})")           
        else:
            if not circ:
                if test:
                    uniformhist_lin = df_lin["final list"].to_numpy()
                    uniformhist_log = df_log["final list"].to_numpy()
                    
                    norm_lin = np.abs(1 / (np.sum(uniformhist_lin) * logbinsize))
                    norm_log = np.abs(1 / (np.sum(uniformhist_log) * logbinsize))
                    
                    gammahist_lin = df_lin["gamma list"].to_numpy()
                    gammahist_log = df_log["gamma list"].to_numpy()
                    
                    gammanorm_lin = np.abs(1/ (np.sum(gammahist_lin) * logbinsize))
                    gammanorm_log = np.abs(1/ (np.sum(gammahist_log) * logbinsize))    
                    
                    circhist_lin = df_lin["circular list"].to_numpy()
                    circhist_log = df_log["circular list"].to_numpy()
                    
                    circnorm_lin = np.abs(1 / (np.sum(circhist_lin) * logbinsize))
                    circnorm_log = np.abs(1 / (np.sum(circhist_log) * logbinsize))
                    
                    StepPatch = ax.stairs(uniformhist_lin * norm_lin, bins, edgecolor = "g", fill = False, alpha = 0.5, label = f"Uniform Dist. (Linear @ alpha = {alpha_step})") # Uniform Dist (Linear)
                    StepPatch = ax.stairs(gammahist_lin * gammanorm_lin, bins, edgecolor = "r", fill = False, alpha = 0.5, label = f"Gamma Dist. (Linear)") # Gamma Dist (Linear)
                    StepPatch = ax.stairs(circhist_lin * circnorm_lin, bins, edgecolor = "b", fill = False, alpha = 0.5, label = f"Circular Dist. (Linear)") # Circ Dist (Linear)
                    
                    StepPatch = ax.stairs(uniformhist_log * norm_log, bins, edgecolor = "black", fill = False, ls = "--", alpha = 1, label = f"Uniform Dist. (Log @ alpha = {alpha_step})") # Uniform Dist (Log)
                    StepPatch = ax.stairs(gammahist_log * gammanorm_log, bins, edgecolor = "purple", fill = False, ls = "--", alpha = 1, label = f"Gamma Dist. (Log)") # Gamma Dist (Log)
                    StepPatch = ax.stairs(circhist_log * circnorm_log, bins, edgecolor = "orange", fill = False, ls = "--", alpha = 1, label = f"Circular Dist. (Log)") # Circ Dist (Log)
                else:
                    try:
                        circhist = df_unity["circular list"].to_numpy()
                        ecircnorm = np.abs(1 / (np.sum(circhist) * logbinsize))
                    except KeyError:
                        print("Circular not found, continuing without it....")
                    
                    uniformhist = df_unity["final list"].to_numpy()
                    gammahist = df_unity["gamma list"].to_numpy()
                    
                    norm = np.abs(1 / (np.sum(uniformhist) * logbinsize))
                    
                    gammanorm_final = np.abs(1/ (np.sum(gammahist) * logbinsize))
                    result = sum(uniformhist)
                    # print(result, sum(gammahist))

                    StepPatch = ax.stairs(uniformhist * norm, bins, edgecolor = colorlist[0], fill = False, label = "Uniform Dist.") # Uniform Dist
                    StepPatch = ax.stairs(gammahist * gammanorm_final, bins, edgecolor = colorlist[1], fill = False, label = "Gamma Dist.") # Gamma Dist
                    try:
                        StepPatch = ax.stairs(circhist * ecircnorm, bins, edgecolor = colorlist[2], fill = False, label = "Circular Dist.") # Circular Dist
                    except UnboundLocalError:
                        print()

                    if alpha_step == 0:
                        plt.figtext(0.935, 0.01, "(b)", fontsize = 30)
                    elif alpha_step == -1:
                        plt.figtext(0.935, 0.01, "(a)", fontsize = 30)
            else:
                circhist = df_unity["circular list"].to_numpy()

                ecircnorm = np.abs(1 / (np.sum(circhist) * logbinsize))
            
                print(sum(circhist))
            
                StepPatch = ax.stairs(circhist * ecircnorm, bins, edgecolor = colorlist[2], fill = False, label = "Circular Dist.") # Circular Dist


        ax.grid(True,color = "grey", linestyle="--", linewidth="0.25", axis = "x", which = "both")
        ax.set_xlim(0.5,20.5)
        ax.set_ylim(0,10)
        ax.set_xscale("log")
        ax.set_xlabel(r"Semimajor Axis [$\log{a/R_e}$]")    
        ax.set_ylabel(r"Counts")
        
        if len(dist) == 0:
            # handles = [patches.Rectangle((0,0),1,1,color = c, ec = "w") for c in colorlist]    
            ax.legend()
        else:
            ax.legend()
            
        fig.tight_layout()
        if len(dist) == 0:
            if circ == False and test == False:
                try:
                    plt.savefig(f'/College_Projects/Microlensing Separation/Figures/UnityHist_eccent_incline_{self.numestep}_0002_{which}_alpha_{alpha_step}_new.png')
                except OSError:
                    plt.savefig(f"C:/Users/victo/College_Projects/Microlensing Separation/Figures/UnityHist_eccent_incline_{self.numestep}_0002_{which}_alpha_{alpha_step}_new.png")
            else:
                if test:
                    try:    
                        plt.savefig(f'/College_Projects/Microlensing Separation/Figures/UnityHist_{self.numestep}_{alpha_step}_{which}_test.png')
                    except OSError:
                        plt.savefig(f"C:/Users/victo/College_Projects/Microlensing Separation/Figures/UnityHist_{self.numestep}_{alpha_step}_{which}_test.png")
                else:
                    try:    
                        plt.savefig(f'/College_Projects/Microlensing Separation/Figures/UnityHist_eccent_incline_{self.wnum}_0002_circ_{which}.png')
                    except OSError:
                        plt.savefig(f"C:/Users/victo/College_Projects/Microlensing Separation/Figures/UnityHist_eccent_incline_{self.wnum}_0002_circular_{which}.png")
        else:
            if test:
                try:
                    plt.savefig(f'/College_Projects/Microlensing Separation/Figures/UnityHist_{alpha_step}_{which}_{dist}_test.png')
                except OSError:
                    plt.savefig(f"C:/Users/victo/College_Projects/Microlensing Separation/Figures/UnityHist_{alpha_step}_{which}_{dist}_test.png")
            else:
                try:
                    plt.savefig(f'/College_Projects/Microlensing Separation/Figures/UnityHist_eccent_incline_{self.numestep}_0002_{which}_{dist}.png')
                except OSError:
                    plt.savefig(f"C:/Users/victo/College_Projects/Microlensing Separation/Figures/UnityHist_eccent_incline_{self.numestep}_0002_{which}_{dist}.png")
        return bins
    
    def statistics(self, which, alpha_step = 1, test = False):
        """
        """
         # Create variables for bin sizes
        nbin = 200
        amin = 0.5
        amax = 21
        # Make log bins for all
        bins = np.geomspace(amin,amax, nbin)

        try: 
            if test: # Only works for Log currently
                print("Test flag activated")
                filename = f'/College_Projects/Microlensing Separation/Results/UnityHist_eccent_incline_{self.numestep}_0002_{which}_alpha_{alpha_step}.csv'
                filename_2 = f'/College_Projects/Microlensing Separation/Results/UnityHist_eccent_incline_{self.numestep}_0002_{"Linear"}_alpha_{alpha_step}.csv'
            else: 
                filename = f'/College_Projects/Microlensing Separation/Results/UnityHist_eccent_incline_{self.numestep}_0002_{which}_alpha_{alpha_step}.csv'
            df_stats = pd.read_csv(filename)
        except FileNotFoundError:
            if test: # Only works for Log currently
                print("Test flag activated")
                filename = f'/Users/victo/College_Projects/Microlensing Separation/Results/UnityHist_eccent_incline_{self.numestep}_0002_{which}_alpha_{alpha_step}.csv'
                filename_2 = f'/Users/victo/College_Projects/Microlensing Separation/Results/UnityHist_eccent_incline_{self.numestep}_0002_{"Linear"}_alpha_{alpha_step}.csv'
            else:
                filename = f'/Users/victo/College_Projects/Microlensing Separation/Results/UnityHist_eccent_incline_{self.numestep}_0002_{which}_alpha_{alpha_step}.csv'

        df_stats = pd.read_csv(filename)
        if test:
            df_stats_2 = pd.read_csv(filename_2)
            df_stats_2["cumul_norm"] = np.cumsum(df_stats_2["final list"]) / np.abs(sum(df_stats_2["final list"]))
            df_stats_2["cumul_gamma"] = np.cumsum(df_stats_2["gamma list"]) / np.abs(sum(df_stats_2["gamma list"]))
            df_stats_2["cumul_circ"] = np.cumsum(df_stats_2["circular list"]) / np.abs(sum(df_stats_2["circular list"]))
            df_stats_2["bins"] = bins[:-1]
        df_stats["cumulative"] = 0
        df_stats["cumul_gamma"] = 0
        df_stats["cumul_circ"] = np.cumsum(df_stats["circular list"]) / np.abs(sum(df_stats["circular list"]))
        df_stats["bins"] = bins[:-1]


        hist = df_stats["final list"].to_numpy()
        # Make Gamma Calculation
        gammanorm = np.abs(1/ (np.sum(df_stats["gamma list"])))

        cumulative = 0
        cumul_gamma = 0
        cumul_norm = np.abs(1 / (sum(df_stats["final list"])))
        for i in range(len(df_stats["final list"])):
            cumulative = cumulative + df_stats.loc[i, "final list"]
            cumul_gamma = cumul_gamma + df_stats.loc[i, "gamma list"]
            df_stats.loc[i, "cumulative"] = cumulative
            df_stats.loc[i, "cumul_gamma"] = cumul_gamma
        df_stats["cumul_norm"] = df_stats["cumulative"] * cumul_norm
        df_stats["cumul_gamma"] = df_stats["cumul_gamma"] * gammanorm
        fig, ax = plt.subplots(figsize = (9,9), sharex=True,sharey=True,gridspec_kw=dict(hspace=0,wspace=0))
        fig.suptitle(f"Cumulative Distribution Function \n alpha = {alpha_step}")
        if test:
            ax.plot(bins[:-1], df_stats_2["cumul_norm"], ls = "-", c = "red", lw = 3, alpha = 0.5, label = f"Uniform Dist. (Linear @ alpha = {alpha_step})") # Normal Line NORMAL DIST.
            ax.plot(bins[:-1], df_stats_2["cumul_gamma"], ls = "--", c = "purple", lw = 3, alpha = 0.5, label = f"Gamma Dist. (Linear)") # Dashed Line GAMMA DIST.
            ax.plot(bins[:-1], df_stats_2["cumul_circ"], ls = ":", c = "orange", lw = 3, alpha = 0.5, label = f"Circular Dist. (Linear)") # Dotted Line CIRCULAR DIST.
            ax.plot(bins[:-1], df_stats["cumul_norm"], ls = "-", c = "g", lw = 3, alpha = 0.5, label = f"Uniform Dist. ({which} @ alpha = {alpha_step})") # Normal Line NORMAL DIST.
            ax.plot(bins[:-1], df_stats["cumul_gamma"], ls = "--", c = "r", lw = 3, alpha = 0.5, label = f"Gamma Dist. ({which})") # Dashed Line GAMMA DIST.
            ax.plot(bins[:-1], df_stats["cumul_circ"], ls = ":", c = "b", lw = 3, alpha = 0.5, label = f"Circular Dist. ({which})") # Dotted Line CIRCULAR DIST.
        else:
            ax.plot(bins[:-1], df_stats["cumul_norm"], ls = "-", c = "g", lw = 3, alpha = 0.75, label = "Uniform Dist.") # Normal Line NORMAL DIST.
            ax.plot(bins[:-1], df_stats["cumul_gamma"], ls = "--", c = "r", lw = 3, alpha = 0.75, label = "Gamma Dist.") # Dashed Line GAMMA DIST.
            ax.plot(bins[:-1], df_stats["cumul_circ"], ls = ":", c = "b", lw = 3, alpha = 0.75, label = "Circular Dist.") # Dotted Line CIRCULAR DIST.
        ax.set_xlim(0.5,20)
        ax.set_ylim(0,1)
        ax.set_xscale("log")
        ax.hlines(0.5, xmin = 0, xmax = 200, color = "k", ls = (0, (5, 8)), alpha = 0.75, lw = 3) # Loosely Dashed
        ax.hlines(0.5+(0.6827/2), xmin = 0, xmax = 200, color = "k", ls = (0, (1, 1)), alpha = 0.75, lw = 3) # Dotted
        ax.hlines(0.5-(0.6287/2), xmin = 0, xmax = 200, color = "k", ls = (0, (1, 1)), alpha = 0.75, lw = 3) # ^
        ax.hlines(0.5+(0.95/2), xmin = 0, xmax = 200, color = "k", ls = (0, (3, 5, 1, 5)), alpha = 0.75, lw = 3) # Dashdotted
        ax.hlines(0.5-(0.95/2), xmin = 0, xmax = 200, color = "k", ls = (0, (3, 5, 1, 5)), alpha = 0.75, lw = 3) # ^
        ax.set_xlabel(r"Semimajor Axis [$\log{a/R_e}$]")
        ax.set_ylabel(r"CDF")
        ax.legend(loc = "lower right")
        
        median = round(df_stats["bins"].loc[(df_stats["cumul_norm"] >= 0.495) & (df_stats["cumul_norm"] <= 0.515)].values[0],3)
        median_gamma = round(df_stats["bins"].loc[(df_stats["cumul_gamma"] >= 0.495) & (df_stats["cumul_gamma"] <= 0.595)].values[0],3)
        median_circ = round(df_stats["bins"].loc[(df_stats["cumul_circ"] >= 0.475) & (df_stats["cumul_circ"] <= 0.535)].values[0],3)

        # Upper Lower Percentages
        upper_68_gamma = df_stats["bins"].loc[(df_stats["cumul_gamma"] >= 0.8165) & (df_stats["cumul_gamma"] <= 0.8415)].values[0]
        lower_68_gamma = df_stats["bins"].loc[(df_stats["cumul_gamma"] >= 0.1495) & (df_stats["cumul_gamma"] <= 0.1715)].values[0]
        upper_95_gamma = df_stats["bins"].loc[(df_stats["cumul_gamma"] >= 0.9535) & (df_stats["cumul_gamma"] <= 0.9865)].values[0]
        lower_95_gamma = df_stats["bins"].loc[(df_stats["cumul_gamma"] >= 0.0205) & (df_stats["cumul_gamma"] <= 0.0465)].values[0] 
        
        upper_68 = df_stats["bins"].loc[(df_stats["cumul_norm"] >= 0.8165) & (df_stats["cumul_norm"] <= 0.8415)].values[0]
        lower_68 = df_stats["bins"].loc[(df_stats["cumul_norm"] >= 0.1495) & (df_stats["cumul_norm"] <= 0.1715)].values[0]
        upper_95 = df_stats["bins"].loc[(df_stats["cumul_norm"] >= 0.9535) & (df_stats["cumul_norm"] <= 0.9865)].values[0]
        lower_95 = df_stats["bins"].loc[(df_stats["cumul_norm"] >= 0.0205) & (df_stats["cumul_norm"] <= 0.0265)].values[0]
        
        upper_68_circ = df_stats["bins"].loc[(df_stats["cumul_circ"] >= 0.8265) & (df_stats["cumul_circ"] <= 0.8515)].values[0]
        lower_68_circ = df_stats["bins"].loc[(df_stats["cumul_circ"] >= 0.1445) & (df_stats["cumul_circ"] <= 0.3215)].values[0]
        upper_95_circ = df_stats["bins"].loc[(df_stats["cumul_circ"] >= 0.9535) & (df_stats["cumul_circ"] <= 0.9865)].values[0]
        lower_95_circ = df_stats["bins"].loc[(df_stats["cumul_circ"] >= 0.0155) & (df_stats["cumul_circ"] <= 0.1000)].values[0]

        np.set_printoptions(legacy = "1.25")

        # Roudn up percents
        percent_68 = round(lower_68,3), round(upper_68,3)
        percent_95 = round(lower_95,3), round(upper_95,3)
        
        percent_68_gamma = round(lower_68_gamma,3), round(upper_68_gamma,3)
        percent_95_gamma = round(lower_95_gamma,3), round(upper_95_gamma,3)
        
        percent_68_circ = round(lower_68_circ,3), round(upper_68_circ,3)
        percent_95_circ = round(lower_95_circ,3), round(upper_95_circ,3)
        
        # IMPORT STATS TO HELP WITH 68% 95% VALUES
        statistics = [median, percent_68, percent_95]
        print(f"Stats for  Uniform distribution and alpha = {alpha_step}: ")
        print(' Median: ', median, " 68% Intervals: ", percent_68, " 95% Intervals: ", percent_95)
        
        print(f"Stats for  Gamma distribution and alpha = {alpha_step}: ")
        print(' Median: ', median_gamma, " 68% Intervals: ", percent_68_gamma, " 95% Intervals: ", percent_95_gamma)
        
        print(f"Stats for  Circular distribution and alpha = {alpha_step}: ")
        print(' Median: ', median_circ, " 68% Intervals: ", percent_68_circ, " 95% Intervals: ", percent_95_circ)
        
        # rect = dict(boxstyle = "round", alpha = 1, facecolor = "white")
        # textstr = "\n".join((f'Median: {median}', f'68% Intervals: {percent_68}', f'95% Intervals: {percent_95}'))
        # ax.text(0.65, 0.75, textstr, transform = ax.transAxes, fontsize = 10, verticalalignment = "top", bbox = rect)
        plt.tight_layout()
        try:
            if test:
                plt.savefig(f'/College_Projects/Microlensing Separation/Figures/CDF_{self.numestep}_{which}_alpha_{alpha_step}_test.png')
            else:
                plt.savefig(f'/College_Projects/Microlensing Separation/Figures/CDF_{self.numestep}_{which}_alpha_{alpha_step}.png')
        except OSError:
            if test:
                plt.savefig(f'C:/Users/victo/College_Projects/Microlensing Separation/Figures/CDF_{self.numestep}_{which}_alpha_{alpha_step}_test.png')
            else:
                plt.savefig(f'C:/Users/victo/College_Projects/Microlensing Separation/Figures/CDF_{self.numestep}_{which}_alpha_{alpha_step}.png')       
        return statistics

    def stepalpha(self,which, alpha = 1, circ = False):
        """
        Combines the normal and circular dist.'s .csv files to 
        create multiple alpha files for comparison to testalpha .csv file.
        """
        # NEW METHOD TO OPEN FILES?
        # First detects if file is exists, if it doesn't it 
        # will then try another file location
        if not circ:
            try:    
                filename = f'/College_Projects/Microlensing Separation/Results/UnityHist_eccent_incline_{self.numestep}_0002_{which}.csv'
                pd.read_csv(filename)
            except FileNotFoundError:
                filename = f'/Users/victo/College_Projects/Microlensing Separation/Results/UnityHist_eccent_incline_{self.numestep}_0002_{which}.csv'
            # Once it detects one, it will check if the circular file exists
            # if it does, then it will flag it for combination and file creation
            # NOTE: CHECKS USING wnum INSTEAD OF numestep
            try:
                try:
                    filename2 = f'/College_Projects/Microlensing Separation/Results/UnityHist_eccent_incline_{self.wnum}_0002_circular_{which}.csv'
                    pd.read_csv(filename2)
                except FileNotFoundError:
                    filename2 = f'/Users/victo/College_Projects/Microlensing Separation/Results/UnityHist_eccent_incline_{self.wnum}_0002_circular_{which}.csv'
                files_separated = True
            except FileNotFoundError:
                print("No separate circular file found, following default path")
                files_separated = False
        else:
            try:
                files_separated = True
                filename2 = f'/College_Projects/Microlensing Separation/Results/UnityHist_eccent_incline_{self.wnum}_0002_circular_{which}.csv'
                pd.read_csv(filename2)
            except FileNotFoundError:
                filename2 = f'/Users/victo/College_Projects/Microlensing Separation/Results/UnityHist_eccent_incline_{self.wnum}_0002_circular_{which}.csv'
        
        if not files_separated:
            df = pd.read_csv(filename)
            df_new = df.copy()
            if  alpha == 0: # FOR REAL LINEAR, alpha = 0, FOR REAL LOG, alpha = -1
                mult_bins = [1 for num in range(0,199)]
            elif alpha == -1:
                mult_bins = [1 for num in range(0,199)]
            else:
                nbin = 200
                amin = 0.5
                amax = 21

                bins = np.geomspace(amin,amax, nbin)
                mult_bins = bins
                # THIS MIGHT BE DIFFERENT FROM WHAT IT IS SUPPOSED TO BE
                df["bins"] = bins[:-1]
            
            # print(mult_bins)
            for i in range(len(df_new["final list"])):
                if alpha < -1: # For (alpha -2) 
                    df_new.loc[i, "final list"] = round(df_new.loc[i, "final list"] * mult_bins[i]**(-2))
                    df_new.loc[i, "gamma list"] = round(df_new.loc[i, "gamma list"] * mult_bins[i]**(-2))
                    df_new.loc[i, "circular list"] = round(df_new.loc[i, "circular list"] * mult_bins[i]**(-2))
                elif alpha == 1: # For (alpha 1) values
                    df_new.loc[i, "final list"] = round(df_new.loc[i, "final list"] * mult_bins[i])
                    df_new.loc[i, "gamma list"] = round(df_new.loc[i, "gamma list"] * mult_bins[i])
                    df_new.loc[i, "circular list"] = round(df_new.loc[i, "circular list"] * mult_bins[i])
                elif alpha > 1: # For (alpha 2) values
                    df_new.loc[i, "final list"] = round(df_new.loc[i, "final list"] * mult_bins[i]**2)
                    df_new.loc[i, "gamma list"] = round(df_new.loc[i, "gamma list"] * mult_bins[i]**2)
                    df_new.loc[i, "circular list"] = round(df_new.loc[i, "circular list"] * mult_bins[i]**2)
                elif alpha == 0 or alpha == -1: # Linear (0) and Log (-1)
                    df_new.loc[i, "final list"] = round(df_new.loc[i, "final list"] * mult_bins[i])
                    df_new.loc[i, "gamma list"] = round(df_new.loc[i, "gamma list"] * mult_bins[i])
                    df_new.loc[i, "circular list"] = round(df_new.loc[i, "circular list"] * mult_bins[i])
        else:
            if not circ:
                df = pd.read_csv(filename)
                df2 = pd.read_csv(filename2)
                df_new = df.copy()
                df2_new = df2.copy()
                df_comb = pd.concat([df,df2], axis = 1, names = ["final list, circular list"])
            else:
                df = pd.read_csv(filename2)
                df_comb = df.copy()
                
            nbin = 200
            amin = 0.5
            amax = 21

            bins = np.geomspace(amin,amax, nbin)
            mult_bins = bins
            
            # THIS MIGHT BE DIFFERENT FROM WHAT IT IS SUPPOSED TO BE
            df["bins"] = bins[:-1]
            if not circ:
                for i in range(len(df_comb["final list"])):
                    if which == "Linear":
                        if alpha < -1: # For (alpha -2) 
                            df_comb.loc[i, "final list"] = round(df_comb.loc[i, "final list"] * mult_bins[i]**(-2))
                            df_comb.loc[i, "circular list"] = round(df_comb.loc[i, "circular list"] * mult_bins[i]**(-2))
                            df_comb.loc[i, "gamma list"] = round(df_comb.loc[i, "gamma list"] * mult_bins[i]**(-2))
                        elif alpha == -1: # For (alpha -1)
                            df_comb.loc[i, "final list"] = round(df_comb.loc[i, "final list"] * mult_bins[i] ** (-1))
                            df_comb.loc[i, "circular list"] = round(df_comb.loc[i, "circular list"] * mult_bins[i] ** (-1))
                            df_comb.loc[i, "gamma list"] = round(df_comb.loc[i, "gamma list"] * mult_bins[i]**(-1))
                        elif alpha == 0: # CENTER FOR LINEAR
                            df_comb.loc[i, "final list"] = round(df_comb.loc[i, "final list"])
                            df_comb.loc[i, "circular list"] = round(df_comb.loc[i, "circular list"])
                            df_comb.loc[i, "gamma list"] = round(df_comb.loc[i, "gamma list"])
                        elif alpha == 1: # For (alpha 1) 
                            df_comb.loc[i, "final list"] = round(df_comb.loc[i, "final list"] * mult_bins[i] ** (1))
                            df_comb.loc[i, "circular list"] = round(df_comb.loc[i, "circular list"] * mult_bins[i] ** (1))
                            df_comb.loc[i, "gamma list"] = round(df_comb.loc[i, "gamma list"] * mult_bins[i] ** (1))
                        elif alpha > 1: # For (alpha 2) 
                            df_comb.loc[i, "final list"] = round(df_comb.loc[i, "final list"] * mult_bins[i]** (2))
                            df_comb.loc[i, "circular list"] = round(df_comb.loc[i, "circular list"] * mult_bins[i]**(2))
                            df_comb.loc[i, "gamma list"] = round(df_comb.loc[i, "gamma list"] * mult_bins[i] ** (2))                    
                    elif which == "Log":
                        if alpha < -1: # For (alpha -2) 
                            df_comb.loc[i, "final list"] = round(df_comb.loc[i, "final list"] * mult_bins[i] ** (-1))
                            df_comb.loc[i, "circular list"] = round(df_comb.loc[i, "circular list"] * mult_bins[i] ** (-1))
                            df_comb.loc[i, "gamma list"] = round(df_comb.loc[i, "gamma list"] * mult_bins[i] ** (-1))
                        elif alpha == -1: # CENTER FOR LOG
                            df_comb.loc[i, "final list"] = round(df_comb.loc[i, "final list"])
                            df_comb.loc[i, "circular list"] = round(df_comb.loc[i, "circular list"])
                            df_comb.loc[i, "gamma list"] = round(df_comb.loc[i, "gamma list"])
                        elif alpha == 0: # For (alpha 0) 
                            df_comb.loc[i, "final list"] = round(df_comb.loc[i, "final list"] * mult_bins[i] ** (1))
                            df_comb.loc[i, "circular list"] = round(df_comb.loc[i, "circular list"] * mult_bins[i] ** (1))
                            df_comb.loc[i, "gamma list"] = round(df_comb.loc[i, "gamma list"] * mult_bins[i] ** (1))
                        elif alpha == 1: # For (alpha 1) 
                            df_comb.loc[i, "final list"] = round(df_comb.loc[i, "final list"] * mult_bins[i] ** (2))
                            df_comb.loc[i, "circular list"] = round(df_comb.loc[i, "circular list"] * mult_bins[i] ** (2))
                            df_comb.loc[i, "gamma list"] = round(df_comb.loc[i, "gamma list"] * mult_bins[i] ** (2))
                        elif alpha > 1: # For (alpha 2) 
                            df_comb.loc[i, "final list"] = round(df_comb.loc[i, "final list"] * mult_bins[i] ** (3))
                            df_comb.loc[i, "circular list"] = round(df_comb.loc[i, "circular list"] * mult_bins[i] ** (3))
                            df_comb.loc[i, "gamma list"] = round(df_comb.loc[i, "gamma list"] * mult_bins[i] ** (3))
                    elif which == "Power":
                        if alpha < -1: # For (alpha -2) 
                            df_comb.loc[i, "final list"] = round(df_comb.loc[i, "final list"] * mult_bins[i]**(-3))
                            df_comb.loc[i, "circular list"] = round(df_comb.loc[i, "circular list"] * mult_bins[i]**(-3))
                            df_comb.loc[i, "gamma list"] = round(df_comb.loc[i, "gamma list"] * mult_bins[i] ** (-3))
                        elif alpha == -1: # For (alpha -1)
                            df_comb.loc[i, "final list"] = round(df_comb.loc[i, "final list"] * mult_bins[i] ** (-2))
                            df_comb.loc[i, "circular list"] = round(df_comb.loc[i, "circular list"] * mult_bins[i] ** (-2))
                            df_comb.loc[i, "gamma list"] = round(df_comb.loc[i, "gamma list"] * mult_bins[i] ** (-2))
                        elif alpha == 0: # For (alpha 0) 
                            df_comb.loc[i, "final list"] = round(df_comb.loc[i, "final list"] * mult_bins[i]** (-1))
                            df_comb.loc[i, "circular list"] = round(df_comb.loc[i, "circular list"] * mult_bins[i]** (-1))
                            df_comb.loc[i, "gamma list"] = round(df_comb.loc[i, "gamma list"] * mult_bins[i] ** (-1))
                        elif alpha == 1: # CENTER FOR POWER
                            df_comb.loc[i, "final list"] = round(df_comb.loc[i, "final list"])
                            df_comb.loc[i, "circular list"] = round(df_comb.loc[i, "circular list"])
                            df_comb.loc[i, "gamma list"] = round(df_comb.loc[i, "gamma list"])
                        elif alpha > 1: # For (alpha 2) 
                            df_comb.loc[i, "final list"] = round(df_comb.loc[i, "final list"] * mult_bins[i] ** (1))
                            df_comb.loc[i, "circular list"] = round(df_comb.loc[i, "circular list"] * mult_bins[i] ** (1))
                            df_comb.loc[i, "gamma list"] = round(df_comb.loc[i, "gamma list"] * mult_bins[i] ** (1))
            else:
                for i in range(len(df_comb["circular list"])):
                    if which == "Linear":
                            if alpha < -1: # For (alpha -2) 
                                df_comb.loc[i, "circular list"] = round(df_comb.loc[i, "circular list"] * mult_bins[i] ** (-2))
                            elif alpha == -1: # For (alpha -1)
                                df_comb.loc[i, "circular list"] = round(df_comb.loc[i, "circular list"] * mult_bins[i] ** (-1))
                            elif alpha == 0: # CENTER FOR LINEAR
                                df_comb.loc[i, "circular list"] = round(df_comb.loc[i, "circular list"])
                            elif alpha == 1: # For (alpha 1) 
                                df_comb.loc[i, "circular list"] = round(df_comb.loc[i, "circular list"] * mult_bins[i] ** (1))
                            elif alpha > 1: # For (alpha 2) 
                                df_comb.loc[i, "circular list"] = round(df_comb.loc[i, "circular list"] * mult_bins[i] ** (2))
                    elif which == "Log":
                            if alpha < -1: # For (alpha -2) 
                                df_comb.loc[i, "circular list"] = round(df_comb.loc[i, "circular list"] * mult_bins[i] ** (-1))
                            elif alpha == -1: # CENTER FOR LOG
                                df_comb.loc[i, "circular list"] = round(df_comb.loc[i, "circular list"])
                            elif alpha == 0: # For (alpha 0) 
                                df_comb.loc[i, "circular list"] = round(df_comb.loc[i, "circular list"] * mult_bins[i])
                            elif alpha == 1: # For (alpha 1) 
                                df_comb.loc[i, "circular list"] = round(df_comb.loc[i, "circular list"] * mult_bins[i] ** (2))
                            elif alpha > 1: # For (alpha 2) 
                                df_comb.loc[i, "circular list"] = round(df_comb.loc[i, "circular list"] * mult_bins[i] ** (3))
                    elif which == "Power":
                        if alpha < -1: # For (alpha -2) 
                            df_comb.loc[i, "circular list"] = round(df_comb.loc[i, "circular list"] * mult_bins[i]**(-3))
                        elif alpha == -1: # For (alpha -1)
                            df_comb.loc[i, "circular list"] = round(df_comb.loc[i, "circular list"] * mult_bins[i] ** (-2))
                        elif alpha == 0: # For (alpha 0) 
                            df_comb.loc[i, "circular list"] = round(df_comb.loc[i, "circular list"] * mult_bins[i]** (-1))
                        elif alpha == 1: # CENTER FOR POWER
                            df_comb.loc[i, "circular list"] = round(df_comb.loc[i, "circular list"])
                        elif alpha > 1: # For (alpha 2) 
                            df_comb.loc[i, "circular list"] = round(df_comb.loc[i, "circular list"] * mult_bins[i]**(1))
                                
        # Creates .csv file for comparing to testalpha main file
        if not circ:
            try:
                file_name_new = f'/College_Projects/Microlensing Separation/Results/UnityHist_eccent_incline_{self.numestep}_0002_{which}_alpha_{alpha}.csv'
                df_comb.to_csv(file_name_new, index = False)
            except OSError:
                file_name_new = f'/Users/victo/College_Projects/Microlensing Separation/Results/UnityHist_eccent_incline_{self.numestep}_0002_{which}_alpha_{alpha}.csv'
                df_comb.to_csv(file_name_new, index = False)
        else:
            try:
                file_name_circ = f'/College_Projects/Microlensing Separation/Results/UnityHist_eccent_incline_{self.wnum}_{which}_alpha_{alpha}_circ.csv'
                df_comb.to_csv(file_name_circ, index = False)
            except OSError:
                file_name_circ = f'/Users/victo/College_Projects/Microlensing Separation/Results/UnityHist_eccent_incline_{self.wnum}_{which}_alpha_{alpha}_circ.csv'
                df_comb.to_csv(file_name_circ, index = False)
        
        print(f"File with alpha = {alpha} saved successfully")
        return df_comb
    
    def testalpha(self, which, circ = False):
        """
        ONLY WORKS WITH ALPHA = 0 and ALPHA = -1 (Soon to be Alpha = 1 as well)
        
        Creates a foundation file which is used in UnityPlotHistLoad
        function for comparing all of its test alpha values from stepalpha.
        """
        nbin = 200
        amin = 0.5
        amax = 21

        if which == "Linear":
            alpha = 0
        elif which == "Log":
            alpha = -1
        elif which == "Power":
            alpha = 1
        else:
            return(print("Cannot run, must be Linear, Log, or Power distribution."))
        
        
        bins = np.geomspace(amin,amax, nbin)
        mult_bins = bins
        if circ:
            print(f"Circ Flag Activated")
            try: 
                filename = f'/College_Projects/Microlensing Separation/Results/UnityHist_eccent_incline_{self.wnum}_{which}_alpha_{alpha}_circ.csv'
                pd.read_csv(filename)
            except FileNotFoundError:
                filename = f'/Users/victo/College_Projects/Microlensing Separation/Results/UnityHist_eccent_incline_{self.wnum}_{which}_alpha_{alpha}_circ.csv'
        else:
            try:
                filename = f'/College_Projects/Microlensing Separation/Results/UnityHist_eccent_incline_{self.numestep}_0002_{which}_alpha_{alpha}.csv'
                pd.read_csv(filename)
            except FileNotFoundError:
                filename = f'/Users/victo/College_Projects/Microlensing Separation/Results/UnityHist_eccent_incline_{self.numestep}_0002_{which}_alpha_{alpha}.csv'
            

        df = pd.read_csv(filename)
        df_new = pd.DataFrame()
        # print(mult_bins)

        df["bins"] = bins[:-1]
        if circ:
            df_new["alpha 1 circ"] = df["circular list"].copy()
            df_new["alpha 0 circ"] = df["circular list"].copy()
            df_new["alpha -1 circ"] = df["circular list"].copy()
            
            if alpha == 0 or alpha == -1 or alpha == 1:
                for i in range(len(df_new["alpha 0 circ"])):
                    if alpha == 0:
                        df_new.loc[i, "alpha 1 circ"] = round(df_new.loc[i, "alpha 0 circ"] * mult_bins[i]**(1))
                        # CENTER FOR alpha = 0 (So just the same as the normal)
                        df_new.loc[i, "alpha 0 circ"] = round(df_new.loc[i, "alpha 0 circ"])
                        df_new.loc[i, "alpha -1 circ"] = round(df_new.loc[i, "alpha 0 circ"] * mult_bins[i]**(-1))
                    elif alpha == -1:
                        df_new.loc[i, "alpha 1 circ"] = round(df_new.loc[i, "alpha -1 circ"] * mult_bins[i]**(2))
                        df_new.loc[i, "alpha 0 circ"] = round(df_new.loc[i, "alpha -1 circ"]* mult_bins[i]**(1))
                        # CENTER FOR alpha = -1 (So just the same as the normal)
                        df_new.loc[i, "alpha -1 circ"] = round(df_new.loc[i, "alpha -1 circ"])
                    elif alpha == 1:
                        # CENTER FOR alpha = 1 (So just the same as the normal)
                        df_new.loc[i, "alpha 1 circ"] = round(df_new.loc[i, "alpha 1 circ"])
                        df_new.loc[i, "alpha 0 circ"] = round(df_new.loc[i, "alpha 1 circ"]* mult_bins[i]**(-1))
                        df_new.loc[i, "alpha -1 circ"] = round(df_new.loc[i, "alpha 1 circ"] * mult_bins[i]**(-2))
            else:
                return(print(f"Warning: {alpha} is not a valid integer. Please use (1), (0), or (-1) as your options for alpha"))
        else:
            df_new["alpha 2"] = df["final list"].copy()
            df_new["alpha 2 gamma"] = df["gamma list"].copy()
            df_new["alpha 2 circ"] = df["circular list"].copy()
            df_new["alpha 1"] = df["final list"].copy()
            df_new["alpha 1 gamma"] = df["gamma list"].copy()
            df_new["alpha 1 circ"] = df["circular list"].copy()
            df_new["alpha 0"] = df["final list"].copy()
            df_new["alpha 0 gamma"] = df["gamma list"].copy()
            df_new["alpha 0 circ"] = df["circular list"].copy()
            df_new["alpha -1"] = df["final list"].copy()
            df_new["alpha -1 gamma"] = df["gamma list"].copy()
            df_new["alpha -1 circ"] = df["circular list"].copy()
            df_new["alpha -2"] = df["final list"].copy()
            df_new["alpha -2 gamma"] = df["gamma list"].copy()
            df_new["alpha -2 circ"] = df["circular list"].copy()

            if alpha == 0 or alpha == -1 or alpha == 1:
                for i in range(len(df_new["alpha 0"])):
                    if alpha == 0:
                        df_new.loc[i, "alpha 2"] = round(df_new.loc[i, "alpha 0"] * mult_bins[i]**(2))
                        df_new.loc[i, "alpha 2 gamma"] = round(df_new.loc[i, "alpha 0 gamma"] * mult_bins[i]**(2))
                        df_new.loc[i, "alpha 2 circ"] = round(df_new.loc[i, "alpha 0 circ"] * mult_bins[i]**(2))
                        df_new.loc[i, "alpha 1"] = round(df_new.loc[i, "alpha 0"] * mult_bins[i]**(1))
                        df_new.loc[i, "alpha 1 gamma"] = round(df_new.loc[i, "alpha 0 gamma"] * mult_bins[i]**(1))
                        df_new.loc[i, "alpha 1 circ"] = round(df_new.loc[i, "alpha 0 circ"] * mult_bins[i]**(1))
                        # CENTER FOR alpha = 0 (So just the same as the normal)
                        df_new.loc[i, "alpha 0"] = round(df_new.loc[i, "alpha 0"])
                        df_new.loc[i, "alpha 0 gamma"] = round(df_new.loc[i, "alpha 0 gamma"])
                        df_new.loc[i, "alpha 0 circ"] = round(df_new.loc[i, "alpha 0 circ"])
                        df_new.loc[i, "alpha -1"] = round(df_new.loc[i, "alpha 0"] * mult_bins[i]**(-1))
                        df_new.loc[i, "alpha -1 gamma"] = round(df_new.loc[i, "alpha 0 gamma"] * mult_bins[i]**(-1))
                        df_new.loc[i, "alpha -1 circ"] = round(df_new.loc[i, "alpha 0 circ"] * mult_bins[i]**(-1))
                        df_new.loc[i, "alpha -2"] = round(df_new.loc[i, "alpha 0"] * mult_bins[i]**(-2))
                        df_new.loc[i, "alpha -2 gamma"] = round(df_new.loc[i, "alpha 0 gamma"] * mult_bins[i]**(-2))
                        df_new.loc[i, "alpha -2 circ"] = round(df_new.loc[i, "alpha 0 circ"] * mult_bins[i]**(-2))
                    
                    elif alpha == -1:
                        df_new.loc[i, "alpha 2"] = round(df_new.loc[i, "alpha -1"] * mult_bins[i]**(3))
                        df_new.loc[i, "alpha 2 gamma"] = round(df_new.loc[i, "alpha -1 gamma"] * mult_bins[i]**(3))
                        df_new.loc[i, "alpha 2 circ"] = round(df_new.loc[i, "alpha -1 circ"] * mult_bins[i]**(3))
                        df_new.loc[i, "alpha 1"] = round(df_new.loc[i, "alpha -1"] * mult_bins[i]**(2))
                        df_new.loc[i, "alpha 1 gamma"] = round(df_new.loc[i, "alpha -1 gamma"] * mult_bins[i]**(2))
                        df_new.loc[i, "alpha 1 circ"] = round(df_new.loc[i, "alpha -1 circ"] * mult_bins[i]**(2))
                        df_new.loc[i, "alpha 0"] = round(df_new.loc[i, "alpha -1"] * mult_bins[i]**(1))
                        df_new.loc[i, "alpha 0 gamma"] = round(df_new.loc[i, "alpha -1 gamma"] * mult_bins[i]**(1))
                        df_new.loc[i, "alpha 0 circ"] = round(df_new.loc[i, "alpha -1 circ"] * mult_bins[i]**(1))
                        # CENTER FOR alpha = -1 (So just the same as the normal)
                        df_new.loc[i, "alpha -1"] = round(df_new.loc[i, "alpha -1"])
                        df_new.loc[i, "alpha -1 gamma"] = round(df_new.loc[i, "alpha -1 gamma"])
                        df_new.loc[i, "alpha -1 circ"] = round(df_new.loc[i, "alpha -1 circ"])
                        df_new.loc[i, "alpha -2"] = round(df_new.loc[i, "alpha -1"] * mult_bins[i]**(-1))
                        df_new.loc[i, "alpha -2 gamma"] = round(df_new.loc[i, "alpha -1 gamma"] * mult_bins[i]**(-1))
                        df_new.loc[i, "alpha -2 circ"] = round(df_new.loc[i, "alpha -1 circ"] * mult_bins[i]**(-1))
                        
                    elif alpha == 1:
                        df_new.loc[i, "alpha 2"] = round(df_new.loc[i, "alpha 1"] * mult_bins[i]**(1))
                        df_new.loc[i, "alpha 2 gamma"] = round(df_new.loc[i, "alpha 1 gamma"] * mult_bins[i]**(1))
                        df_new.loc[i, "alpha 2 circ"] = round(df_new.loc[i, "alpha 1 circ"] * mult_bins[i]**(1))
                        # CENTER FOR alpha = 1
                        df_new.loc[i, "alpha 1"] = round(df_new.loc[i, "alpha 1"])
                        df_new.loc[i, "alpha 1 gamma"] = round(df_new.loc[i, "alpha 1 gamma"])
                        df_new.loc[i, "alpha 1 circ"] = round(df_new.loc[i, "alpha 1 circ"])
                        df_new.loc[i, "alpha 0"] = round(df_new.loc[i, "alpha 1"] * mult_bins[i]**(-1))
                        df_new.loc[i, "alpha 0 gamma"] = round(df_new.loc[i, "alpha 1 gamma"] * mult_bins[i]**(-1))
                        df_new.loc[i, "alpha 0 circ"] = round(df_new.loc[i, "alpha 1 circ"] * mult_bins[i]**(-1))
                        df_new.loc[i, "alpha -1"] = round(df_new.loc[i, "alpha 1"] * mult_bins[i]**(-2))
                        df_new.loc[i, "alpha -1 gamma"] = round(df_new.loc[i, "alpha 1 gamma"] * mult_bins[i]**(-2))
                        df_new.loc[i, "alpha -1 circ"] = round(df_new.loc[i, "alpha 1 circ"] * mult_bins[i]**(-2))
                        df_new.loc[i, "alpha -2"] = round(df_new.loc[i, "alpha 1"] * mult_bins[i]**(-3))
                        df_new.loc[i, "alpha -2 gamma"] = round(df_new.loc[i, "alpha 1 gamma"] * mult_bins[i]**(-3))
                        df_new.loc[i, "alpha -2 circ"] = round(df_new.loc[i, "alpha 1 circ"] * mult_bins[i]**(-3))
            else:
                return(print(f"Warning: {alpha} is not a valid integer. Please use (1), (0), or (-1) as your options for alpha"))
        if circ:
            try:
                file_name_new = f'/College_Projects/Microlensing Separation/Results/UnityHist_{self.wnum}_alpha_{alpha}_test_circ.csv'
                df_new.to_csv(file_name_new, index = False)
            except OSError:
                file_name_new = f'/Users/victo/College_Projects/Microlensing Separation/Results/UnityHist_{self.wnum}_alpha_{alpha}_test_circ.csv'
                df_new.to_csv(file_name_new, index = False)
        else:
            try:
                file_name_new = f'/College_Projects/Microlensing Separation/Results/UnityHist_{self.numestep}_alpha_{alpha}_test.csv'
                df_new.to_csv(file_name_new, index = False)
            except OSError:
                file_name_new = f'/Users/victo/College_Projects/Microlensing Separation/Results/UnityHist_{self.numestep}_alpha_{alpha}_test.csv'
                df_new.to_csv(file_name_new, index = False)    
        print(f"Test File with {which} distribution saved successfully")
        
        return df_new

if __name__ == "__main__":
    numestep = 100
    numdiv = 4 
    wnum = 100 # THIS DETERMINES HOW MANY POSITIONS IN THE ARRAY THERE ARE
    inum = wnum
    # FOR REAL LINEAR, alpha = 0, FOR REAL LOG, alpha = -1, FOR REAL POWER, alpha = 1
    which = "Linear"
    alpha = -1 # For test = True, this becomes the comparison to which
    inclination = True # KEEP IN MIND THIS VALUE
    circ = False
    gamma_bool = False
    test = True
    unity = False
    dist = "uniform"
    # specify = []
    specify = [0.5, np.pi/3]
    tothist = Sep_plot(numestep=numestep, numdiv=numdiv, wnum = wnum)
    # rlist = tothist.MultiPlotProj(w = 0, start = 0.5, end = 20, step = 0.5, specify = specify)
    # specify = [eccentricity, inclination]
    # rtemp = tothist.MultiPlotHist(w = 0, step = 0.002, end = 20, which = which , specify = specify) # Only works for Log (has all 3)
    
   # CompleteHistLoad includes Omega and Inclination marginalization!
    # tothist.CompleteHistGen(which = which, unity = unity)
    # tothist.CompleteHistLoad(which = which, inclination = inclination)
    
    # UnityPlotHistGen includes Inclination and Eccentricity Marginalization!
    # folder = tothist.UnityPlotHistGen(which = which, unity = unity, circ = circ, gamma_bool = gamma_bool, inclination = inclination)
    load = tothist.UnityPlotHistLoad(which = which, alpha_step = alpha, dist = dist, circ = circ, test = test)
    # cdf = tothist.statistics(which = which, alpha_step = alpha, test = test)
    
    # Note: stepalpha function can also combine uniform and circular distributions!
    # alpha = tothist.stepalpha(which = which, alpha = alpha, circ = circ)
    # test = tothist.testalpha(which = which, circ = circ)



