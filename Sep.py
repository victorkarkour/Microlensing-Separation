import astropy.constants as ac
import numpy as np
import scipy.signal as signal
import scipy.optimize as sc
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
from multiprocessing import Pool
import gc
import math
from collections import Counter
from scipy.stats import gamma

def NewtRaf(M, e, maxiter=50, tol=1e-8):
    """
    Custom vectorized Newton-Raphson implementation for Kepler's equation.
    More memory efficient than scipy's newton.
    
    Parameters
    ----------
    M : array-like
        Mean anomaly
    e : float
        Eccentricity
    maxiter : int, optional
        Maximum number of iterations
    tol : float, optional
        Convergence tolerance
        
    Returns
    -------
    array-like
        Eccentric anomaly
    """
    # Old Newton-Raphson
    initial = M + e * np.sin(M)
    Solution = np.zeros_like(M)
    
    Solution = sc.newton(Kepler, initial, fprime=DKepler, args=(M,e))
    return Solution
    
    # Initial guess
    # E = M + e * np.sin(M)
    # Vectorized Newton-Raphson iteration
    # for _ in range(maxiter):
    #     # Calculate function value and derivative
    #     f = Kepler(E,M,e)
    #     df = DKepler(E,M,e)
        
    #     # Update step
    #     dE = f / df
    #     E = E - dE
        
    #     # Check convergence
    #     if np.all(np.abs(dE) <= tol):
    #         break
            
    # return E

def Kepler(En,Mn,ec):
    """
    Returns the Kepler Transcendental Equation.
    
    -------------------------------
    ### Parameters
    
    En : arraylike <br>
        
        Eccentric Anomaly
        
    Mn : arraylike <br>
        
        Combination of Period and semimajor axis
        
    ec : float <br>
        
        Eccentricity
        
    -------------------------------
    ### Returns
    
        Kepler Equation
    """
    # print(En)
    return(En-ec*np.sin(En)-Mn)

def DKepler(En,Mn,ec):
    """
    Returns the Derivative of the Kepler Equation.
    
    -------------------------------
    ### Parameters
    
    En : arraylike <br>
        
        Eccentric Anomaly
        
    Mn : arraylike <br>
        
        Combination of Period and semimajor axis
        
    ec : float <br>
        
        Eccentricity
        
    -------------------------------
    ### Returns
    
        Derivative of Kepler Equation
    """
    return(1-ec*np.cos(En))

def OrbGeo(t0=0, a=1, w = 0, W = 0, i = 0, e = 0):
    """
    Creates and calculates the X and Y axis of a planet's orbit.
    
    --------
    ### Parameters
    
    t0 : float <br>
        time of periapsis (closest to star)
        
    a : float <br>
        semimajor axis 
    
    w : float (in radians) <br>
        argument of periapsis
        
    W : float (in radians) <br>
        longitude of ascending node
    
    i : float (in radians) <br>
        inclination angle
        
    e : float <br>
        eccentricity
        
    -------
    ### Returns 
    
    x : list <br>
        position of planet's orbit on the x-axis according to time
    
    y : list <br>
        position of planet's orbit on the y-axis according to time
    
    t : list <br>
        time associated with planet's position
    """
    xt = []
    yt = []
    # Assume time at periastron (closest point to star),
    # t0, as our starting time so t0 = 0
    # Check with Scott to see if we can do this
    # t0 = 0
    
    # Initial Equations
    A = a*(np.cos(W)*np.cos(w) - np.sin(W)*np.sin(w)*np.cos(i))
    B = a*(np.sin(W)*np.cos(w) + np.cos(W)*np.sin(w)*np.cos(i))
    F = a*(-np.cos(W)*np.sin(w) - np.sin(W)*np.cos(w)*np.cos(i))
    G = a*(-np.sin(W)*np.sin(w) + np.cos(W)*np.cos(w)*np.cos(i))
     
    # Equation for P (G*M_sun = 1 in Solar Units)
    P = np.sqrt((4*np.pi)/(1)*a**3)
    
    t = np.linspace(0, P, 4000)
    
    for k in t:
        # Function for E
        M = (2*np.pi/P) * (k-t0)
        
        # Solve Kepler Equation
        Et = NewtRaf(M, e)
    
        # Function of X and Y according to E
        Xt = np.cos(Et) - e
        Yt = np.sqrt(1-e**2)*np.sin(Et)
    
        # Actual x any y functions of t
        xt.append(A*Xt + F*Yt)
        yt.append(B*Xt + G*Yt)
    
    return(xt, yt, t)

def OrbGeoAlt(t0=0.0, a=1.0, w = 0.0, W = 0.0, i = 0.0, e = 0.0):
    """
    Creates and calculates the X and Y axis of a planet's orbit 
    using a different method for calculation.
    
    --------
    ### Parameters
    
    t0 : float <br>
        time of periapsis (closest to star)
        
    a : float <br>
        semimajor axis 
    
    w : float (in radians) <br>
        argument of periapsis
        
    W : float (in radians) <br>
        longitude of ascending node
    
    i : float (in radians) <br>
        inclination angle
        
    e : float <br>
        eccentricity
        
    -------
    ### Returns 
    
    x : list <br>
        position of planet's orbit on the x-axis according to time
    
    y : list <br>
        position of planet's orbit on the y-axis according to time
    
    t : list <br>
        time associated with planet's position
    """
    
    # Initial Equations
    A = a*(np.cos(W)*np.cos(w) - np.sin(W)*np.sin(w)*np.cos(i))
    B = a*(np.sin(W)*np.cos(w) + np.cos(W)*np.sin(w)*np.cos(i))
    F = a*(-np.cos(W)*np.sin(w) - np.sin(W)*np.cos(w)*np.cos(i))
    G = a*(-np.sin(W)*np.sin(w) + np.cos(W)*np.cos(w)*np.cos(i))
    
    # Create time function (start to whatever start was plus 2*pi)
    # nphase = int(800* a ** 1.5)
    
    phi = np.linspace(t0,(1)*(2.0*np.pi)+t0, 10000)
    
    # Place the time function into the Eccentric Anomaly
    # Solve Kepler Equation in chunks to manage memory
    # Et = NewtRaf(phi, e)
    
    # chunk_size = 10000
    # Et = np.zeros_like(phi)
    
    # for i in range(0, len(phi), chunk_size):
    #     end_idx = min(i + chunk_size, len(phi))
    #     Et[i:end_idx] = NewtRaf(phi[i:end_idx], e)
    
    Et = NewtRaf(phi,e)
    
    # Function of X and Y according to E
    Xt = np.cos(Et) - e
    Yt = np.sqrt(1-e**2)*np.sin(Et)
    
    # Actual x any y functions of t
    xt = A * Xt + F * Yt
    yt = B * Xt + G * Yt
    
    return (xt, yt, phi)

def Velocity(param):
    """
    Calculates the velocity of a planet's orbit.
    
    ------
    ### Parameters
    
    t : arraylike <br>
        time function of a planet's orbit
        
    param : list of floats <br>
        parameters used to create the Orbital Geometry of Planet's Orbit
        
    size : integer <br>
        how large the change of time will be 
        
    ------
    ### Returns
    
    vel : array of floats <br>
        velocity of a planet's orbit
    """
    # Change of t will be the same for all values
    # because of how np.linespace works (evenly spaced)
    # MIGHT HAVE TO CHANGE HOW LARGE THE CHANGE OF TIME
    # TO ALLOW FOR VELOCITY TO TAKE PLACE
    changet = (np.pi*2)/10000
    e, i, w, end, step, start = param
    vel = []
    # In case inclination angle is used, then it makes sure to include it
    
    stepthrough = np.arange(start, end + step, step)
    
    for val in stepthrough:
    
        # Orbital Calculations
        x1, y1, t1 = OrbGeoAlt(-changet, a = val, e = e, w = w, i = i)
        x2, y2, t2 = OrbGeoAlt(changet, a = val, e = e, w = w, i = i)
    
        # Change of degrees function
        # because of array arithmetic, we don't need to use for loops
        changedeg = np.sqrt(np.add(np.power(np.subtract(x2,x1),2),np.power(np.subtract(y2,y1),2)))
        
        # Velocity function
        vel.append(np.divide(changedeg,(2.0*changet)))
    # print("Last Velocity output: ", vel[-1])
    # print("Minimum Velocity: ", np.min(vel))
    return vel

def DotSize(vel, velmax, velmin):
    """
    Calculates the dot size of a planet's orbit using its velocity.
    
    ------
    ### Parameters
    
    vel : array of floats <br>
        velocity of a planet's orbit
        
    velmax : float <br>
        maximum velocity of ENTIRE PLOT
        
    velmin : float <br>
        minimum velocity of ENTIRE PLOT
        
    ------
    ### Returns
    
    ratio : array of floats <br>
        dot size proportional to velocity
    """
    # Makes an empty array of the length of the vel array
    ratio = np.empty(len(vel))
    
    # for i in range(len(vel)):
    #     # Similar to if statement in changedeg,
    #     # except for the ratio
    #     if vel[i] == 0:
    #         ratio.append(0.1)
    #     else:
    #         # ratio equation (deprecated)
    #         # ratio.append((np.abs(velmin)/np.abs(vel[i]))*2+0.1)
            
    #         # ratio equation (NEW)
    # num = np.subtract(np.log10(np.abs(vel)) , np.log10(np.abs(velmin)))
    # denom = np.subtract(np.log10(np.abs(velmax)) , np.log10(np.abs(velmin)))
    
    # Numerator eq
    num = np.subtract(np.abs(vel) , np.abs(velmin))
    # Denominator eq
    denom = np.abs(velmax) - np.abs(velmin) 
    # Combined eq
    ratio =  (40.0 - (np.divide(num,denom)) * 39.0)
    
    return ratio

def Rchange(param, coords = False, inclination = False):
    """
    Stores all the points in the data where the radius of the orbit
    is very close to the Einstein ring radius.
    
    --------
    ### Parameters
        
    x : arraylike <br>
        x position of a planet's orbit according to time
        
    y : arraylike <br>
        y position of a planet's orbit according to time
    
    a : float <br>
        semimajor axis of planet's orbit    
    ------
    ### Returns
    
    rlist : list of floats <br>
        values at which the radius was close to the ring radius

    """
    # rlistlog = [] 
    # rlistlin = []
    # By making this a dictionary, it vastly improves layout of counts of semimajor axis dots
    totlindict = {}
    totlogdict = {}
    totlinsemidict = {}
    totevaldict = {}
    # Coordinate Lists
    xlist = []
    ylist = []
    r0 = 1 # Einstein Ring Radius
    
    # Has different parameter sets if conditions are met
    if coords == True:
        e, i, w, end, step, start = param
        Linear = True
    else:
        e, i, w, end, step, Linear, inclination = param
        # if inclination:
        #     istep = i
    
    # Only steps through Linear portion of points
    if Linear == "Linear":
        # Linear Portion
        # stepthrough = np.arange(0.5, end + step, step)
        stepthrough = stepdata(0, 0.5, end, 10000)
        # Goes through each value of a in the stepthrough
        for aval in stepthrough:
            for ival in i:
                for eval in e:    
                    x, y, t = OrbGeoAlt(a = aval, e = eval, i = ival ,w = w)
                    r = np.sqrt(x**2+y**2)
                    # Whereever there is this value, it finds the indices of each point in the list
                    conlin = np.where(np.abs(r-r0)<=0.01)
        
                    if coords == False and inclination:
                        # Has brackets with 0 b/c conlin is an array of length 1, to get to values u must flatten
                        if aval in totlindict:
                            totlindict[aval] += len(conlin[0])
                        else:
                            totlindict[aval] = len(conlin[0])
                        if eval in totevaldict:
                            totevaldict[eval] += len(conlin[0])
                        else:
                            totevaldict[eval] = len(conlin[0])
                else: 
                    # Same thing as coords == False but has coord lists for Multiplot
                    totlindict[val] = len(conlin[0])
                    if coords:
                        xlist.append(np.where(np.abs(r-r0)<=0.01, x, None))
                        ylist.append(np.where(np.abs(r-r0)<=0.01, y, None))
        return totlindict, xlist, ylist, totlogdict, totevaldict
    elif Linear == "Log":
        if inclination == False:
            # Log Portion
            # steps = np.linspace(0, 10000,10000)
            # loga = np.log10(0.5) + steps/10000 * (np.log10(end)-np.log10(0.5))
            # stepthrough = 10**loga 
            stepthrough = stepdata(1, 0.5, end, 10000)
            for val in stepthrough:
                x, y, t = OrbGeoAlt(a = val, e = e, i = i ,w = w)
    
                r = np.sqrt(x**2+y**2)
        
                conlog = np.where(np.abs(r-r0)<=0.01)
                totlogdict[val] = len(conlog[0])
            gc.collect()
            # Linear Portion
            # stepthrough = np.arange(0.5, end + step, step)
            stepthrough = stepdata(0, 0.5, end, 10000)
            for val in stepthrough:
                x, y, t = OrbGeoAlt(a = val, e = e, i = i ,w = w)
    
                r = np.sqrt(x**2+y**2)
            
                conlin = np.where(np.abs(r-r0)<=0.01)
                totlindict[val] = len(conlin[0])
            # Power Law Portion
            stepthrough = stepdata(2, 0.5, end, 10000)
            for val in stepthrough:
                x, y, t = OrbGeoAlt(a = val, e = e, i = i ,w = w)
    
                r = np.sqrt(x**2+y**2)
            
                conlin = np.where(np.abs(r-r0)<=0.01)
                totlinsemidict[val] = round(len(conlin[0]))
            return totlindict, xlist, ylist, totlogdict, totlinsemidict
        else:
            if Linear == "Log":
                # Log Portion
                steps = np.linspace(0, 10000,10000)
                loga = np.log10(0.5) + steps/10000 * (np.log10(end)-np.log10(0.5))
                stepthrough = 10**loga
                for aval in stepthrough:
                    for ival in i:
                        for eval in e:
                            x, y, t = OrbGeoAlt(a = aval, e = eval, i = ival ,w = w)
                            r = np.sqrt(x**2+y**2)
                    
                            conlog = np.where(np.abs(r-r0)<=0.01)
                            if aval in totlogdict:
                                totlogdict[aval] += len(conlog[0])
                            else:
                                totlogdict[aval] = len(conlog[0])
                            if eval in totevaldict:
                                totevaldict[eval] += len(conlog[0])
                            else:
                                totevaldict[eval] = len(conlog[0])
            return totlindict, xlist, ylist, totlogdict, totevaldict
    elif Linear == "Linear / a":
        # Linear / a Portion
        stepthrough = stepdata(0.5, 0.5, end, 10000)
        for aval in stepthrough:
            for ival in i:
                for eval in e:
                    x, y, t = OrbGeoAlt(a = aval, e = eval, i = ival ,w = w)
                    r = np.sqrt(x**2+y**2)
                    # Whereever there is this value, it finds the indices of each point in the list
                    conlin = np.where(np.abs(r-r0)<=0.01)
                    # Has brackets with 0 b/c conlin is an array of length 1, to get to values u must flatten
                
                    # REMINDER: THIS IS FOR LINEAR / A, I JUST REMOVED THE totlinsemidict FROM THIS FOR EASIER INTERPRETATION
                    if aval in totlindict:
                        totlindict[aval] = round(len(conlin[0]) / aval)
                    else:
                        totlindict[aval] = round(len(conlin[0]) / aval)
                    if eval in totevaldict:
                        totevaldict[eval] += len(conlin[0])
                    else:
                        totevaldict[eval] = len(conlin[0])
        return totlindict, xlist, ylist, totlogdict, totevaldict

def stepdata(alpha, xmin, xmax, nsamples):
    """
    """
    step = np.linspace(0,1,nsamples+2)[1:-1]
    
    if alpha == 1.:
        return xmin * (xmax/xmin) ** step
    else:
        exp = 1. - alpha
        return (step * (xmax**exp - xmin**exp) + xmin**exp) ** (1 / exp)

def DataProj(w = 0, start = 0.5, end = 20, step = 0.5):
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
        result = pool.map(WorkProj, param)
    
    
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

def WorkProj(param):
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
            x, y, t = OrbGeoAlt(a = val, e = e, i = i ,w = w)
            
            # Appends results of orbital projection
            list1.append((x,y))
            # if val == 0 and e == 0 and i == 0:
            #     listt = []
            #     listt.append(t)
    
    return list1, param, listt

def DataHist(w = 0, step = 0.002, end = 10, which = "Linear", inclination = False, istep = None, estep = None):
    """
    """
    # Goes from both Linear and Log calculations to just Linear
    Linear = which
    
    # Parameters for Paralellization
    param = [
    # Row 1
    (0. , 0., w, end, step, Linear, inclination), (0., np.pi/6, w, end, step, Linear, inclination), (0., np.pi/3, w, end, step, Linear, inclination), (0., np.pi/2, w, end, step, Linear, inclination),
    # Row 2
    (0.5, 0., w, end, step, Linear, inclination), (0.5, np.pi/6, w, end, step, Linear, inclination), (0.5, np.pi/3, w, end, step, Linear, inclination), (0.5, np.pi/2, w, end, step, Linear, inclination),
    # Row 3
    (0.9, 0., w, end, step, Linear, inclination), (0.9, np.pi/6, w, end, step, Linear, inclination), (0.9, np.pi/3, w, end, step, Linear, inclination), (0.9, np.pi/2, w, end, step, Linear, inclination)
        
    ]
    
    if inclination:
        # param = [
        #     # Row 1
        #     (0.1, istep, w, end, step, Linear, inclination), (0.40, istep, w, end, step, Linear, inclination), (0.7, istep, w, end, step, Linear, inclination),
        #     # Row 2     
        #     (0.2, istep, w, end, step, Linear, inclination), (0.50, istep, w, end, step, Linear, inclination), (0.8, istep, w, end, step, Linear, inclination),     
        #     # Row 3     
        #     (0.3, istep, w, end, step, Linear, inclination), (0.6, istep, w, end, step, Linear, inclination), (0.9, istep, w, end, step, Linear, inclination)
            
        #          ]
        param = [estep, istep, w, end, step, Linear, inclination]
    
    start = time.perf_counter()
    # Multi Processing
    if inclination == True:
        totlist = Rchange(param = param)
    else:
        with Pool(processes = 6) as pool:
            totlist = pool.map(Rchange, param)
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

def MultiPlotProj(w = 0, start = 0.5, end = 20, step = 0.5):
    """
    Creates a 3 by 3 plot of 3 planetary orbits each with varying semimajor axes
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
     
     3 by 3 Plot of Planetary Orbits
    
    """
    
    # Initialize Lists
    listt = []
    rlist = []
    
    # Data points being created for each plots
    # 3 for each section
    
    # Gets everything ready for multiprocessing of orbital projections
    list, totparam, listt = DataProj(w = w, start = start, end = end, step = step)
    
    # DEPRECATED
    # vlistmin = []
    # vlistmax = []
    # veltot = []
    # for i in range(len(list)):
    #     iter = list[i]
    #     j = 0
    #     vellist = []
    #     param = totparam[i]
    #     for j in range(len(iter)):
    #         initialx, initialy = iter[j]
    #         t = listt[j]
    #         parameter = param[j]
    #         # Calculates velocity
    #         vel = Velocity(t, parameter)
    #         vel = vel.tolist()
    #         vellist.append(vel)
    #         # Takes only the local min and max
    #         vlistmin.append(min(vel))
    #         # print(min(vel))
    #         vlistmax.append(max(vel))
    #     veltot.append(vellist)
    # # Afterwards, takes the listed values and finds the global max and global min
    # velmax = np.max(vlistmax)
    # velmin = 0.1
    
    fig, axs = plt.subplots(3,4, figsize = (11,9), sharex=True,sharey=True,gridspec_kw=dict(hspace=0,wspace=0))               
    fig.suptitle("Orbital Projection with Alterations in e, i, and "r"$\omega$ = $\frac{\pi}{4}$")
    # Iterates through each subplot in the 3x3 figure
    for j, ax  in enumerate(axs.flatten()):
        # Takes the first data set in the list
        # These contain 12 other data sets  
        
        iterlist = list[j]
        g = 0
        param = totparam[j]
        
        # Finds points <= 0.01 for each projection
        rtemp, xchange, ychange, rtemp_log = Rchange(param, coords = True)
        
        vel = Velocity(param)
        # Iterates through the data clump to access
        # the data set 
        for g in range(len(iterlist)):
            # This contains each data set in the data clump
            initialx, initialy = iterlist[g]
            # Don't know why this is here, keeping it just in case
            # t = listt[g]
            # Calculates the velocity of each data point in the data set
            
            rlist.append(rtemp)
            # print("Last Velocity Output: ", vel[-1])
            velmax = np.max(vel[g])
            # # IMPORTANT!!!!!
            velmin = 0.1
            # velmin = np.min(vel)
            
            dot = DotSize(vel[g],velmax,velmin)
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
            ax.grid(True,color = "grey", linestyle="--", linewidth="0.25")
        
        # Plots an Einstein Ring Radius of 1 around each plot
        Circ2 = patches.Circle((0,0), 1, ec="k", fill=False, linestyle = ":", linewidth = 1)
        
        # Adds the Circle to the plot
        ax.add_patch(Circ2)
        
        # Just grabs the labels for each plot just before it iterates through again
        if j == 0:
            handles, labels = ax.get_legend_handles_labels()
            
        # Limits
        ax.set_xlim(-2,2)
        ax.set_ylim(-2,2)
    
    # Shows the legend of each data point
    fig.legend(handles,labels,fontsize="small")
    # Shows where the data values change according to the plot 
    plt.text(-15.75,7.95,"e=0")
    plt.text(-15.75,3.95,"e=0.5")
    plt.text(-15.75,-0.1,"e=0.9")
    plt.text(-12.25,10.25,"i=0")
    plt.text(-8.25,10.25,"i=30")
    plt.text(-4.25,10.25,"i=60")
    plt.text(0,10.25,"i=90")
    # plt.text()
    
    # Saves to Figure Folder
    plt.savefig("/College Projects/Microlensing Separation/Figures/Multi_a05_20_omega_pi_4.png")
    plt.show()
    
    return rlist

def MultiPlotHist(w = 0, step = 0.002, end = 10, which = "Log"):
    """
    """
    
    # totlinlist = [[] for _ in range(12)]
    # totloglist = [[] for _ in range(12)]
    # totlinsemilist = [[] for _ in range(12)]
    
    if which == "Linear":
        colorlist = ["black"]
    else:
        colorlist = ["black", "red", "blue"]
    
    totlist, param = DataHist(w = w, step = step, end = end, which = which)
    
    
    fig, axs = plt.subplots(3,4, figsize = (9,9), sharex=True,sharey=True,gridspec_kw=dict(hspace=0,wspace=0))               
    if which == "Linear":
        fig.suptitle("Orbital Projection with Alterations in e, i, and "r"$\omega$ = $\frac{\pi}{2}$" f"\n ({which})")
    else:
        fig.suptitle("Orbital Projection with Alterations in e, i, and "r"$\omega$ = $\frac{\pi}{2}$" f"\n (For Linear, Log, & Power Law)")
    # Iterates through each subplot in the 3x4 figure
    
    for j, ax  in enumerate(axs.flatten()):
        steplindict, x, y, steplogdict, steplinsemidict = totlist[j]
        
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
                weights=weights_lin
            )
        else:
            datahist_lin, bins, patches_lin = ax.hist(
                totlinlist, bins=logbins_lin, range=(0.5, end+0.5),
                stacked=True, histtype="step", alpha = 0.75, edgecolor = "black",
                weights=weights_lin, fc = "none",
            )
            datahist_log, bins, patches_log = ax.hist(
                totloglist, bins=logbins_log, range=(0.5, end+0.5),
                stacked=True, histtype="step", alpha = 0.75, edgecolor = "red",
                weights=weights_log, fc = "none",
            )
            datahist_linsemi , bins, patches_linsemi = ax.hist(
                totlinsemilist, bins=logbins_lin, range=(0.5, end+0.5),
                stacked=True, histtype="step", alpha = 0.40, edgecolor = "blue",
                weights=weights_linsemi, fc = "none",
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
        ax.text(6e0, 8.5, f"$e = {iterparam[0]}$") 
        ax.text(6e0, 7.5, f"$i = {round(iterparam[1],2)}$")
        ax.grid(True,color = "grey", linestyle="--", linewidth="0.25", axis = "x", which = "both")
        
    # plt.text(1.25e-6,25,"e=0")
    # plt.text(1.25e-6,15,"e=0.5")
    # plt.text(1.25e-6,5,"e=0.9")
    # plt.text(4e-5,30.5,"i=0")
    # plt.text(1.5e-3,30.5,"i=30")
    # plt.text(5.5e-2,30.5,"i=60")
    # plt.text(2.75, 30.5,"i=90")
    handles = [patches.Rectangle((0,0),1,1,color = c, ec = "w") for c in colorlist]
    if which == "Linear":
        labels = ["Linear"]
    else:
        labels = ["Linear", "Log", r"Power Law: $\alpha = 2$"]
    fig.legend(handles, labels)
    if which == "Linear":
        plt.savefig('/College Projects/Microlensing Separation/Figures/MultiHist_omega_pi_2_0001_Linear.png')
    else:
        plt.savefig('/College Projects/Microlensing Separation/Figures/MultiHist_omega_pi_2_0001_LinLawLog.png')
    plt.show()
    return totlist

def CompletePlotHist(param):
    """
    
    """
    w, step, end, inclination, which, estep_outer = param
    
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
    tothistlist = []
    evalhistlist = []
    
    # Create variables for bin sizes
    nbin = 200
    amin = 0.5
    amax = 21
    # Make logbinsizes for all
    logbinsize = (np.log10(amin)-np.log10(amax))/nbin
    logbins = np.geomspace(amin,amax, nbin)
    evalbins = np.linspace(0,0.9,80)
    
    
    # For making the stepthrough of omega
    wstep = np.linspace(0,np.pi/2,2)
    if inclination:
        # REMEMBER TO REMOVE IF STATMENTS FOR LINEAR (will eventually want linear in both)
        cosstep = np.linspace(0,1,2)
        istep = np.arccos(cosstep)
        # Only works if estep_outer has values in the list
        if len(estep_outer) != 0:
            estep = estep_outer
        else:
            estep = np.linspace(0,0.9, 2)
    for i in wstep:
        print("Value of omega currently: ", i, " and current position in array: ", np.where(wstep == i))
        # Each omega calculates its own data groups
        if inclination:
            steptotlist, param = DataHist(w = i, step = step, end = end, which = which, inclination = inclination, istep = istep, estep = estep)
        else:
            steptotlist, param = DataHist(w = i, step = step, end = end, which = which)
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
                # for j in range(len(steptotlist)):
                    # steplindict, x, y, steplogdict, blank = steptotlist[j]
                    steplindict, x, y, steplogdict, evaldict = steptotlist
                    # histlist = tothistlist[j]
                    histlist = tothistlist
                    evallist = evalhistlist
                    if which == "Log":
                        # Log histogram
                        totlogiter = steplogdict
                        totloglist = [key for key, val in totlogiter.items() for _ in range(val)]
                        hist_log, histbins_log = np.histogram(totloglist,bins = logbins, range=(0.5, end+0.5))
                        histlist.append((hist_log, histbins_log))
                        # E value histogram
                        totevaliter = evaldict
                        totevallist = [key for key, val in totevaliter.items() for _ in range(val)]
                        hist_eval, histbins_eval = np.histogram(totevallist,bins = evalbins, range=(0.0, 0.9))
                        evallist.append((hist_eval, histbins_eval))
                    elif which == "Linear":
                        # Linear histogram
                        totliniter = steplindict
                        totlinlist = [key for key, val in totliniter.items() for _ in range(val)]
                        hist_lin, histbins_lin = np.histogram(totlinlist,bins = logbins, range=(0.5, end+0.5))
                        histlist.append((hist_lin, histbins_lin))
                        # E value histogram
                        totevaliter = evaldict
                        totevallist = [key for key, val in totevaliter.items() for _ in range(val)]
                        hist_eval, histbins_eval = np.histogram(totevallist,bins = evalbins, range=(0.0, 0.9))
                        evallist.append((hist_eval, histbins_eval))
                    elif which == "Linear / a":
                        # Linear / a histogram
                        totliniter = steplindict
                        totlinlist = [key for key, val in totliniter.items() for _ in range(val)]
                        hist_lin, histbins_lin = np.histogram(totlinlist,bins = logbins, range=(0.5, end+0.5))
                        histlist.append((hist_lin, histbins_lin))
                        # E value histogram
                        totevaliter = evaldict
                        totevallist = [key for key, val in totevaliter.items() for _ in range(val)]
                        hist_eval, histbins_eval = np.histogram(totevallist,bins = evalbins, range=(0.0, 0.9))
                        evallist.append((hist_eval, histbins_eval))
                            
                    else:
                        return(print(f"Warning: {which} is not a valid point. Please use (Log) or (Linear) as your options"))
                    
                    tothistlist = histlist
                    evalhistlist = evallist
        gc.collect()
    
    # for j in range(12):
    #     # Turns dictionaries into lists for easier processing
    #     totlinlist[j] = [key for key, val in totlindict[j].items() for _ in range(val)]
    #     totloglist[j] = [key for key, val in totlogdict[j].items() for _ in range(val)]
                
    if inclination or len(estep_outer) == 0:
        fig, ax = plt.subplots(figsize = (9,9), sharex=True,sharey=True,gridspec_kw=dict(hspace=0,wspace=0))
        fig.suptitle("Orbital Projection with Alterations in e = 0.0-0.9, "r"$\cos{i} = 0$ to 1 , and " r"$\omega$ = $0$ to $\frac{\pi}{2}$" f"\n ({which})")
    elif len(estep_outer) == 0:
        fig, axs = plt.subplots(3,4, figsize = (9,9), sharex=True,sharey=True,gridspec_kw=dict(hspace=0,wspace=0))               
        fig.suptitle("Orbital Projection with Alterations in e = 0.0-0.9, "r"$\cos{i} = 0$ to 1 , and " r"$\omega$ = $0$ to $\frac{\pi}{2}$" f"\n ({which})")
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
    elif len(estep_outer) != 0:
        # for j, ax  in enumerate(axs.flatten()):
            # histlist = tothistlist[j]
            # iterparam = param[j]
            histlist = tothistlist
            iterparam = param
            
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
        
        
    
    # Saves plot
    if inclination == False:
        fig.legend(handles, labels)
        plt.savefig('/College Projects/Microlensing Separation/Figures/CompleteHist_0002_LinLog.png')
    elif len(estep_outer) == 0:
        fig.legend(handles, labels)
        plt.savefig('/College Projects/Microlensing Separation/Figures/CompleteHist_eccent_incline_10_0002_Log.png')
        plt.show()
    return tothistlist, evalhistlist

def UnityPlotHist(numestep = 10, numdiv = 2, which = "Log", etype = "Uniform"):
    """
    """
    totlist = []
    evallist = []
    
    if which == "Log":
        colorlist = ["red"]
        labels = ["Log"]
    elif which == "Linear":
        colorlist = ["black"]
        labels = ["Linear"]
    else:
        colorlist = ["blue"]
        labels = ["Linear / a"]
    
    # Create variables for bin sizes
    nbin = 200
    amin = 0.5
    amax = 21
    # Make logbinsizes for all
    logbinsize = (np.log10(amin)-np.log10(amax))/nbin
    
    
    # Initialize Lists
    if etype == "Uniform":
        estep = np.linspace(0,0.9, numestep)
    elif etype == "Gamma":
        alpha = 1.35 # Shape (Alpha)
        theta = 5.05 # Scale (Beta)
        x = np.linspace(0,0.9,numestep)
        estep = gamma.pdf(x, a = alpha, scale = theta)
    elif etype == "Circular":
        estep = 0
    esteplist = []*numdiv
    param = []
    eccbinsize = (0-1)/80
    
    # Slices estep into parts for parallelization
    slices = int(numestep / numdiv)
    if etype != "Circular":
        for i in range(numdiv):
            esteplist.append(estep[i*slices:(i+1)*slices])
    
            # Step, end, inclincation, which, estep
            param.append((0, 0.002, 20, True, which, esteplist[i]))
        
        # Processing using parallelization        
        with Pool(processes = numdiv) as pool:
                tothistlist = pool.map(CompletePlotHist, param)
        for j in range(len(tothistlist)):
            totlist.append(tothistlist[j][0])
            evallist.append(tothistlist[j][1])
                        
    else:
        totlist, evallist = CompletePlotHist(0,0.002,20,True, which)
        
        
    

    # FIGURE FOR SEMIMAJOR AXIS
    fig, ax = plt.subplots(figsize = (9,9), sharex=True,sharey=True,gridspec_kw=dict(hspace=0,wspace=0))
    fig.suptitle("Orbital Projection with Marginalizations in e = 0.0-0.9, "r"$\cos{i} = 0$ to 1 , and " r"$\omega$ = $0$ to $\frac{\pi}{2}$" f"\n ({which})")        
    # Combine Histogram Calculations
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
                    norm = np.abs(1 / (logbinsize * np.sum(tothist)))
                    StepPatch = ax.stairs(tothist * norm, bins, edgecolor = colorlist[0], fill = False)
                else:
                    tothist = tothist + hist
    ax.grid(True,color = "grey", linestyle="--", linewidth="0.25", axis = "x", which = "both")
    ax.set_xlim(0.5,20.5)
    ax.set_xscale("log")
    ax.set_xlabel(r"Semimajor Axis [$\log{a/R_e}$]")    
    ax.set_ylabel(r"Counts")
    
    handles = [patches.Rectangle((0,0),1,1,color = c, ec = "w") for c in colorlist]    
        
    fig.legend(handles, labels)
    plt.savefig('/College Projects/Microlensing Separation/Figures/CompleteHist_eccent_incline_80_0002_Log.png')

    # ECCENTRICITY FIGURE    
    fig, ax = plt.subplots(figsize = (9,9), sharex=True,sharey=True,gridspec_kw=dict(hspace=0,wspace=0))
    fig.suptitle("Eccentricity of Normalized Separation with Marginalizations in "r"$\cos{i} = 0$ to 1 , and " r"$\omega$ = $0$ to $\frac{\pi}{2}$" f"\n ({which})")        
    # Combine Histogram Calculations
    
    for i in range(len(evallist)):
        histlist = evallist[i]
        for val in range(len(histlist)):
                hist, bins = histlist[val]
                if etype == "Gamma":
                    norm = np.abs(1 / (eccbinsize * np.sum(hist)))
                    hist = norm * hist
                
                if val == 0:
                    if i == 0:
                        evalhist = np.zeros_like(hist)
                    evalhist = evalhist + hist
                
                elif val == len(histlist)-1 and i == len(evallist)-1:
                    evalhist = evalhist + hist
                    if etype == "Uniform":
                        norm = np.abs(1 / (eccbinsize * np.sum(evalhist)))
                        StepPatch = ax.stairs(evalhist * norm, bins, edgecolor = colorlist[0], fill = False)
                    
                    StepPatch = ax.stairs(evalhist , bins, edgecolor = colorlist[0], fill = False)
                else:
                    evalhist = evalhist + hist
    
    ax.grid(True,color = "grey", linestyle="--", linewidth="0.25", axis = "x", which = "both")
    ax.set_xlim(0,1)
    # Comment might change
    # ax.set_xscale("log")
    ax.set_xlabel(r"Eccentricity")    
    ax.set_ylabel(r"Counts")
    
    handles = [patches.Rectangle((0,0),1,1,color = c, ec = "w") for c in colorlist]    
        
    fig.legend(handles, labels)
    plt.savefig('/College Projects/Microlensing Separation/Figures/Eccentricity_Log.png')
    
    return tothist
    
if __name__ == "__main__":
    # rlist = MultiPlotProj(w = np.pi/4., start = 0.5, end = 20, step = 0.5)
    # rtemp = MultiPlotHist(w = np.pi/2., step = 0.002, end = 20, which = "Log")
    # clist = CompletePlotHist(step = 0.002, end = 20, inclination = True, which = "Log")
    tothist = UnityPlotHist(numestep=2, numdiv=2, which = "Log", etype = "Uniform")

# x,y,t = OrbGeoAlt(a=1, e=0.0,w=np.pi/4, i = np.pi/6)
# param = [0.5, 0.5, np.pi/2, 0]
# vel = Velocity(t, param)
# vmax = np.max(vel)
# vmin = 0.1
# dot = DotSize(vel, vmax, vmin)
# rtemp, xchange, ychange = Rchange(x,y,param[0])
# fig, axs = plt.subplots(figsize = (8,8))
# axs.scatter(x,y, s = dot)
# data = axs.scatter(xchange, ychange, s = 8, color = "yellow")
# Circ2 = patches.Circle((0,0), 1, ec="k", fill=False, linestyle = ":", linewidth = 1)
# axs.add_patch(Circ2)
# axs.grid(True,color = "grey", linestyle="--", linewidth="0.25")
# axs.set_xlim(-2,2)
# axs.set_ylim(-2,2)
# plt.show()
# rlist = Rchange(x,y,1.5)
# print(rlist)
# print(InvVelocity(t,x,y)[-1])
# fig, axs = plt.subplots(3,3, figsize = (7,7), sharex=True,sharey=True,gridspec_kw=dict(hspace=0,wspace=0))
# fig.suptitle("Velocity of Orbital Projections for New Equation")
# colors = ["g", "r", "b"]
# labels = ["a = 0.5", "a = 1.0", "a = 1.5"]

