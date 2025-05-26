import astropy.constants as ac
import numpy as np
import scipy.differentiate as diff
import scipy.optimize as sc
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def NewtRaf(M, e):
    """
    Uses scipy functions to produce a 
    root of a real function using the
    Newton-Raphson method.
    
    -------------------------------
    ### Parameters
    
    M : float <br>
        A combination of Period and semimajor axis.
    
    e : float <br>
        Constant of eccentricity.
    
    -------------------------------
    ### Returns
    
    Solution : float <br>
        Solution to Kepler Equation
    """
    initial = M+e*np.sin(M) # Bascially starting with M as our initial guess
    Solution = np.zeros_like(M)
    
    Ei = initial # Initial Guess
    Mi = M # Iterating through M
        
    # Uses Kepler's Equation and Derivative to solve
    Solution = sc.newton(Kepler, Ei, fprime = DKepler, args = (Mi,e))
    
    # Previous attempt at solving for E
    # Elist = []
    # for i in range(n-1):
    #     if i == 0:
    #         Eprime = diff.derivative(fE,Elist[i])
    #         Elist.append(initial-(fE/Eprime))
    #     Eprime = diff.derivative(fE,Elist[i])
    #     Elist.append(Elist[i]-(fE/Eprime))
    # return(Elist[-1])
    
    return(Solution)

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
    
    """
    xt = []
    yt = []
    # Assume time at periastron (closest point to star),
    # t0, as our starting time so t0 = 0
    # Check with Scott to see if we can do this
    t0 = 0
    
    # Initial Equations
    A = a*(np.cos(W)*np.cos(w) - np.sin(W)*np.sin(w)*np.cos(i))
    B = a*(np.sin(W)*np.cos(w) - np.cos(W)*np.sin(w)*np.cos(i))
    F = a*(-np.cos(W)*np.sin(w) - np.sin(W)*np.cos(w)*np.cos(i))
    G = a*(-np.sin(W)*np.sin(w) + np.cos(W)*np.cos(w)*np.cos(i))
     
    # Equation for P (G*M_sun = 1 in Solar Units)
    P = np.sqrt((4*np.pi)/(1)*a**3)
    
    t = np.linspace(t0, t0 + P, 4000)
    
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

def OrbGeoAlt(t0=0, a=1, w = 0, W = 0, i = 0, e = 0):
    
    xt = []
    yt = []
    # Assume time at periastron (closest point to star),
    # t0, as our starting time so t0 = 0
    # Check with Scott to see if we can do this
    t0 = 0
    
    # Initial Equations
    A = a*(np.cos(W)*np.cos(w) - np.sin(W)*np.sin(w)*np.cos(i))
    B = a*(np.sin(W)*np.cos(w) - np.cos(W)*np.sin(w)*np.cos(i))
    F = a*(-np.cos(W)*np.sin(w) - np.sin(W)*np.cos(w)*np.cos(i))
    G = a*(-np.sin(W)*np.sin(w) + np.cos(W)*np.cos(w)*np.cos(i))
    
    j = np.linspace(1,4000,4000)
    phi = [(val/4000)*(2*np.pi)+t0 for val in j]
    
    for k in range(len(phi)):
        
        # Solve Kepler Equation
        Et = NewtRaf(M = phi[k], e=e)
    
        # Function of X and Y according to E
        Xt = np.cos(Et) - e
        Yt = np.sqrt(1-e**2)*np.sin(Et)
    
        # Actual x any y functions of t
        xt.append(A*Xt + F*Yt)
        yt.append(B*Xt + G*Yt)
    
    
    return (xt, yt, phi)

def Velocity(t,x,y, param):
    """
    Calculates the inverse velocity of a planet's orbit.
    
    ------
    ### Parameters
    
    t : arraylike <br>
        time function of a planet's orbit
        
    x : arraylike <br>
        x position of a planet's orbit according to time
        
    y : arraylike <br>
        y position of a planet's orbit according to time
        
    ------
    ### Returns
    
    vel : array of floats <br>
        inverse velocity of a planet's orbit
    """
    # Change of t will be the same for all values
    # because of how np.linespace works (evenly spaced)
    param = param
    changet = np.abs(t[1] - t[0])
    vel = []
    if len(param) > 3:
        i = param[-1]
    else:
        i = 0
    x1, y1, t1 = OrbGeo(-changet, a = param[0], e = param[1], w = param[2], i = i)
    x1array = np.array(x1)
    y1array = np.array(y1)
    x2, y2, t2 = OrbGeo(changet, a = param[0], e = param[1], w = param[2], i = i)
    x2array = np.array(x2)
    y2array = np.array(y2)
    # for i in range(len(t)):
    #     t1 = t[i] - changet
    #     t2 = t[i] + changet
    #     x1 = x[i]
    #     y1 = y[i]
    #     # If i is just before the last iteration,
    #     # it means it made a complete orbit
    #     if i == len(t)-1:
    #         # THIS SOLUTION FORCES
    #         # OUR TIME FUNCTION TO ALWAYS BE 
    #         # AN EVEN INTEGER NUMBER OF PI
    #         x2 = x[-2]
    #         y2 = y[-2]
    #     else:
    #         x2 = x[i+1]
    #         y2 = y[i+1]
        
        # Change of degrees function
    changedeg = np.sqrt(np.add(np.power(np.subtract(x2array,x1array),2),np.power(np.subtract(y2array,y1array),2)))
        
        # In case we have a 0 change of deg,
        # we set this to 0
        # if changedeg == 0:
        #     vel.append(0)
        # else:
            # Velocity equation
            # vel.append(np.abs(np.log10(1/np.abs((changedeg/(t2-t1))))))
    vel = (np.abs(np.divide(changedeg,(2*changet))))
    # Find max Velocity for solving for ratio
    return vel

def DotSize(vel, velmax, velmin):
    """
    Calculates the dot size of a planet's orbit using its velocity.
    
    ------
    ### Parameters
    
    vel : arraylike <br>
        velocity of a planet's orbit
        
    velmax : float <br>
        maximum velocity of ENTIRE PLOT
        
    velmin : float <br>
        minimum velocity of ENTIRE PLOT
        
    ------
    ### Returns
    
    ratio : list of floats <br>
        dot size proportional to velocity
    """
    ratio = []
    
    for i in range(len(vel)):
        # Similar to if statement in changedeg,
        # except for the ratio
        if vel[i] == 0:
            ratio.append(0.1)
        else:
            # ratio equation (deprecated)
            # ratio.append((np.abs(velmin)/np.abs(vel[i]))*2+0.1)
            
            # ratio equation (NEW)
            ratio.append(np.abs(40.0 - ((np.abs(vel[i]) - np.abs(velmin))/(np.abs(velmax) - np.abs(velmin)))*39.1))
    
    # One possible way to fix dot size is to isolate the last column with if statements
    # and solve for only 1/v    
    
    return ratio

def Rchange(t,x,y):
    """
    Stores all the points in the data where the radius of the orbit
    is very close to the Einstein ring radius.
    
    --------
    ### Parameters
    
    t : arraylike <br>
        time function of a planet's orbit
        
    x : arraylike <br>
        x position of a planet's orbit according to time
        
    y : arraylike <br>
        y position of a planet's orbit according to time
        
    ------
    ### Returns
    
    rlist : list of floats <br>
        times at which the radius was close to the ring radius
    
    num : list of integers <br>
        position of times in time function
    """
    
    rlist = [] # Time at which it was less than 0.001
    num = [] # Position in list given length of x,y,& t
    r0 = 1 # Einstein Ring Radius
    # All values equated from t have the same length
    for i in range(len(t)):
        x1 = x[i]
        y1 = y[i]
        # Radius equation
        r = np.sqrt(x1**2+y1**2)
        
        if np.abs(r-r0) <= 0.001:
            # Stores the point in time in which t was found
            rlist.append(t[i])
            num.append(i)
    return rlist , num

def MultiPlot(t0 = 0, a=1, w = 0, W = 0, i = 0, e = 0, n = 3):
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
    # Data points being created for each plots
    # 3 for each section
    # Initialize Lists
    listt = []
    vlistmin = []
    vlistmax = []
    veltot = []
    
    # Top Left
    k=0.5
    list1 = []
    paramlist1 = []
    while k <= 1.5:
        x1, y1, t1 = OrbGeo(a=k,e=0, w=w)
        list1.append((x1,y1))
        listt.append(t1)
        paramlist1.append([k, 0, w])
        k+=0.5
    # Middle Left
    k=0.5
    list2=[]
    paramlist2 = []
    while k <= 1.5:
        x1, y1, t1 = OrbGeo(a=k,e=0.5, w=w)
        list2.append((x1,y1))
        paramlist2.append([k, 0.5, w])
        k+=0.5
    # Bottom Left    
    k=0.5
    list3=[]
    paramlist3 = []
    while k <= 1.5:
        x1, y1, t1 = OrbGeo(a=k,e=0.9, w=w)
        list3.append((x1,y1))
        paramlist3.append([k, 0.9, w])
        k+=0.5
    
    # x1, y1 = OrbGeo(t,e = 0,)
    # x11, y2 = OrbGeo(t,e = 0.5)
    # x111, y3 = OrbGeo(t,e = 0.9)
    
    # Top Middle
    k=0.5
    list4=[]
    paramlist4 = []
    while k <= 1.5:
        x1, y1, t1 = OrbGeo(a=k, e = 0, i = np.pi/4, w = w)
        list4.append((x1,y1))
        paramlist4.append([k, 0, w, np.pi/4])
        k+=0.5
    # Center
    k=0.5
    list5=[]
    paramlist5 = []
    while k <= 1.5:
        x1, y1, t1 = OrbGeo(a=k, e = 0.5, i = np.pi/4, w = w)
        list5.append((x1,y1))
        paramlist5.append([k, 0.5, w, np.pi/4])
        k+=0.5
    # Bottom Middle
    k=0.5
    list6=[]
    paramlist6 = []
    while k <= 1.5:
        x1, y1, t1 = OrbGeo(a = k, e = 0.9, i = np.pi/4, w = w)
        list6.append((x1,y1))
        paramlist6.append([k, 0.9, w, np.pi/4])
        k+=0.5
    
    # x2, y11 = OrbGeo(t,e = 0, i = np.pi/4, w = np.pi/2)
    # x22, y22 = OrbGeo(t,e = 0.5, i = np.pi/4, w = np.pi)
    # x222, y33 = OrbGeo(t,e = 0.9, i = np.pi/4, w = 3*np.pi/2)
    
    # Top Right
    k=0.5
    list7=[]
    paramlist7 = []
    while k <= 1.5:
        x1, y1, t1 = OrbGeo(a = k, e = 0, i = np.pi/2, w = w)
        list7.append((x1,y1))
        paramlist7.append([k, 0, w, np.pi/2])
        k+=0.5
    # Middle Right
    k=0.5
    list8=[]
    paramlist8 = []
    while k <= 1.5:
        x1, y1, t1 = OrbGeo(a = k, e = 0.5, i = np.pi/2, w = w)
        list8.append((x1,y1))
        paramlist8.append([k, 0.5, w, np.pi/2])
        k+=0.5
    # Bottom Right
    k=0.5
    list9=[]
    paramlist9 = []
    while k <= 1.5:
        x1, y1, t1 = OrbGeo(a = k, e = 0.9, i = np.pi/2, w = w)
        list9.append((x1,y1))
        paramlist9.append([k, 0.9, w, np.pi/2])
        k+=0.5
    
    # x3, y111 = OrbGeo(t, e = 0, i = np.pi/2)
    # x33, y222 = OrbGeo(t, e=0.5, i = np.pi/2)
    # x333, y333 = OrbGeo(t, e = 0.9, i = np.pi/2)
    
    list = [
        list1,list4,list7,
        list2,list5,list8,
        list3,list6,list9
    ]
    totparam = [
        paramlist1, paramlist2, paramlist3,
        paramlist4, paramlist5, paramlist6,
        paramlist7, paramlist8, paramlist9           
    ]
    
    # Initially Calculates the Velocity to place these into a list
    for i in range(len(list)):
        iter = list[i]
        j = 0
        vellist = []
        param = totparam[i]
        for j in range(len(iter)):
            initialx, initialy = iter[j]
            t = listt[j]
            parameter = param[j]
            # Calculates velocity
            vel = Velocity(t,initialx,initialy, parameter)
            vel = vel.tolist()
            vellist.append(vel)
            # Takes only the local min and max
            vlistmin.append(min(vel))
            # print(min(vel))
            vlistmax.append(max(vel))
        veltot.append(vellist)
    # Afterwards, takes the listed values and finds the global max and global min
    velmax = np.max(vlistmax)
    velmin = np.min(vlistmin)
    
    fig, axs = plt.subplots(n,n, figsize = (7,7), sharex=True,sharey=True,gridspec_kw=dict(hspace=0,wspace=0))               
    fig.suptitle("Orbital Projection with Alterations in e, i, and "r"$\omega$ = $\pi / 2$")
    # Iterates through each subplot in the 3x3 figure
    for j, ax  in enumerate(axs.flatten()):
        # Takes the first data clump in the list
        # This contains three other data sets
        iterlist = list[j]
        g = 0
        param = totparam[j]
        # Iterates through the data clump to access
        # the data set
        for g in range(len(iterlist)):
            # This contains each data set in the data clump
            initialx, initialy = iterlist[g]
            t = listt[g]
            parameter = param[g]
            # Calculates the velocity of each data point in the data set
            vel = Velocity(t, initialx, initialy, parameter)
            dot = DotSize(vel,velmax,velmin)
            
            # Determines the colors of each data set according
            # to its positioning
            if g == 0:
                color = "g"
                label = "a = 0.5"
            elif g == 1:
                color = "r"
                label = "a = 1.0"
            else:
                color = "b"
                label = "a = 1.5"
            # Plots the data set, including the dot size according to velocity        
            dataproj = ax.scatter(initialx, initialy,s = dot, color = color, label = label)
            
            # Creates the grid for each plot
            ax.grid(True,color = "grey", linestyle="--", linewidth="0.25")
        
        # Circ1 = patches.Circle((0,0), 0.5, ec= "b", fill=False, linestyle = ":", linewidth = 1)
        
        # Plots an Einstein Ring Radius of 1 around each plot
        Circ2 = patches.Circle((0,0), 1, ec="k", fill=False, linestyle = ":", linewidth = 1)
        
        
        # Circ3 = patches.Circle((0,0), 1.5, ec="r", fill=False, linestyle = ":", linewidth = 1)
        
        # ax.add_patch(Circ1)
        
        # Adds the Circle to the plot
        ax.add_patch(Circ2)
        # ax.add_patch(Circ3)
        
        # Just grabs the labels for each plot just before it iterates through again
        if j == 0:
            handles, labels = ax.get_legend_handles_labels()
            
        # Limits
        ax.set_xlim(-2,2)
        ax.set_ylim(-2,2)
    # fig.legend([Circ1,Circ2,Circ3,linelabel],["a = 0.5", "a = 1", "a = 1.5", "Observed Orbit"], fontsize = "small")
    # Shows the legend of each data point
    fig.legend(handles,labels,fontsize="small")
    # Shows where the data values change according to the plot 
    plt.text(-11.5,7.90,"e=0")
    plt.text(-11.5,3.90,"e=0.5")
    plt.text(-11.5,-0.1,"e=0.9")
    plt.text(-8,10.5,"i=0")
    plt.text(-4,10.5,"i=45")
    plt.text(-0.5,10.5,"i=90")
    # plt.text()
    plt.show()
    
    return vlistmin, vlistmax, veltot

# vlistmin, vlistmax, veltot = MultiPlot(w = np.pi/2)
# print(list)
x,y,t = OrbGeoAlt(a=1, e=0.9,i=np.pi/2)
fig, axs = plt.subplots(figsize = (7,7))
axs.scatter(x,y)
axs.set_xlim(-2,2)
plt.show()
# vel = InvVelocity(t,x,y)
# print(min(vel))
# rlist, num = Rchange(t,x,y)
# print(num)
# print(rlist)
# print(InvVelocity(t,x,y)[-1])
# fig, axs = plt.subplots(3,3, figsize = (7,7), sharex=True,sharey=True,gridspec_kw=dict(hspace=0,wspace=0))
# fig.suptitle("Velocity of Orbital Projections")
# colors = ["g", "r", "b"]
# labels = ["a = 0.5", "a = 1.0", "a = 1.5"]

# for j, ax  in enumerate(axs.flatten()):
#     if j == 0:
#         ax.hist(veltot[j], 10, stacked = True, facecolor = colors, label = labels, log = True)
#     elif j == 6 or 7 or 8:
#         ax.hist(veltot[j], 10, stacked = False, facecolor = colors, log = True)
#     else:
#         ax.hist(veltot[j], 10, stacked = True, facecolor = colors, log = True)
#     ax.autoscale()    
# fig.legend()
# plt.show()
# line = np.linspace(-0.5,0.5,5)
# y = [0, 0, 0, 0 ,0]
# dots = [0.1, 1, 2 ,4 ,5]
# fig, ax = plt.subplots()
# ax.scatter(line,y, s = dots, label = "0.1, 1, 2, 4, 5")
# ax.set_xlim(-.75,.75)
# ax.set_ylim(-.75,.75)
# fig.legend()
# plt.show()