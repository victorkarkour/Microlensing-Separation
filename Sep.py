import astropy.constants as ac
import numpy as np
import scipy.signal as signal
import scipy.optimize as sc
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
from multiprocessing import Pool
import gc

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
    e, i, w, end, step = param
    vel = []
    # In case inclination angle is used, then it makes sure to include it
    
    stepthrough = np.arange(0.5, end + step, step)
    
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
    
    num = np.subtract(np.abs(vel) , np.abs(velmin))
    
    denom = np.abs(velmax) - np.abs(velmin) 
    
    ratio =  (40.0 - (np.divide(num,denom)) * 39.0)
    
    return ratio

def Rchange(param, coords = False):
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
    rlistlog = [] # Point at which it was less than 0.001
    rlistlin = []
    xlist = []
    ylist = []
    r0 = 1 # Einstein Ring Radius
    e, i, w, end, step, Linear = param
    
    if Linear == True:
        stepthrough = np.arange(0.5, end + step, step)
        
        for val in stepthrough:
            x, y, t = OrbGeoAlt(a = val, e = e, i = i ,w = w)
    
            r = np.sqrt(x**2+y**2)

            if coords == False:
                rlistlin.append(np.where(np.abs(r-r0)<=0.01, val, 0))
            else: 
                rlistlin.append(np.where(np.abs(r-r0)<=0.01, val, 0))
                xlist.append(np.where(np.abs(r-r0)<=0.01, x, None))
                ylist.append(np.where(np.abs(r-r0)<=0.01, y, None))
    elif Linear == False:
        # Log Portion
        steps = np.linspace(0, 10000,10000)
        loga = np.log10(0.5) + steps/10000 * (np.log10(end)-np.log10(0.5))
        stepthrough = 10**loga 
   
        for val in stepthrough:
            x, y, t = OrbGeoAlt(a = val, e = e, i = i ,w = w)
    
            r = np.sqrt(x**2+y**2)
        
            rlistlog.append(np.where(np.abs(r-r0)<=0.01, val, 0))
        gc.collect()
        # Linear Portion
        stepthrough = np.arange(0.5, end + step, step)
        for val in stepthrough:
            x, y, t = OrbGeoAlt(a = val, e = e, i = i ,w = w)
    
            r = np.sqrt(x**2+y**2)

            if coords == False:
                rlistlin.append(np.where(np.abs(r-r0)<=0.01, val, 0))
            else: 
                rlistlin.append(np.where(np.abs(r-r0)<=0.01, val, 0))
                xlist.append(np.where(np.abs(r-r0)<=0.01, x, None))
                ylist.append(np.where(np.abs(r-r0)<=0.01, y, None))
               
    return rlistlin, xlist, ylist, rlistlog

def DataProj(t0 = 0.0, a=1.0, w = 0.0, W = 0.0, i = 0.0, e = 0.0, startinga = 1.25):
    """
    
    """
    listt = []
    
    if startinga == 1.25:
        # Top Left
        k=0.75
        list1 = []
        paramlist1 = []
        while k <= 1.25:
            x1, y1, t1 = OrbGeoAlt(a=k,e=0, w=w)
            list1.append((x1,y1))
            listt.append(t1)
            paramlist1.append([0.0, w, 0])
            k+=0.25
        # Middle Left
        k=0.75
        list2=[]
        paramlist2 = []
        while k <= 1.25:
          x1, y1, t1 = OrbGeoAlt(a=k,e=0.5, w=w)
          list2.append((x1,y1))
          # listt.append(t1)
          paramlist2.append([k, 0.5, w])
          k+=0.25
        # Bottom Left    
        k=0.75
        list3=[]
        paramlist3 = []
        while k <= 1.25:
            x1, y1, t1 = OrbGeoAlt(a=k,e=0.9, w=w)
            list3.append((x1,y1))
            # listt.append(t1)
            paramlist3.append([k, 0.9, w])
            k+=0.25
        print("Column 1 Complete.")
    
    
        # Top Middle
        k=0.75
        list4=[]
        paramlist4 = []
        while k <= 1.25:
            x1, y1, t1 = OrbGeoAlt(a=k, e = 0, i = np.pi/6.0, w = w)
            list4.append((x1,y1))
            # listt.append(t1)
            paramlist4.append([k, 0.0, w, np.pi/4.0])
            k+=0.25
        # Center
        k=0.75
        list5=[]
        paramlist5 = []
        while k <= 1.25:
            x1, y1, t1 = OrbGeoAlt(a=k, e = 0.5, i = np.pi/6.0, w = w)
            list5.append((x1,y1))
            # listt.append(t1)
            paramlist5.append([k, 0.5, w, np.pi/4.0])
            k+=0.25
        # Bottom Middle
        k=0.75
        list6=[]
        paramlist6 = []
        while k <= 1.25:
            x1, y1, t1 = OrbGeoAlt(a = k, e = 0.9, i = np.pi/6.0, w = w)
            list6.append((x1,y1))
            # listt.append(t1)
            paramlist6.append([k, 0.9, w, np.pi/4.0])
            k+=0.25
        print("Column 2 Complete.")
        
        # Top Right
        k=0.75
        list7=[]
        paramlist7 = []
        while k <= 1.25:
            x1, y1, t1 = OrbGeoAlt(a = k, e = 0.0, i = np.pi/3.0, w = w)
            list7.append((x1,y1))
            # listt.append(t1)
            paramlist7.append([k, 0.0, w, np.pi/2.0])
            k+=0.25
        # Middle Right
        k=0.75
        list8=[]
        paramlist8 = []
        while k <= 1.25:
            x1, y1, t1 = OrbGeoAlt(a = k, e = 0.5, i = np.pi/3.0, w = w)
            list8.append((x1,y1))
            # listt.append(t1)
            paramlist8.append([k, 0.5, w, np.pi/2.0])
            k+=0.25
        # Bottom Right
        k=0.75
        list9=[]
        paramlist9 = []
        while k <= 1.25:
            x1, y1, t1 = OrbGeoAlt(a = k, e = 0.9, i = np.pi/3.0, w = w)
            list9.append((x1,y1))
            # listt.append(t1)
            paramlist9.append([k, 0.9, w, np.pi/2.0])
            k+=0.25
        print("Column 3 Complete.")
            
        k=0.75
        list10=[]
        paramlist10 = []
        while k <= 1.25:
            x1, y1, t1 = OrbGeoAlt(a = k, e = 0.0, i = np.pi/2.0, w = w)
            list10.append((x1,y1))
            # listt.append(t1)
            paramlist10.append([k, 0.9, w, np.pi/2.0])
            k+=0.25
        k=0.75
        list11=[]
        paramlist11 = []
        while k <= 1.25:
            x1, y1, t1 = OrbGeoAlt(a = k, e = 0.5, i = np.pi/2.0, w = w)
            list11.append((x1,y1))
            # listt.append(t1)
            paramlist11.append([k, 0.9, w, np.pi/2.0])
            k+=0.25
        k=0.75
        list12=[]
        paramlist12 = []
        while k <= 1.25:
            x1, y1, t1 = OrbGeoAlt(a = k, e = 0.9, i = np.pi/2.0, w = w)
            list12.append((x1,y1))
            # listt.append(t1)
            paramlist12.append([k, 0.9, w, np.pi/2.0])
            k+=0.25
        print("Column 4 Complete.")
            
    elif startinga == 1.5: 
        # Top Left
        k=0.5
        list1 = []
        paramlist1 = []
        while k <= 1.5:
            x1, y1, t1 = OrbGeoAlt(a=k,e=0, w=w)
            list1.append((x1,y1))
            listt.append(t1)
            paramlist1.append([k, 0.0, w])
            k+=0.5
        # Middle Left
        k=0.5
        list2=[]
        paramlist2 = []
        while k <= 1.5:
          x1, y1, t1 = OrbGeoAlt(a=k,e=0.5, w=w)
          list2.append((x1,y1))
          # listt.append(t1)
          paramlist2.append([k, 0.5, w])
          k+=0.5
        # Bottom Left    
        k=0.5
        list3=[]
        paramlist3 = []
        while k <= 1.5:
            x1, y1, t1 = OrbGeoAlt(a=k,e=0.9, w=w)
            list3.append((x1,y1))
            # listt.append(t1)
            paramlist3.append([k, 0.9, w])
            k+=0.5
        print("Column 1 Complete.")
    
        # Top Middle
        k=0.5
        list4=[]
        paramlist4 = []
        while k <= 1.5:
            x1, y1, t1 = OrbGeoAlt(a=k, e = 0, i = np.pi/6.0, w = w)
            list4.append((x1,y1))
            # listt.append(t1)
            paramlist4.append([k, 0.0, w, np.pi/4.0])
            k+=0.5
        # Center
        k=0.5
        list5=[]
        paramlist5 = []
        while k <= 1.5:
            x1, y1, t1 = OrbGeoAlt(a=k, e = 0.5, i = np.pi/6.0, w = w)
            list5.append((x1,y1))
            # listt.append(t1)
            paramlist5.append([k, 0.5, w, np.pi/4.0])
            k+=0.5
        # Bottom Middle
        k=0.5
        list6=[]
        paramlist6 = []
        while k <= 1.5:
            x1, y1, t1 = OrbGeoAlt(a = k, e = 0.9, i = np.pi/6.0, w = w)
            list6.append((x1,y1))
            # listt.append(t1)
            paramlist6.append([k, 0.9, w, np.pi/4.0])
            k+=0.5
        print("Column 2 Complete.")    
    
        # Top Right
        k=0.5
        list7=[]
        paramlist7 = []
        while k <= 1.5:
            x1, y1, t1 = OrbGeoAlt(a = k, e = 0.0, i = np.pi/3.0, w = w)
            list7.append((x1,y1))
            # listt.append(t1)
            paramlist7.append([k, 0.0, w, np.pi/2.0])
            k+=0.5
        # Middle Right
        k=0.5
        list8=[]
        paramlist8 = []
        while k <= 1.5:
            x1, y1, t1 = OrbGeoAlt(a = k, e = 0.5, i = np.pi/3.0, w = w)
            list8.append((x1,y1))
            # listt.append(t1)
            paramlist8.append([k, 0.5, w, np.pi/2.0])
            k+=0.5
        # Bottom Right
        k=0.5
        list9=[]
        paramlist9 = []
        while k <= 1.5:
            x1, y1, t1 = OrbGeoAlt(a = k, e = 0.9, i = np.pi/3.0, w = w)
            list9.append((x1,y1))
            # listt.append(t1)
            paramlist9.append([k, 0.9, w, np.pi/2.0])
            k+=0.5
        print("Column 3 Complete.")
        
        k=0.5
        list10=[]
        paramlist10 = []
        while k <= 1.5:
            x1, y1, t1 = OrbGeoAlt(a = k, e = 0.0, i = np.pi/2.0, w = w)
            list10.append((x1,y1))
            # listt.append(t1)
            paramlist10.append([k, 0.9, w, np.pi/2.0])
            k+=0.5    
        k=0.5
        list11=[]
        paramlist11 = []
        while k <= 1.5:
            x1, y1, t1 = OrbGeoAlt(a = k, e = 0.5, i = np.pi/2.0, w = w)
            list11.append((x1,y1))
            # listt.append(t1)
            paramlist11.append([k, 0.9, w, np.pi/2.0])
            k+=0.5
        k=0.5
        list12=[]
        paramlist12 = []
        while k <= 1.5:
            x1, y1, t1 = OrbGeoAlt(a = k, e = 0.9, i = np.pi/2.0, w = w)
            list12.append((x1,y1))
            # listt.append(t1)
            paramlist12.append([k, 0.9, w, np.pi/2.0])
            k+=0.5
        print("Column 4 Complete.")    
    else:
        # Top Left
        k=0.5
        list1 = []
        paramlist1 = []
        while k <= startinga:
            x1, y1, t1 = OrbGeoAlt(a=k,e=0, w=w)
            list1.append((x1,y1))
            listt.append(t1)
            k+=0.5
        paramlist1 = [(0, 0, w)]
        # Middle Left
        k=0.5
        list2=[]
        paramlist2 = []
        while k <= startinga:
          x1, y1, t1 = OrbGeoAlt(a=k,e=0.5, w=w)
          list2.append((x1,y1))
          k+=0.5
        paramlist2 = [(0.5, 0, w)]
        # Bottom Left    
        k=0.5
        list3=[]
        paramlist3 = []
        while k <= startinga:
            x1, y1, t1 = OrbGeoAlt(a=k,e=0.9, w=w)
            list3.append((x1,y1))
            k+=0.5
        paramlist3 = [(0.9, 0, w)]
        print("Column 1 Complete.")
    
        # Top Middle
        k=0.5
        list4=[]
        paramlist4 = []
        while k <= startinga:
            x1, y1, t1 = OrbGeoAlt(a=k, e = 0, i = np.pi/6.0, w = w)
            list4.append((x1,y1))
            k+=0.5
        paramlist4 = [(0.0, np.pi/6.0, w)]    
        # Center
        k=0.5
        list5=[]
        paramlist5 = []
        while k <= startinga:
            x1, y1, t1 = OrbGeoAlt(a=k, e = 0.5, i = np.pi/6.0, w = w)
            list5.append((x1,y1))
            k+=0.5
        paramlist5 = [(0.5, np.pi/6.0, w)] 
        # Bottom Middle
        k=0.5
        list6=[]
        paramlist6 = []
        while k <= startinga:
            x1, y1, t1 = OrbGeoAlt(a = k, e = 0.9, i = np.pi/6.0, w = w)
            list6.append((x1,y1))
            k+=0.5
        paramlist6 = [(0.9, np.pi/6.0, w)] 
        print("Column 2 Complete.")

        # Top Right
        k=0.5
        list7=[]
        paramlist7 = []
        while k <= startinga:
            x1, y1, t1 = OrbGeoAlt(a = k, e = 0.0, i = np.pi/3.0, w = w)
            list7.append((x1,y1))           
            k+=0.5
        paramlist7 = [(0.0, np.pi/3.0, w)] 
        # Middle Right
        k=0.5
        list8=[]
        paramlist8 = []
        while k <= startinga:
            x1, y1, t1 = OrbGeoAlt(a = k, e = 0.5, i = np.pi/3.0, w = w)
            list8.append((x1,y1))
            k+=0.5
        paramlist8 = [(0.5, np.pi/3.0, w)] 
        # Bottom Right
        k=0.5
        list9=[]
        paramlist9 = []
        while k <= startinga:
            x1, y1, t1 = OrbGeoAlt(a = k, e = 0.9, i = np.pi/3.0, w = w)
            list9.append((x1,y1))
            k+=0.5
        paramlist9 = [(0.9, np.pi/3.0, w)] 
        print("Column 3 Complete.")
        
        k=0.5
        list10=[]
        paramlist10 = []
        while k <= startinga:
            x1, y1, t1 = OrbGeoAlt(a = k, e = 0.0, i = np.pi/2.0, w = w)
            list10.append((x1,y1))
            k+=0.5 
        paramlist10 = [(0.0, np.pi/2.0, w)]    
        k=0.5
        list11=[]
        paramlist11 = []
        while k <= startinga:
            x1, y1, t1 = OrbGeoAlt(a = k, e = 0.5, i = np.pi/2.0, w = w)
            list11.append((x1,y1))
            k+=0.5
        paramlist11 = [(0.5, np.pi/2.0, w)]  
        k=0.5
        list12=[]
        paramlist12 = []
        while k <= startinga:
            x1, y1, t1 = OrbGeoAlt(a = k, e = 0.9, i = np.pi/2.0, w = w)
            list12.append((x1,y1))
            k+=0.5
        paramlist12 = [(0.9, np.pi/2.0, w)]  
        print("Column 4 Complete.")
                   
    totlist = [
        list1,list4,list7, list10,
        list2,list5,list8, list11,
        list3,list6,list9, list12
    ]
    totparam = [
        paramlist1, paramlist4, paramlist7, paramlist10,
        paramlist2, paramlist5, paramlist8, paramlist11,
        paramlist3, paramlist6, paramlist9, paramlist12           
    ]
    return totlist, totparam, listt

def DataHist(w = 0, step = 0.001, end = 10, Linearonly = True):
    """
    """
    Linear = Linearonly
    param = [
    # Row 1
    (0. , 0., w, end, step, Linear), (0., np.pi/6, w, end, step, Linear), (0., np.pi/3, w, end, step, Linear), (0., np.pi/2, w, end, step, Linear),
    # Row 2
    (0.5, 0., w, end, step, Linear), (0.5, np.pi/6, w, end, step, Linear), (0.5, np.pi/3, w, end, step, Linear), (0.5, np.pi/2, w, end, step, Linear),
    # Row 3
    (0.9, 0., w, end, step, Linear), (0.9, np.pi/6, w, end, step, Linear), (0.9, np.pi/3, w, end, step, Linear), (0.9, np.pi/2, w, end, step, Linear)
        
    ]
    
    start = time.perf_counter()
    
    with Pool(processes = 6) as pool:
        totlist = pool.map(Rchange, param)
    
    
    end_time = time.perf_counter()
    totaltime = end_time - start
    print(f"Time to Compute was {totaltime:.4f} seconds.")
    
    # totlist = [
    #     list1,list4,list7, list10,
    #     list2,list5,list8, list11,
    #     list3,list6,list9, list12
    # ]
    
    return totlist, param

def MultiPlotProj(t0 = 0.0, a=1.0, w = 0.0, W = 0.0, i = 0.0, e = 0.0, startinga = 1.5):
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
    list, totparam, listt = DataProj(t0, a, w, W, i, e, startinga)
    
    # # Initially Calculates the Velocity to place these into a list
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
        # Takes the first data clump in the list
        # This contains three other data sets
        iterlist = list[j]
        g = 0
        param = totparam[j]
        e, i, w = param[0]
        rtemp, xchange, ychange = Rchange(param = [e, i, w, startinga, 0.5], coords = True)
        vel = Velocity(param = [e, i, w, startinga, 0.5])
        # Iterates through the data clump to access
        # the data set 
        for g in range(len(iterlist)):
            # This contains each data set in the data clump
            initialx, initialy = iterlist[g]
            t = listt[g]
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
            if startinga == 1.25:
                if g == 0:
                    label = "a = 0.75"
                    color = colorlist[g]
                elif g == 1:
                    label = "a = 1.0"
                    color = colorlist[g]
                else:
                    label = "a = 1.25"
                    color = colorlist[g]
            elif startinga == 1.5:
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
                rangelist = np.arange(0.5,startinga+0.5,0.5)
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

def MultiPlotHist(w = 0, step = 0.001, end = 10, Linearonly = True):
    """
    """
    if Linearonly == True:
        colorlist = ["black"]
    else:
        colorlist = ["black", "red"]
    
    totlist, param = DataHist(w = w, step = step, end = end, Linearonly = Linearonly)
    
    
    fig, axs = plt.subplots(3,4, figsize = (9,9), sharex=True,sharey=True,gridspec_kw=dict(hspace=0,wspace=0))               
    if Linearonly == True:
        fig.suptitle("Orbital Projection with Alterations in e, i, and "r"$\omega$ = $0$ (Linear)")
    else:
        fig.suptitle("Orbital Projection with Alterations in e, i, and "r"$\omega$ = $0$ (Linear & Log)")
    # Iterates through each subplot in the 3x4 figure
    
    for j, ax  in enumerate(axs.flatten()):
        iterlistlin, x, y, iterlistlog = totlist[j]
        # Convert list of arrays to a single array, filtering out zeros
        flat_data_lin = np.concatenate([arr[arr != 0] for arr in iterlistlin])
        flat_data_log = np.concatenate([arr[arr != 0] for arr in iterlistlog])
        # Create variables for bin sizes
        nbin = 200
        amin = 0.5
        amax = 21
        # Make logbinsizes for all
        logbinsize = (np.log10(amin)-np.log10(amax))/nbin
        
        # Calculate weights for normalization
        # Check if this is wrong or not, cause of the np.ones_like()
        weights_lin = np.abs(np.ones_like(flat_data_lin) / (len(flat_data_lin) * logbinsize))    
        weights_log = np.abs(np.ones_like(flat_data_log) / (len(flat_data_log) * logbinsize))  
        # Creates the log spaced bins for our data
        # Stupid fix to stupid problems :)
        if j == 0 and Linearonly == True:
            logbins_lin = np.linspace(amin,amax,1000+1)
        else:
            if j == 0:
                logbins_lin = np.linspace(amin,amax,nbin)
                logbins_log = np.geomspace(amin,amax,nbin)
            else:    
                logbins_lin = np.geomspace(amin,amax, nbin)
                logbins_log = np.geomspace(amin,amax,nbin)
        if Linearonly == True:
            
            datahist_lin, bins, patches_lin = ax.hist(
                flat_data_lin, bins=logbins_lin, range=(0.5, end+0.5),
                stacked=True, histtype="barstacked",
                weights=weights_lin
            )
        else:
            datahist_lin, bins, patches_lin = ax.hist(
                flat_data_lin, bins=logbins_lin, range=(0.5, end+0.5),
                stacked=True, histtype="barstacked",
                weights=weights_lin,
            )
            datahist_log, bins, patches_log = ax.hist(
                flat_data_log, bins=logbins_log, range=(0.5, end+0.5),
                stacked=True, histtype="barstacked",
                weights=weights_log
            )
        
        if Linearonly == True:
            for patch in patches_lin:
                patch.set_facecolor("black")
                patch.set_edgecolor("black")
                patch.set_fill(False)
        else:
            for patch in patches_lin:
                patch.set_facecolor("black")
                patch.set_edgecolor("black")
                patch.set_fill(False)
            for patch in patches_log:
                patch.set_facecolor("red")
                patch.set_edgecolor("red")
                patch.set_fill(False)
        ax.set_xlim(0.5,20.5)
        ax.set_ylim(0,10)
        ax.set_xscale("log")
        
    
    plt.text(1.25e-6,25,"e=0")
    plt.text(1.25e-6,15,"e=0.5")
    plt.text(1.25e-6,5,"e=0.9")
    plt.text(4e-5,30.5,"i=0")
    plt.text(1.5e-3,30.5,"i=30")
    plt.text(5.5e-2,30.5,"i=60")
    plt.text(2.75, 30.5,"i=90")
    handles = [patches.Rectangle((0,0),1,1,color = c, ec = "k") for c in colorlist]
    if Linearonly == True:
        labels = ["Linear"]
    else:
        labels = ["Linear", "Log"]
    fig.legend(handles, labels)
    if Linearonly == True:
        plt.savefig('/College Projects/Microlensing Separation/Figures/MultiHist_omega_0_0001_Linear.png')
    else:
        plt.savefig('/College Projects/Microlensing Separation/Figures/MultiHist_omega_0__0001_LinLog.png')
    plt.show()
    return iterlistlin


if __name__ == "__main__":
    # rlist = MultiPlotProj(w = np.pi/4., startinga= 20)
    rtemp = MultiPlotHist(w = 0, step = 0.001, end = 20, Linearonly = False)

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