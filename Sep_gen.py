import numpy as np
import scipy.optimize as sc
import gc
import time

class Sep_gen:
    def __init__(self):
        self.param = []
    @staticmethod
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
    
        Solution = sc.newton(Sep_gen.Kepler, initial, fprime=Sep_gen.DKepler, args=(M,e))
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
    @staticmethod
    def Kepler(En, Mn, ec):
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
    @staticmethod
    def DKepler(En, Mn, ec):
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

    # def OrbGeo(t0=0, a=1, w = 0, W = 0, i = 0, e = 0):
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
    @staticmethod
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
        
        Et = Sep_gen.NewtRaf(phi,e)
        
        # Function of X and Y according to E
        Xt = np.cos(Et) - e
        Yt = np.sqrt(1-e**2)*np.sin(Et)
        
        # Actual x any y functions of t
        xt = A * Xt + F * Yt
        yt = B * Xt + G * Yt
        
        return (xt, yt, phi)
    @staticmethod
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
            x1, y1, t1 = Sep_gen.OrbGeoAlt(-changet, a = val, e = e, w = w, i = i)
            x2, y2, t2 = Sep_gen.OrbGeoAlt(changet, a = val, e = e, w = w, i = i)
        
            # Change of degrees function
            # because of array arithmetic, we don't need to use for loops
            changedeg = np.sqrt(np.add(np.power(np.subtract(x2,x1),2),np.power(np.subtract(y2,y1),2)))
            
            # Velocity function
            vel.append(np.divide(changedeg,(2.0*changet)))
        # print("Last Velocity output: ", vel[-1])
        # print("Minimum Velocity: ", np.min(vel))
        return vel
    @staticmethod
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
    @staticmethod
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
            Linear = "Linear"
        else:
            e, i, w, end, step, Linear, inclination = param
            # if inclination:
            #     istep = i
        
        # Only steps through Linear portion of points
        if Linear == "Linear":
            # Linear Portion
            # Goes through each value of a in the stepthrough
            if isinstance(i, np.ndarray):
                stepthrough = Sep_gen.stepdata(0, 0.5, end, 10000)
                # Goes through each value of a in the stepthrough
                for aval in stepthrough:
                    for ival in i:
                        if isinstance(e, np.ndarray):
                            for eval in e:    
                                x, y, t = Sep_gen.OrbGeoAlt(a = aval, e = eval, i = ival ,w = w)
                                r = np.sqrt(x**2+y**2)
                                # Whereever there is this value, it finds the indices of each point in the list
                                conlin = np.where(np.abs(r-r0)<=0.01)
                    
                                if coords == False and inclination:
                                    # Has brackets with 0 b/c conlin is an array of length 1, to get to values u must flatten
                                    if aval in totlindict:
                                        totlindict[aval] += len(conlin[0])
                                    else:
                                        totlindict[aval] = len(conlin[0])
                                    if eval == 0:
                                        if aval in totevaldict:
                                            totevaldict[aval] += len(conlin[0])
                                        else:
                                            totevaldict[aval] = len(conlin[0])
            else:
                if coords:
                    totlindict = []
                stepthrough = np.arange(0.5, end + step, step)
                for aval in stepthrough:
                    x, y, t = Sep_gen.OrbGeoAlt(a = aval, e = e, i = i ,w = w)
                    r = np.sqrt(x**2+y**2)
                    # Whereever there is this value, it finds the indices of each point in the list
                    conlin = np.where(np.abs(r-r0)<=0.01)
                    # Same thing as coords == False but has coord lists for Multiplot
                    if aval in totlindict:
                        totlindict[aval] += len(conlin[0])
                    else:
                        totlindict[aval] = len(conlin[0])
                    if coords:
                        totlindict.append(np.where(np.abs(r-r0)<=0.01 , aval, 0))
                        xlist.append(np.where(np.abs(r-r0)<=0.01 , x, None))
                        ylist.append(np.where(np.abs(r-r0)<=0.01 , y, None))
            return totlindict, xlist, ylist, totlogdict, totevaldict
        elif Linear == "Log":
            if inclination == False:
                # Log Portion
                # steps = np.linspace(0, 10000,10000)
                # loga = np.log10(0.5) + steps/10000 * (np.log10(end)-np.log10(0.5))
                # stepthrough = 10**loga 
                stepthrough = Sep_gen.stepdata(1, 0.5, end, 10000)
                for val in stepthrough:
                    x, y, t = Sep_gen.OrbGeoAlt(a = val, e = e, i = i ,w = w)
        
                    r = np.sqrt(x**2+y**2)
            
                    conlog = np.where(np.abs(r-r0)<=0.01)
                    totlogdict[val] = len(conlog[0])
                gc.collect()
                # Linear Portion
                # stepthrough = np.arange(0.5, end + step, step)
                stepthrough = Sep_gen.stepdata(0, 0.5, end, 10000)
                for val in stepthrough:
                    x, y, t = Sep_gen.OrbGeoAlt(a = val, e = e, i = i ,w = w)
        
                    r = np.sqrt(x**2+y**2)
                
                    conlin = np.where(np.abs(r-r0)<=0.01)
                    totlindict[val] = len(conlin[0])
                # Power Law Portion
                stepthrough = Sep_gen.stepdata(2, 0.5, end, 10000)
                for val in stepthrough:
                    x, y, t = Sep_gen.OrbGeoAlt(a = val, e = e, i = i ,w = w)
        
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
                            if isinstance(e, np.ndarray):
                                for eval in e:
                                    x, y, t = Sep_gen.OrbGeoAlt(a = aval, e = eval, i = ival ,w = w)
                                    r = np.sqrt(x**2+y**2)
                            
                                    conlog = np.where(np.abs(r-r0)<=0.01)
                                    if aval in totlogdict:
                                        totlogdict[aval] += len(conlog[0])
                                    else:
                                        totlogdict[aval] = len(conlog[0])
                                    if eval == 0:
                                        if aval in totevaldict:
                                            totevaldict[aval] += len(conlin[0])
                                        else:
                                            totevaldict[aval] = len(conlin[0])
                            else:
                                x, y, t = Sep_gen.OrbGeoAlt(a = aval, e = e, i = ival ,w = w)
                                r = np.sqrt(x**2+y**2)
                            
                                conlog = np.where(np.abs(r-r0)<=0.01)
                                if aval in totlogdict:
                                    totlogdict[aval] += len(conlog[0])
                                else:
                                    totlogdict[aval] = len(conlog[0])
                return totlindict, xlist, ylist, totlogdict, totevaldict
        elif Linear == "Linear / a":
        # Linear / a Portion
            stepthrough = Sep_gen.stepdata(0.5, 0.5, end, 10000)
            for aval in stepthrough:
                for ival in i:
                    if isinstance(e, np.ndarray):
                        for eval in e:
                            x, y, t = Sep_gen.OrbGeoAlt(a = aval, e = eval, i = ival ,w = w)
                            r = np.sqrt(x**2+y**2)
                            # Whereever there is this value, it finds the indices of each point in the list
                            conlin = np.where(np.abs(r-r0)<=0.01)
                            # Has brackets with 0 b/c conlin is an array of length 1, to get to values u must flatten
                        
                            # REMINDER: THIS IS FOR LINEAR / A, I JUST REMOVED THE totlinsemidict FROM THIS FOR EASIER INTERPRETATION
                            if aval in totlindict:
                                totlindict[aval] = round(len(conlin[0]) / aval)
                            else:
                                totlindict[aval] = round(len(conlin[0]) / aval)
                            if eval == 0:
                                if aval in totevaldict:
                                    totevaldict[aval] += len(conlin[0])
                                else:
                                    totevaldict[aval] = len(conlin[0])
                    else:
                        x, y, t = Sep_gen.OrbGeoAlt(a = aval, e = eval, i = ival ,w = w)
                        r = np.sqrt(x**2+y**2)
                        # Whereever there is this value, it finds the indices of each point in the list
                        conlin = np.where(np.abs(r-r0)<=0.01)
                        # Has brackets with 0 b/c conlin is an array of length 1, to get to values u must flatten
                        
                        # REMINDER: THIS IS FOR LINEAR / A, I JUST REMOVED THE totlinsemidict FROM THIS FOR EASIER INTERPRETATION
                        if aval in totlindict:
                            totlindict[aval] = round(len(conlin[0]) / aval)
                        else:
                            totlindict[aval] = round(len(conlin[0]) / aval)
            return totlindict, xlist, ylist, totlogdict, totevaldict
    @staticmethod
    def stepdata(alpha, xmin, xmax, nsamples):
        
        """
        """
        step = np.linspace(0,1,nsamples+2)[1:-1]
        
        if alpha == 1.:
            return xmin * (xmax/xmin) ** step
        else:
            exp = 1. - alpha
            return (step * (xmax**exp - xmin**exp) + xmin**exp) ** (1 / exp)
    @staticmethod
    def HistGen(param):
        """
        """
        step, end, inclination, which, estep_outer, inum, wnum, _ = param
        
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
                # steptotlist, param = Sep_plot.DataHist(w = k, step = step, end = end, which = which, inclination = inclination, istep = istep, estep = estep)
                param = [estep, istep, k, end, step, which, inclination]
            elif inclination:
                # steptotlist, param = Sep_plot.DataHist(w = k, step = step, end = end, which = which, inclination = inclination, istep = istep)
                param = [[], istep, k, end, step, which, inclination]
            
            start = time.perf_counter()
            # Multi Processing
            if inclination == True and len(estep) != 0:
                steptotlist = Sep_gen.Rchange(param = param)
            
            end_time = time.perf_counter()
            totaltime = end_time - start
            print(f"Time to Compute was {totaltime:.4f} seconds.")    
            
            
            # Once complete, takes the data through each set
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
        
        return tothistlist, evalhistlist
        
# param = [0, 1, np.pi/2, 0, 0, 0.1]
# x = Sep_gen()
# print(x.OrbGeoAlt())