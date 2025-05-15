import astropy.constants as ac
import numpy as np
import scipy.differentiate as derivative

def NewtRaf(fE, n = 2):
    initial = fE
    Elist = []
    for i in range(n-1):
        if i == 0:
            Eprime = derivative(fE,Elist[i])
            Elist.append(initial-(fE/Eprime))
        Eprime = derivative(fE,Elist[i])
        Elist.append(Elist[i]-(fE/Eprime))
    return(Elist[-1])

def OrbGeo(t, a=1, w = 0, W = 0, i = 0, e = 0, n=2):
    """
    Creates and calculates the X and Y axis of a planet's orbit
    """
    # Assume time at periastron (closest point to star),
    # t0, as our starting time so t0 = 0
    # Check with Scott to see if we can do this
    t0 = 0
    
    # Initial Equations
    A = a*(np.cos(W)*np.cos(w) - np.sin(W)*np.sin(w)*np.cos(i))
    B = a*(np.sin(W)*np.cos(w) - np.cos(W)*np.sin(w)*np.cos(i))
    F = a*(-np.cos(W)*np.sin(w) - np.sin(W)*np.cos(w)*np.cos(i))
    G = a*(-np.sin(W)*np.sin(w) + np.cos(W)*np.cos(w)*np.cos(i))
    
    # Equation for E (G*M_sun = 1 in Solar Units)
    P = np.sqrt((4*np.pi)/(1)*a**3)
    # Function for E
    fE = (2*np.pi/P) * (t-t0) + e*np.sin(E) - E
    E = NewtRaf(fE,n)
    
    Xt = np.cos(E) - e
    Yt = np.sqrt(1-e**2)*np.sin(E)
    
    