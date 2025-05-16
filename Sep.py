import astropy.constants as ac
import numpy as np
import scipy.differentiate as diff
import scipy.optimize as sc

def NewtRaf(M, e):
    initial = M+e*np.sin(M) # Bascially starting with M as our initial guess
    Solution = np.zeros_like(M)
    
    for i in range(len(M)):
        Ei = initial[i]
        Mi = M[i]
        
        Solution[i] = sc.newton(Kepler, Ei, fprime = DKepler, args = (Mi,e))
    
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
    print(En)
    return(En-ec*np.sin(En)-Mn)

def DKepler(En,Mn,ec):
    return(1-ec*np.cos(En))

def OrbGeo(t, a=1, w = 0, W = 0, i = 0, e = 0):
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
    M = (2*np.pi/P) * (t-t0)
    Et = NewtRaf(M, e)
    
    Xt = np.cos(Et) - e
    Yt = np.sqrt(1-e**2)*np.sin(Et)
    
    xt = A*Xt + F*Yt
    yt = B*Xt + G*Yt
    
    return(xt, yt)
def pltOrb(t, a=1, w = 0, W = 0, i = 0, e = 0):
    return
    



t = np.linspace(0,3,4)
print(OrbGeo(t,e=0.1))