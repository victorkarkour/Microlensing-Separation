import astropy.constants as ac
import numpy as np
import scipy.differentiate as diff
import scipy.optimize as sc
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
    # print(En)
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
def MultiPlot(t, a=1, w = 0, W = 0, i = 0, e = 0, n = 3):
    
    x1, y1 = OrbGeo(t, e=0)
    x11, y2 = OrbGeo(t, e=0.5)
    x111, y3 = OrbGeo(t, e=0.9)
    
    x2, y11 = OrbGeo(t,e = 0, i = 45, w = np.pi/2)
    x22, y22 = OrbGeo(t,e = 0.5, i = 45, w = np.pi)
    x222, y33 = OrbGeo(t,e = 0.9, i = 45, w = 3*np.pi/2)
    
    x3, y111 = OrbGeo(t, e = 0, i = 90)
    x33, y222 = OrbGeo(t, e=0.5, i = 90)
    x333, y333 = OrbGeo(t, e = 0.9, i = 90)
    
    list = [
        (x1,y1),(x2,y11),(x3,y111),
        (x11,y2),(x22,y22),(x33,y222),
        (x111,y3),(x222,y33),(x333,y333)
    ]
    
    fig, axs = plt.subplots(n,n, figsize = (7,7), sharex=True,sharey=True,gridspec_kw=dict(hspace=0,wspace=0))
    # Circ1 = patches.Circle((0,0),0.5, ec= "b",fill=False)
    # Circ2 = patches.Circle((0,0),1,ec="purple",fill=False)
    # Circ3 = patches.Circle((0,0),1.5,ec="r",fill=False)                
    fig.suptitle("Orbital Projection with Alterations in e, i, and $\omega$")
    for j, ax  in enumerate(axs.flatten()):
        
        initialx, initialy = list[j] 
        
        dataproj = ax.plot(initialx, initialy, color = "g")
        
        Circ1 = patches.Circle((0,0), 0.5, ec= "b", fill=False, linestyle = ":", linewidth = 1)
        Circ2 = patches.Circle((0,0), 1, ec="purple", fill=False, linestyle = ":", linewidth = 1)
        Circ3 = patches.Circle((0,0), 1.5, ec="r", fill=False, linestyle = ":", linewidth = 1)
        
        ax.add_patch(Circ1)
        ax.add_patch(Circ2)
        ax.add_patch(Circ3)
        
        ax.set_xlim(-2,2)
        ax.set_ylim(-2,2)
        if j == 0:
            linelabel = dataproj[0]
        
    fig.legend([Circ1,Circ2,Circ3,linelabel],["a = 0.5", "a = 1", "a = 1.5", "Observed Orbit"], fontsize = "small")
    plt.show()
    
    return None

t = np.linspace(0,2*np.pi,5000)
MultiPlot(t)
# x,y = OrbGeo(t, e=0)
# fig, ax = plt.subplots()
# ax.plot(x,y)
# ax.set_xlabel("x-axis")
# ax.set_ylabel("y-axis")
# ax.set_xlim(-2,2)
# ax.set_ylim(-2,2)
# ax.set_title("Orbital Projection of Exoplanet")
# plt.show()