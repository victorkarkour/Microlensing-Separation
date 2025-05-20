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

def InvVelocity(t,x,y):
    changet = np.abs(t[1] - t[0])
    vel = []
    for i in range(len(t)):
        t1 = t[i] - changet
        t2 = t[i] + changet
        x1 = x[i]
        y1 = y[i]
        if i == len(t)-1:
            # THIS SOLUTION FORCES
            # OUR TIME FUNCTION TO ALWAYS BE 
            # AN EVEN INTEGER NUMBER OF PI
            x2 = x[-1]
            y2 = y[-1]
        else:
            x2 = x[i+1]
            y2 = y[i+1]
        
        changedeg = np.sqrt((x1-x2)**2+(y1-y2)**2)
        
        if changedeg == 0:
            vel.append(0)
        else:
            vel.append(1/(changedeg/changet))
    return vel

def Rchange(t,x,y):
    rlist = [] # Time at which it was less than 0.001
    num = [] # Position in list given length of x,y,& t
    r0 = 1 # Einstein Ring Radius
    for i in range(len(t)):
        x1 = x[i]
        y1 = y[i]
        
        r = np.sqrt(x1**2+y1**2)
        
        if np.abs(r-r0) <= 0.001:
            rlist.append(t[i])
            num.append(i)
    return rlist , num

def MultiPlot(t, a=1, w = 0, W = 0, i = 0, e = 0, n = 3):
    
    k=0.5
    list1=[]
    while k <= 1.5:
        x1, y1 = OrbGeo(t,a=k,e=0)
        list1.append((x1,y1))
        k+=0.5
    k=0.5
    list2=[]
    while k <= 1.5:
        x1, y1 = OrbGeo(t,a=k,e=0.5)
        list2.append((x1,y1))
        k+=0.5
    k=0.5
    list3=[]
    while k <= 1.5:
        x1, y1 = OrbGeo(t,a=k,e=0.9)
        list3.append((x1,y1))
        k+=0.5
    
    # x1, y1 = OrbGeo(t,e = 0,)
    # x11, y2 = OrbGeo(t,e = 0.5)
    # x111, y3 = OrbGeo(t,e = 0.9)
    
    k=0.5
    list4=[]
    while k <= 1.5:
        x1, y1 = OrbGeo(t,a=k, e = 0, i = np.pi/4, w = np.pi/2)
        list4.append((x1,y1))
        k+=0.5
    k=0.5
    list5=[]
    while k <= 1.5:
        x1, y1 = OrbGeo(t,a=k, e = 0.5, i = np.pi/4, w = np.pi)
        list5.append((x1,y1))
        k+=0.5
    k=0.5
    list6=[]
    while k <= 1.5:
        x1, y1 = OrbGeo(t, a = k, e = 0.9, i = np.pi/4, w = 3*np.pi/2)
        list6.append((x1,y1))
        k+=0.5
    
    # x2, y11 = OrbGeo(t,e = 0, i = np.pi/4, w = np.pi/2)
    # x22, y22 = OrbGeo(t,e = 0.5, i = np.pi/4, w = np.pi)
    # x222, y33 = OrbGeo(t,e = 0.9, i = np.pi/4, w = 3*np.pi/2)
    
    k=0.5
    list7=[]
    while k <= 1.5:
        x1, y1 = OrbGeo(t, a = k, e = 0, i = np.pi/2)
        list7.append((x1,y1))
        k+=0.5
    k=0.5
    list8=[]
    while k <= 1.5:
        x1, y1 = OrbGeo(t, a = k, e = 0.5, i = np.pi/2)
        list8.append((x1,y1))
        k+=0.5
    k=0.5
    list9=[]
    while k <= 1.5:
        x1, y1 = OrbGeo(t, a = k, e = 0.9, i = np.pi/2)
        list9.append((x1,y1))
        k+=0.5
    
    # x3, y111 = OrbGeo(t, e = 0, i = np.pi/2)
    # x33, y222 = OrbGeo(t, e=0.5, i = np.pi/2)
    # x333, y333 = OrbGeo(t, e = 0.9, i = np.pi/2)
    
    list = [
        list1,list4,list7,
        list2,list5,list8,
        list3,list6,list9
    ]
    
    fig, axs = plt.subplots(n,n, figsize = (7,7), sharex=True,sharey=True,gridspec_kw=dict(hspace=0,wspace=0))               
    fig.suptitle("Orbital Projection with Alterations in e, i, and "r"$\omega$")
    for j, ax  in enumerate(axs.flatten()):
        
        iterlist = list[j]
        g = 0
        for g in range(len(iterlist)):
            initialx, initialy = iterlist[g]
            vel = InvVelocity(t,initialx,initialy) 
            if g == 0:
                color = "g"
                label = "a = 0.5"
            elif g == 1:
                color = "r"
                label = "a = 1.0"
            else:
                color = "b"
                label = "a = 1.5"    
            dataproj = ax.scatter(initialx, initialy,s = vel, color = color, label = label)
            ax.grid(True,color = "grey", linestyle="--", linewidth="0.25")
        if j == 0:
            handles, labels = ax.get_legend_handles_labels()
        # Circ1 = patches.Circle((0,0), 0.5, ec= "b", fill=False, linestyle = ":", linewidth = 1)
        # Circ2 = patches.Circle((0,0), 1, ec="purple", fill=False, linestyle = ":", linewidth = 1)
        # Circ3 = patches.Circle((0,0), 1.5, ec="r", fill=False, linestyle = ":", linewidth = 1)
        
        # ax.add_patch(Circ1)
        # ax.add_patch(Circ2)
        # ax.add_patch(Circ3)
        
        ax.set_xlim(-2,2)
        ax.set_ylim(-2,2)
    # fig.legend([Circ1,Circ2,Circ3,linelabel],["a = 0.5", "a = 1", "a = 1.5", "Observed Orbit"], fontsize = "small")
    fig.legend(handles,labels,fontsize="small")
    plt.text(-11.5,7.90,"e=0")
    plt.text(-11.5,3.90,"e=0.5")
    plt.text(-11.5,-0.1,"e=0.9")
    plt.text(-8,10.5,"i=0")
    plt.text(-4,10.5,"i=45")
    plt.text(-0.5,10.5,"i=90")
    plt.show()
    
    return n

t = np.linspace(0,4*np.pi,400)
MultiPlot(t)
# x,y = OrbGeo(t, e=0)
# print(InvVelocity(t,x,y)[-1])
# fig, ax = plt.subplots()
# ax.plot(x,y)
# ax.set_xlabel("x-axis")
# ax.set_ylabel("y-axis")
# ax.set_xlim(-2,2)
# ax.set_ylim(-2,2)
# ax.set_title("Orbital Projection of Exoplanet")
# plt.show()