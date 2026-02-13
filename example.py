import numpy as np
from scipy.stats import gamma
import scipy.optimize as sc
import pandas as pd
import matplotlib.pyplot as plt

# # Assume M and e are already defined, for example:
# M = np.array([0.5, 1.0, 1.5]) # Example numpy array for Mean Anomaly
# e = 0.1 # Example eccentricity

# # Kepler's Equation (E - e*sin(E) - M = 0)
# def Kepler(E_val, M_val, e_val):
#     return E_val - e_val * np.sin(E_val) - M_val

# # Derivative of Kepler's Equation with respect to E
# def dKepler_dE(E_val, M_val, e_val):
#     return 1 - e_val * np.cos(E_val)

# # Initial guess for E (Eccentric Anomaly)
# E_initial_guess = M + e * np.sin(M) # A common starting guess for Kepler's equation

# Solution = np.zeros_like(M) # Initialize Solution as a NumPy array of the same shape as M

# for i in range(len(M)):
#     Ei = E_initial_guess[i]
#     Mi = M[i]
#     ecc = e # Assuming e is a scalar, or you'd also need ei = e[i] if e is an array

# #     # Calling sc.newton:
# #     # func: Kepler function, where E_val is the variable it's solving for.
# #     # x0: The initial guess for E_val (Ei).
# #     # fprime: The derivative function.
# #     # args: A tuple of additional arguments for Kepler and dKepler_dE,
# #     #       which are (Mi, ecc) in this case.
# #     Solution[i] = sc.newton(Kepler, Ei, fprime=dKepler_dE, args=(Mi, ecc))

# # print("E_initial_guess:", E_initial_guess)
# # print("Solution (Eccentric Anomaly):", Solution)

# # # You can verify the solution by plugging it back into Kepler's equation
# # # The result should be very close to zero
# # print("Verification (Kepler(Solution, M, e)):", Kepler(Solution, M, e))
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import numpy as np

# # Example data (replace with your actual 'list')
# n = 2
# list = [
#     (np.linspace(-1, 1, 50), np.linspace(-0.5, 0.5, 50)),
#     (np.linspace(-0.8, 0.8, 50), np.linspace(-1.2, 1.2, 50)),
#     (np.linspace(-1.5, 1.5, 50), np.linspace(-0.3, 0.3, 50)),
#     (np.linspace(-0.2, 0.2, 50), np.linspace(-1.8, 1.8, 50)),
# ]

# fig, axs = plt.subplots(n, n, figsize=(7, 7), sharex=True, sharey=True, gridspec_kw=dict(hspace=0, wspace=0))
# fig.suptitle("Orbital Projection with Alterations in e, i, and $\omega$")

# # Initialize a variable to store the line object
# line_handle = None

# for j, ax in enumerate(axs.flatten()):
#     initialx, initialy = list[j]
#     line, = ax.plot(initialx, initialy, color="g")  # Capture the line object
#     if j == 0:
#         line_handle = line  # Store the line object from the first subplot

#     Circ1 = patches.Circle((0, 0), 0.5, ec="b", fill=False, linestyle=":", linewidth=1)
#     Circ2 = patches.Circle((0, 0), 1, ec="purple", fill=False, linestyle=":", linewidth=1)
#     Circ3 = patches.Circle((0, 0), 1.5, ec="r", fill=False, linestyle=":", linewidth=1)

#     ax.add_patch(Circ1)
#     ax.add_patch(Circ2)
#     ax.add_patch(Circ3)

#     ax.set_xlim(-2, 2)
#     ax.set_ylim(-2, 2)

# # Create the legend using the patch objects and the captured line object
# fig.legend([Circ1, Circ2, Circ3, line_handle], ["a = 0.5", "a = 1", "a = 1.5", "Observed Orbit"], loc='upper right')
# plt.show()
nbin = 200
amin = 0.5
amax = 21

# Create gamma prior
alpha = 1.35 # Shape (Alpha)
theta = 1/5.05 # Scale (Beta = 1 / Scale)
x = np.linspace(0,0.98, nbin-1)
gammastep = gamma.pdf(x, a = alpha, scale = theta)


# Make log bins for all
bins = np.geomspace(amin,amax, nbin)
try: 
    filename = f'/College_Projects/Microlensing Separation/Results/UnityHist_eccent_incline_100_0002_Log_alpha_2.csv'
    df_stats = pd.read_csv(filename)
except FileNotFoundError:
    filename = f'/Users/victo/College_Projects/Microlensing Separation/Results/UnityHist_eccent_incline_100_0002_Log_alpha_2.csv'

df_stats = pd.read_csv(filename)
df_stats["cumulative"] = 0
df_stats["cumul_gamma"] = 0
df_stats["cumul_circ"] = np.cumsum(df_stats["circular list"]) / np.abs(sum(df_stats["circular list"]))
df_stats["bins"] = bins[:-1]


hist = df_stats["final list"].to_numpy()
# Make Gamma Calculation
totgammahist = gammastep * hist

df_stats["cumul_gamma"] = np.cumsum(totgammahist) / np.abs(sum(totgammahist))

cumulative = 0
cumul_norm = np.abs(1 / (sum(df_stats["final list"])))
for i in range(len(df_stats["final list"])):
    cumulative = cumulative + df_stats.loc[i, "final list"]
    df_stats.loc[i, "cumulative"] = cumulative
c = np.cumsum(df_stats["final list"])
df_stats["cumul_norm"] = df_stats["cumulative"] * cumul_norm
fig, ax = plt.subplots(figsize = (9,9), sharex=True,sharey=True,gridspec_kw=dict(hspace=0,wspace=0))
fig.suptitle(f"Cumulative Distribution Function \n alpha = 2")
ax.plot(bins[:-1], df_stats["cumul_norm"], ls = "-", c = "k", marker = "o", lw = 2, markersize = 3, label = "")
ax.plot(bins[:-1], df_stats["cumul_gamma"], ls = "-", c = "r", marker = "o", lw = 2, markersize = 3, alpha = 0.5)
ax.plot(bins[:-1], df_stats["cumul_circ"], ls = "-", c = "b", marker = "o", lw = 2, markersize = 3, alpha = 0.5)
ax.legend(["Uniform Dist.","Gamma Dist.","Circular Dist."])
ax.set_xlim(0.5,20)
ax.set_ylim(0,1)
ax.set_xscale("log")
ax.hlines(0.5, xmin = 0, xmax = 200, color = "r")
ax.hlines(0.5+(0.6827/2), xmin = 0, xmax = 200, color = "r")
ax.hlines(0.5-(0.6287/2), xmin = 0, xmax = 200, color = "r")
ax.hlines(0.5+(0.95/2), xmin = 0, xmax = 200, color = "r")
ax.hlines(0.5-(0.95/2), xmin = 0, xmax = 200, color = "r")
ax.set_xlabel(r"Semimajor Axis [$\log{a/R_e}$]")
ax.set_ylabel(r"CDF")
plt.tight_layout()
plt.savefig(f'C:/Users/victo/College_Projects/Microlensing Separation/Figures/CDF_100_Log_alpha_2.png')
plt.show()