import numpy as np
from scipy.stats import gamma
import scipy.optimize as sc
import pandas as pd
import matplotlib.pyplot as plt
import random

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
# Slices estep into integer parts for parallelization
# Distribute any remainder by giving one extra element to the first `rem` slices.
# This handles cases where numestep < numdiv as well.

# def stepdata(alpha, xmin, xmax, nsamples):
        
#         """
#         """
#         step = np.linspace(0,1,nsamples+2)[1:-1]
        
#         if alpha == 1.:
#             return xmin * (xmax/xmin) ** step
#         else:
#             # normal: alpha = 0 (Linear), alpha = 1 (Log), alpha = -1 (Power) 
#             # ALPHAS ARE SWAPPED FOR This
#             exp = (1. - alpha)
#             return (step * (xmax**exp - xmin**exp) + xmin**exp) ** (1 / exp)
# stepping = stepdata(-1, 0.5, 20, 10000)
# stepthrough = np.logspace(np.log10(0.5), np.log10(20), num=10000)
# print("Log Distribution: ", stepthrough)
# print("Power Law Distribution: ",stepping)


# stepping_log = stepdata(1, 0.5, 20, 10000)
# print("Log Distribution: ", len(stepping_log))
# print("Power Law Distribution: ",len(stepping))
# points = np.linspace(0, 20, 10000)
# fig, axs = plt.subplots()
# axs.plot(points,stepping, label="Power Law")
# axs.plot(points, points, label="Linear")
# axs.plot(points,stepthrough, label="Logarithmic")
# axs.legend()
# # axs.set_xscale("log")
# plt.show()
# points_ecc = np.linspace(0, 0.99, 12)
# print(points_ecc)

rand_float = random.uniform(0, 1)
print(np.arccos(rand_float))

rng = np.random.default_rng()
random = rng.uniform(low = 0, high = np.pi/2, size = 1)
print(random)
print(np.pi/2)