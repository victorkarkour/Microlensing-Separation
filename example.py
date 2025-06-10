# import numpy as np
# import scipy.optimize as sc

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

#     # Calling sc.newton:
#     # func: Kepler function, where E_val is the variable it's solving for.
#     # x0: The initial guess for E_val (Ei).
#     # fprime: The derivative function.
#     # args: A tuple of additional arguments for Kepler and dKepler_dE,
#     #       which are (Mi, ecc) in this case.
#     Solution[i] = sc.newton(Kepler, Ei, fprime=dKepler_dE, args=(Mi, ecc))

# print("E_initial_guess:", E_initial_guess)
# print("Solution (Eccentric Anomaly):", Solution)

# # You can verify the solution by plugging it back into Kepler's equation
# # The result should be very close to zero
# print("Verification (Kepler(Solution, M, e)):", Kepler(Solution, M, e))
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
import numpy as np

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
j = np.arange(0.5,20.5,0.5)
g = [0.5, 1, 1.5, 2, 2.5, 3, 3.5]
label = [f"a = {string}" for string in j]
print(label)