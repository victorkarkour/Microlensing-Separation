import numpy as np
import scipy.optimize as sc

# Assume M and e are already defined, for example:
M = np.array([0.5, 1.0, 1.5]) # Example numpy array for Mean Anomaly
e = 0.1 # Example eccentricity

# Kepler's Equation (E - e*sin(E) - M = 0)
def Kepler(E_val, M_val, e_val):
    return E_val - e_val * np.sin(E_val) - M_val

# Derivative of Kepler's Equation with respect to E
def dKepler_dE(E_val, M_val, e_val):
    return 1 - e_val * np.cos(E_val)

# Initial guess for E (Eccentric Anomaly)
E_initial_guess = M + e * np.sin(M) # A common starting guess for Kepler's equation

Solution = np.zeros_like(M) # Initialize Solution as a NumPy array of the same shape as M

for i in range(len(M)):
    Ei = E_initial_guess[i]
    Mi = M[i]
    ecc = e # Assuming e is a scalar, or you'd also need ei = e[i] if e is an array

    # Calling sc.newton:
    # func: Kepler function, where E_val is the variable it's solving for.
    # x0: The initial guess for E_val (Ei).
    # fprime: The derivative function.
    # args: A tuple of additional arguments for Kepler and dKepler_dE,
    #       which are (Mi, ecc) in this case.
    Solution[i] = sc.newton(Kepler, Ei, fprime=dKepler_dE, args=(Mi, ecc))

print("E_initial_guess:", E_initial_guess)
print("Solution (Eccentric Anomaly):", Solution)

# You can verify the solution by plugging it back into Kepler's equation
# The result should be very close to zero
print("Verification (Kepler(Solution, M, e)):", Kepler(Solution, M, e))