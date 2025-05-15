import astropy as ac
import numpy as np

def OrbGeo(t, a=1, w = 0, W = 0, i = 0, e = 0):
    """
    Creates the X and Y axis of a planet's orbit
    """

    A = a*(np.cos())