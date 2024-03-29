#!/usr/local/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def phillips(Kx, Ky, windDir, windSpeed, A, g):
    """
    Calculate the Phillips spectrum for given wave number components Kx and Ky,
    wind direction, wind speed, Phillips constant A, and gravitational acceleration g.
    
    Parameters:
    - Kx, Ky: Components of the wave vector.
    - windDir: Wind direction vector (should be a 2-element array).
    - windSpeed: Scalar value of the wind speed.
    - A: Phillips constant.
    - g: Gravitational acceleration.
    
    Returns:
    - P: Phillips spectrum value.
    """
    K_sq = Kx**2 + Ky**2
    L = windSpeed**2 / g
    k_norm = np.sqrt(K_sq)
    WK = (Kx/k_norm) * windDir[0] + (Ky/k_norm) * windDir[1]
    P = A / K_sq**2 * np.exp(-1.0 / (K_sq * L**2)) * WK**2
    P[(K_sq == 0) | (WK < 0)] = 0  # Apply conditions to set parts of P to 0
    
    return P


def signGrid(n):
    """
    Create an n x n grid where the sign alternates in a checkerboard pattern.
    
    Parameters:
    - n: The size of the grid.
    
    Returns:
    - sgn: An n x n numpy array with elements alternating between 1 and -1.
    """
    x, y = np.meshgrid(range(1, n+1), range(1, n+1), indexing='ij')
    sgn = np.ones((n, n))
    sgn[(x + y) % 2 == 0] = -1
    return sgn


# Example usage:
n = 5
sgn_matrix = signGrid(n)
print(sgn_matrix)


def init_surf(h, param):
    """
    Initialize and update the surface graphic with a new set of wave parameters.
    
    Parameters:
    - h: A matplotlib figure and axes handle.
    - param: A dictionary containing wave parameters.
    """
    # Define the grid in X-Y space
    x = np.linspace(param['xLim'][0], param['xLim'][1], param['meshsize'])
    y = np.linspace(param['yLim'][0], param['yLim'][1], param['meshsize'])
    X, Y = np.meshgrid(x, y)

    # Initialize wave coefficients
    H0, W, Grid_Sign = initialize_wave(param)  # You'll need to define this function

    # Calculate wave at t0
    t0 = 0
    Z = calc_wave(H0, W, t0, Grid_Sign)  # You'll need to define this function

    # Display the initial wave surface
    h['surf'].remove()
    h['surf'] = h['ax'].plot_surface(X, Y, Z, cmap=cm.coolwarm)
    h['ax'].set_xlim(param['xLim'])
    h['ax'].set_ylim(param['yLim'])
    h['ax'].set_zlim(param['zLim'])

    # Save coeffs for future use (using the figure's `__dict__` for storage)
    h['fig'].__dict__['H0'] = H0
    h['fig'].__dict__['W'] = W
    h['fig'].__dict__['Grid_Sign'] = Grid_Sign
    h['fig'].__dict__['param'] = param

    # Reset the wave animation parameters (you'll need to define `animate_wave`)
    animate_wave(h, True)

    # Display side patch if option checked (you'll need to implement this logic)
    # This part would require a GUI component like a checkbox, which isn't directly
    # handled in this example.
    # init_SidePatch(h['fig'], displaySidePatch)  # Assuming you have this function

# Example setup
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
h = {'fig': fig, 'ax': ax, 'surf': None}

param = {
    'xLim': [-5, 5],
    'yLim': [-5, 5],
    'zLim': [-1, 1],
    'meshsize': 100
}

# Assuming `initialize_wave` and `calc_wave` are defined,
# you could then initialize the surface with something like:
# init_surf(h, param)
