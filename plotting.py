"""
Plotting functions.
- contour
- plot_speed
- plot_one_convergence
@author : Laura Risley, 2022
"""

import matplotlib.pyplot as plt
import numpy as np

from functions import *

def contour(x, y, z, plot_of: str, variable_name: str):
    # 2D contour plot of one variable
    # switch coords from m to km
    x, y = x / 1000, y / 1000
    plt.rcParams['font.family'] = 'Verdana'
    plt.pcolormesh(x, y, z, cmap='viridis', shading='auto')
    plt.xlabel('Longitude ($km$)')
    plt.ylabel('Lattitude ($km$)')
    plt.title(f'{plot_of}')
    if variable_name == 'Elevation':
        units = '$m$'
    elif variable_name == 'Streamfunction' or variable_name == 'Velocity potential':
        units = '$m^2 s^{-1}$'
    else:
        units = '$ms^{-1}$'
    plt.colorbar(label=f'{variable_name} ({units})')
    plt.show()

def plot_speed(u, v, eta_coords, time):
    """
    Plot the speed with arrow presenting the direction.
    Inputs: - u, zonal velocity
            - v, meridional velocity
            - eta_coords, the grid
            - time, what time speed is at
    """
    # Calculate the speed
    w = speed(u, v)
    # extract the coords and convert to km
    x, y = eta_coords['x'] / 1000, eta_coords['y'] / 1000

    # plot the speed
    plt.pcolormesh(x, y, w, cmap='viridis', shading='auto')
    plt.colorbar(label='Speed ($ms^{-1}$)')
    # plot the direction
    # find values of u and v on the eta grid for the direction arrows
    new_u, new_v = interp_zonal(u), interp_merid(v)
    # normalise the arrows
    r = np.power(np.add(np.power(new_u, 2), np.power(new_v, 2)), 0.5)  # could do (u**2+v**2)**0.5
    r = r[::5, ::5]
    # plot the arrows
    plt.quiver(x[::5], y[::5], new_u[::5, ::5] / r, new_v[::5, ::5] / r, color='black')
    plt.xlabel('Longitude ($km$)')
    plt.ylabel('Lattitude ($km$)')
    plt.title(f'Speed ({time})')
    plt.show()

def plot_one_convergence(x_it, plot_of):
    """
    Plot convergence of the cost function/gradient norm at a certain cycle during the assimilation routine.
    Inputs: - x_it, values of the cost function at each iteration
            - plot_of, cost function or gradient
    """
    number_it = len(x_it)
    iterations = np.arange(0, number_it)
    plt.plot(iterations, x_it)
    plt.yscale('log')
    plt.xlabel('Iterations')
    plt.ylabel(f'{plot_of}')
    plt.title(f'Convergence of {plot_of}')
    plt.savefig(f'sweConvergence{plot_of}.png')
    plt.show()