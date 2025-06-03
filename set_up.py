"""
File containing the setup for the problem:
 - Arakawa-C grid
 - Initial array structure based on chosen grid
 - Initial conditions:
     all set to zero
 - Normal boundary conditions:
    set as Dirichlet boundary conditions, u = 0 on the east and west boundaries
                                          v = 0 on north and south boundaries.
    Boundaries set as seen in the NEMO reference manual, p107.
@author : Laura Risley, 2025
"""

import numpy as np

# ------------------------------- Grid -------------------------------
def arakawa_c(ny : int, nx : int):
    # arakawa c grid
    # Y IS ROWS X IS COLUMNS
    eta = np.empty([ny, nx])
    u = np.empty([ny, nx + 1])
    v = np.empty([ny + 1, nx])
    return eta, u, v

# ------------------------------- Array and coordinate structures -------------------------------

def array_structure(parameters : object, dimensions : object, grid: object):
    """
    Set up initial arrays
    """
    # dimensions from dictionary
    nx, ny = int(parameters['L_x'] / dimensions['dx']), int(parameters['L_y'] / dimensions['dy'])
    # arakawa-c grid
    eta_grid, u_grid, v_grid,  = grid(ny, nx)
    return eta_grid, u_grid, v_grid

def array_coords(setup :object, dimensions : object):
    """
    Set up initial coordinates
    """
    # grid length from setup dictionary
    L_x, L_y = setup['L_x'], setup['L_y']
    # dimensions from dictionary
    dx, dy = dimensions['dx'], dimensions['dy']
    # Dictionaries of coordinates with one extra zonal grid point in u and one extra meridional grid point in v
    u_coords = {'x': np.arange(-0.5 * dx, L_x + 0.5 * dx, dx), 'y': np.arange(0, L_y, dy)}
    v_coords = {'x': np.arange(0, L_x, dx), 'y': np.arange(-0.5 * dy, L_y + 0.5 * dy, dy)}
    eta_coords = {'x': np.arange(0, L_x, dx), 'y': np.arange(0, L_y, dy)}
    return eta_coords, u_coords, v_coords

def array_coords_plotting(setup :object, dimensions : object):
    """
    Set up initial coordinates for the plots
    """
    # grid length from setup dictionary
    L_x, L_y = setup['L_x'], setup['L_y']
    # dimensions from dictionary
    dx, dy = dimensions['dx'], dimensions['dy']
    # Dictionaries of coordinates with one extra zonal grid point in u and one extra meridional grid point in v
    u_coords = {'x': np.arange(-0.5 * dx, L_x + 1.5 * dx, dx), 'y': np.arange(0, L_y+dy, dy)}
    v_coords = {'x': np.arange(0, L_x+dx, dx), 'y': np.arange(-0.5 * dy, L_y + 1.5 * dy, dy)}
    eta_coords = {'x': np.arange(0, L_x+dx, dx), 'y': np.arange(0, L_y+dy, dy)}
    return eta_coords, u_coords, v_coords

# ------------------------------- Initial conditions -------------------------------

def initial_eta(eta : np.ndarray):
    # initial condition for eta
    #L_x, L_y = 1e6, 1e6
    #dx, dy = 1e4, 1e4
    #x, y = np.arange(0, L_x, dx), np.arange(0, L_y, dy)
    #X, Y = np.meshgrid(x, y)
    eta[:,:] = 0.0  # eta[:, :] = 0.1*np.exp(-((X-L_x/2)**2/(2*(0.05e+6)**2) + (Y-L_y/2)**2/(2*(0.05e+6)**2))) #initial positive height perturbation #0.0
    return eta


def initial_u(u : np.ndarray):
    # initial condition for u
    u[:, :] = 0.0
    return u


def initial_v(v : np.ndarray):
    # initial condition for u
    v[:, :] = 0.0
    return v

# ------------------------------- Boundary conditions -------------------------------

def zonal_boundary(u : np.ndarray):
    u[:, [0, -1]] = 0
    return u


def meridional_boundary(v : np.ndarray):
    v[[0, -1], :] = 0
    return v