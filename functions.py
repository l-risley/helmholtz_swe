"""
Functions needed in this project.
@author : Laura Risley, 2025
"""

from scipy.interpolate import RegularGridInterpolator, interp2d
from set_up import *
from parameters import *
import numpy as np

# ------------------------------- Interpolation -------------------------------

def interp_zonal(z):
    # interpolate in the zonal direction
    # interpolate u to eta or eta to u
    return 0.5 * (z[:, :-1] + z[:, 1:])


def interp_merid(z):
    # interpolate in the meridional direction
    # interpolate v to eta or eta to v
    return 0.5 * (z[:-1, :] + z[1:, :])

# ------------------------------- Speed -------------------------------

def speed(u,v):
    """
    Calculate the current speed on the elevation grid.
    Inputs: - u, zonal velocity
            - v, meridional velocity
    Output: - w, speed
    """
    # find u^2
    u_2 = np.square(u)
    # find v^2
    v_2 = np.square(v)
    # interpolate both to eta grid
    u_2, v_2 = interp_zonal(u_2), interp_merid(v_2)
    #calculate the speed: w = sqrt(u^2 + v^2)
    w = np.sqrt(u_2 + v_2)
    return w

# ------------------------------- Wind and coriolic functions -------------------------------

def wind_stress(tau_0, y, L):
    """
    Calculate the wind stress vector
    Input: tau_0, constant of wind stress
           y, spatial axis
           L, computation domain
    Output: tau_x, tau_y, wind stress vectors
    """
    tau_x = tau_0 * -np.cos(np.pi * y / L)
    tau_y = tau_0 * 0
    return tau_x, tau_y

def gyre_wind_intensity(t_hrs, t_max, t_min, gtau, gtau_s):
    input = (t_hrs - t_max) / (t_min - t_max) * np.pi
    return gtau - gtau_s * np.cos(input)

def gyre_wind_forcing(tau_t, y, L, vel : str):
    """
    Calculate the wind stress vector using the seasonal forcing in the NEMO gyre configuration.
    Input: tau_t, wind intensity
           y,   spatial axis
           L,   computation domain
           vel, zonal or meridional
    Output: wind stress vector
        """
    if vel == 'u':
        return - tau_t * np.cos(2*np.pi * y / L)
    elif vel == 'v':
        return 0

def gyre_wind_stress_Tpoint(rho_a, drag, u, v):
    # wind stress constant
    wind_const = 1 / (rho_a * drag)
    tau_u, tau_v = interp_zonal(u), interp_merid(v)
    wind = np.sqrt(np.square(tau_u) + np.square(tau_v))
    return np.sqrt(wind_const * wind)

def cor(f_0, beta, y, vel):
    """
    Coriolis term in the forward-backward numerical scheme.
    Inputs: - f_0, beta, approximation of coriolis parameter on a beta plane
            - y, y coordinates in a meshgrid
            - vel, either zonal or meridional velocity
    Outputs: coriolis term
    """
    # when vel = v, interpolating to the u grid
    # when vel = u, interpolating to the v grid
    frac = vel[:-1, :-1] + vel[1:, :-1] + vel[:-1, 1:] + vel[1:, 1:]
    cor = f_0 + beta * y
    return frac * cor * 0.25
# ------------------------------- Derivatives -------------------------------

def dzdx(z, dx: int):
    # Take the difference between u at index j+1 and j
    # size will be (ny, nx) on the eta_grid if dudx
    # size will be (ny+1, nx-1) not on any grid if dvdx
    # eta is interp to the u_grid
    dz = np.diff(z, axis=1)
    return dz / dx

def dzdy(z, dy: int):
    # Take the difference between z at index j+1 and j
    # size will be (ny, nx) on the eta_grid id dvdy
    # size will be (ny-1, nx+1) not on any grid if dudy
    # eta is interp to the v_grid
    dz = np.diff(z, axis=0)
    return dz / dy

# ------------------------------- Filters -------------------------------

def shapiro_filter_2d_dirichlet(input_array):
    """
    Apply the 2nd order shapiro filter, where both the input and output have Dirichlet boundary conditions dirichlet boundary conditions.
    This function assumes that the input array is on the boundary of the whole grid.
    Input:  input_array
    Output: filtered array
    """
    ## apply 2nd order shapiro filtering
    # final array with dirichlet BCs
    x_new = np.zeros_like(input_array)
    # current value
    x_current = 4 * input_array[1:-1, 1:-1]
    # adjacent values
    x_adj = 2 * (input_array[:-2, 1:-1] + input_array[2:, 1:-1] + input_array[1:-1, :-2] + input_array[1:-1, 2:])
    # surrounding values
    x_sr = input_array[:-2, :-2] + input_array[2:, :-2] + input_array[2:, 2:] + input_array[:-2, 2:]
    x_filtered = 1 / 16 * (x_current + x_adj + x_sr)
    print(x_filtered)
    x_new[1:-1, 1:-1] = x_filtered
    return x_new

def shapiro_filter_2d_neumann(input_array):
    """
    Apply the 2nd order shapiro fileter, where both the input and output have neumann boundary conditions.
    Assumes the input array is located at the T-point on an arakawa-C grid.
    Input:  input_array
    Output: filtered array
    """
    # pad the matrix with the edge values to imply neumann boundary conditions
    x = np.pad(input_array, pad_width=1, mode='edge')
    ## apply 2nd order shapiro filtering
    # current value
    x_current = 4 * x[1:-1, 1:-1]
    # adjacent values
    x_adj = 2 * (x[:-2, 1:-1] + x[2:, 1:-1] + x[1:-1, :-2] + x[1:-1, 2:])
    # surrounding values
    x_sr = x[:-2, :-2] + x[2:, :-2] + x[2:, 2:] + x[:-2, 2:]
    x_filtered = 1 / 16 * (x_current + x_adj + x_sr)
    print(x_filtered)
    return x_filtered

def shapiro_filter_2d_u(input_array):
    """
    Apply the 2nd order shapiro filter, where both the input and output have Dirichlet boundary conditions dirichlet boundary conditions.
    This function assumes that the input array is located at a u-point.
    Input:  input_array
    Output: filtered array
    """
    ## apply 2nd order shapiro filtering
    x_new = np.zeros_like(input_array)
    x_new[1:-1, 1:-1] = 1 / 16 * (4 * input_array[1:-1, 1:-1] +
                                  2 * (input_array[:-2, 1:-1] + input_array[2:, 1:-1] + input_array[1:-1, :-2] +
                                       input_array[1:-1, 2:])
                                    + input_array[:-2, :-2] + input_array[2:, :-2] + input_array[2:, 2:] + input_array[:-2, 2:])
    x_new[0, 1:-1]  = 1 / 12 * (4 * input_array[0, 1:-1] +
                                2 * (input_array[1, 1:-1] + input_array[0, :-2] + input_array[0, 2:])
                                  + input_array[1, :-2] + input_array[1, 2:])
    x_new[-1, 1:-1] = 1 / 12 * (4 * input_array[-1, 1:-1] +
                                2 * (input_array[-2, 1:-1] + input_array[-1, :-2] + input_array[-1, 2:])
                                  + input_array[-2, :-2] + input_array[-2, 2:])
    return x_new

def shapiro_filter_2d_v(input_array):
    """
    Apply the 2nd order shapiro filter, where both the input and output have Dirichlet boundary conditions dirichlet boundary conditions.
    This function assumes that the inout array is located at a v-point.
    Input:  input_array
    Output: filtered array
    """
    ## apply 2nd order shapiro filtering
    x_new = np.zeros_like(input_array)
    x_new[1:-1, 1:-1] = 1 / 16 * (4 * input_array[1:-1, 1:-1] +
                                  2 * (input_array[:-2, 1:-1] + input_array[2:, 1:-1] + input_array[1:-1, :-2] +
                                       input_array[1:-1, 2:])
                                    + input_array[:-2, :-2] + input_array[2:, :-2] + input_array[2:, 2:] + input_array[:-2, 2:])
    x_new[1:-1, 0]  = 1 / 12 * (4 * input_array[1:-1, 0] +
                                2 * (input_array[1:-1, 1] + input_array[:-2, 0] + input_array[2:, 0])
                                  + input_array[:-2, 1] + input_array[2:, 1])
    x_new[1:-1, -1] = 1 / 12 * (4 * input_array[1:-1, -1] +
                                2 * (input_array[1:-1, -2] + input_array[:-2, -1] + input_array[2:, -1])
                                  + input_array[:-2, -2] + input_array[2:, -2])
    return x_new

def shapiro_filter_2d_u_adj(input_array):
    """
    Adjoint to shapiro_filter_2d_u.
    """
    u = np.zeros_like(input_array)

    u[-2, 2:] += 1 / 16 * input_array[-1, 1:-1]
    u[-2, :-2] += 1 / 16 * input_array[-1, 1:-1]
    u[-1, 2:] += 2 / 16 * input_array[-1, 1:-1]
    u[-1, :-2] += 2 / 16 * input_array[-1, 1:-1]
    u[-2, 1:-1] += 2 / 16 * input_array[-1, 1:-1]
    u[-1, 1:-1] += 4 / 16 * input_array[-1, 1:-1]

    u[1, 2:] += 1 / 16 * input_array[0, 1:-1]
    u[1, :-2] += 1 / 16 * input_array[0, 1:-1]
    u[0, 2:] += 2 / 16 * input_array[0, 1:-1]
    u[0, :-2] += 2 / 16 * input_array[0, 1:-1]
    u[1, 1:-1] += 2 / 16 * input_array[0, 1:-1]
    u[0, 1:-1] += 4 / 16 * input_array[0, 1:-1]

    u[:-2, 2:] += 1 / 16 * input_array[1:-1, 1:-1]
    u[2:, 2:] += 1 / 16 * input_array[1:-1, 1:-1]
    u[2:, :-2] += 1 / 16 * input_array[1:-1, 1:-1]
    u[:-2, :-2] += 1 / 16 * input_array[1:-1, 1:-1]
    u[1:-1, 2:] += 2 / 16 * input_array[1:-1, 1:-1]
    u[1:-1, :-2] += 2 / 16 * input_array[1:-1, 1:-1]
    u[2:, 1:-1] += 2 / 16 * input_array[1:-1, 1:-1]
    u[:-2, 1:-1] += 2 / 16 * input_array[1:-1, 1:-1]
    u[1:-1, 1:-1] += 4 / 16 * input_array[1:-1, 1:-1]

    return u

def shapiro_filter_2d_v_adj(input_array):
    """
    Adjoint to shapiro_filter_2d_v
    """
    v = np.zeros_like(input_array)

    v[2:, -2] += 1 / 16 * input_array[1:-1, -1]
    v[:-2, -2] += 1 / 16 * input_array[1:-1, -1]
    v[2:, -1] += 2 / 16 * input_array[1:-1, -1]
    v[:-2, -1] += 2 / 16 * input_array[1:-1, -1]
    v[1:-1, -2] += 2 / 16 * input_array[1:-1, -1]
    v[1:-1, -1] += 4 / 16 * input_array[1:-1, -1]

    v[2:, 1] += 1 / 16 * input_array[1:-1, 0]
    v[:-2, 1] += 1 / 16 * input_array[1:-1, 0]
    v[2:, 0] += 2 / 16 * input_array[1:-1, 0]
    v[:-2, 0] += 2 / 16 * input_array[1:-1, 0]
    v[1:-1, 1] += 2 / 16 * input_array[1:-1, 0]
    v[1:-1, 0] += 4 / 16 * input_array[1:-1, 0]

    v[:-2, 2:] += 1 / 16 * input_array[1:-1, 1:-1]
    v[2:, 2:] += 1 / 16 * input_array[1:-1, 1:-1]
    v[2:, :-2] += 1 / 16 * input_array[1:-1, 1:-1]
    v[:-2, :-2] += 1 / 16 * input_array[1:-1, 1:-1]
    v[1:-1, 2:] += 2 / 16 * input_array[1:-1, 1:-1]
    v[1:-1, :-2] += 2 / 16 * input_array[1:-1, 1:-1]
    v[2:, 1:-1] += 2 / 16 * input_array[1:-1, 1:-1]
    v[:-2, 1:-1] += 2 / 16 * input_array[1:-1, 1:-1]
    v[1:-1, 1:-1] += 4 / 16 * input_array[1:-1, 1:-1]

    input_array = 0
    return v

# ------------------------------- General -------------------------------

def split_x(x, ny_cv, nx_cv):
    """
    Split vector x into two matrices for stream function and velocity potential
    Inputs:  x, vector containing sf (streamfunction) and vp (velocity potential) (2*(ny+2)*(nx+2))
             ny, nx, number of eta points on the grid (ny, nx)
    Outputs: sf, streamfunction (ny+2, nx+2)
             vp, velocity potential (ny+2, nx+2)
    """
    # split x into two equal arrays for sf and vp
    sf, vp = np.split(x, 2)
    # reshape sf and vp into matrices
    sf, vp = np.reshape(sf, (ny_cv, nx_cv)), np.reshape(vp, (ny_cv, nx_cv))
    return sf, vp

def split_b(b, ny, nx):
    """
    Split vector b into two matrices for zonal and meridional velocity
    Inputs:  b, vector containing the zonal and meridional velocity ((ny)*(nx+1) + (ny+1)*(nx))
             ny, nx, number of eta points on the grid (ny, nx)
    Outputs: u, zonal velocity (ny, nx+1)
             v, meridional velocity  (ny+1, nx)
    """
    # split b into two arrays for u and v
    u, v = np.split(b, [ny * (nx + 1)])
    # reshape sf and vp into matrices
    u, v = np.reshape(u, (ny, nx + 1)), np.reshape(v, (ny + 1, nx))
    return u, v

# ------------------------------- SISL -------------------------------

def interp_function(grid, data, new_grid, function: str):
    """
    LINEAR Interpolation function
    Interp2d = will only work for array inputs
    Regular grid will only work for a linear method
    """
    # new grid is two arrays of coordinates for interp2d
    # new grid is two matrices of coordinates for regular grid interp

    row, column = grid
    row_new, col_new = new_grid

    if function == 'interp2d':
        # used to find departure points as the grids are different sizes
        # only allows for regular grid inputs
        # interp2d inputs column coords, then row coords
        # but array indexing is row then columns
        interp_fn = interp2d(column, row, data, kind='linear')
        new_data = interp_fn(col_new, row_new)

    elif function == 'Regular':
        # used to interpolate array to dp points, grids are the same size
        # RegularGridInterpolator inputs row coords, then column coords
        # allows for the input into the fn to be non-regular, then interp to a regular grid
        interp_fn = RegularGridInterpolator((row, column), data, method='linear')
        # take the new grid values, flatten and concatenate, so that we have an array of pairs of coordinates
        grid_points = np.array([col_new.flatten(), row_new.flatten()]).T
        # get array of new data
        new_data = interp_fn(grid_points)
        # put back into a matrix
        m, n = np.shape(data)  # shape of data
        new_data = new_data.reshape((m, n))
    return new_data

def dp_bounds(x_dep, y_dep, array_coords):
    """
    Truncate the departure points at the border of the domain,set the departure point to be the on the boundary
    if it is outside the domain.
    Inputs: x_dep, y_dep, matrices of coordinates
            array_coords, dictionary of coordinates for the array
    Outputs: x_dep, y_dep, truncated departure points
    """
    x_dep = np.where(x_dep >= array_coords['x'].min(), x_dep, array_coords['x'].min())
    x_dep = np.where(x_dep <= array_coords['x'].max(), x_dep, array_coords['x'].max())
    y_dep = np.where(y_dep >= array_coords['y'].min(), y_dep, array_coords['y'].min())
    y_dep = np.where(y_dep <= array_coords['y'].max(), y_dep, array_coords['y'].max())
    return x_dep, y_dep

def depart_eta(u, v, dt, coords, method = None):
    """
    Method for finding the departure points of elevation in the Semi-Lagrangian scheme
    using wind either at the end or midpoint of back trajectory.
    Inputs:  u, zonal velocity grid of values,
             v, meridional velocity grid of values
             dt, time step
             coords, dictionary of coordinates for eta, u and v
             method, method to find the departure points, if none use wind at the end point,
                     if mid use wind at the midpoint of back trajectory.
    Outputs: x_dep, y_dep, departure points
    """

    eta_coords, u_coords, v_coords = coords

    x_mat, y_mat = np.meshgrid(eta_coords['x'], eta_coords['y'])

    if method is None:
        x_dep = x_mat - dt * interp_zonal(u)
        y_dep = y_mat - dt * interp_merid(v)

    elif method == 'mid':
        # Finding the midpoint of the back trajectory (we only consider half time-step)
        x_mid = x_mat - 0.5 * dt * interp_zonal(u)
        y_mid = y_mat - 0.5 * dt * interp_merid(v)

        # Truncating the departure points at the border of the domain
        x_mid, y_mid = dp_bounds(x_mid, y_mid, eta_coords)

        # Computing u and v by linear interpolation in the midpoints
        interp_u, interp_v = interp2d(u_coords['x'], u_coords['y'], u), interp2d(v_coords['x'], v_coords['y'], v)
        u_mid, v_mid = interp_u(x_mid[0, :], y_mid[:, 0]), interp_v(x_mid[0, :], y_mid[:, 0])

        # Determining the departure point based on the wind at the midpoint of back trajectory
        x_dep = x_mat - dt * u_mid
        y_dep = y_mat - dt * v_mid

    x_dep, y_dep = dp_bounds(x_dep, y_dep, eta_coords)

    return x_dep, y_dep

def depart_u(u, v, dt, coords, method = None):
    """
    Method for finding the departure points of the zonal velocity in the Semi-Lagrangian scheme
    using wind either at the end or midpoint of back trajectory.
    Inputs:  u, zonal velocity grid of values,
             v, meridional velocity grid of values
             dt, time step
             coords, dictionary of coordinates for eta, u and v
             method, method to find the departure points, if none use wind at the end point,
                     if mid use wind at the midpoint of back trajectory.
    Outputs: x_dep, y_dep, departure points
    """

    eta_coords, u_coords, v_coords = coords

    x_mat, y_mat = np.meshgrid(u_coords['x'], u_coords['y'])

    if method is None:
        x_dep = x_mat - dt * u
        y_dep = y_mat - dt * interp_merid(interp_zonal(v))

    elif method == 'mid':
        # Finding the midpoint of the back trajectory (we only consider half time-step)
        x_mid = x_mat - 0.5 * dt * u
        y_mid = y_mat - 0.5 * dt * interp_zonal(interp_merid(v))

        # Truncating the departure points at the border of the domain
        x_mid, y_mid = dp_bounds(x_mid, y_mid, eta_coords)

        # Computing u and v by linear interpolation in the midpoints
        interp_u, interp_v = interp2d(u_coords['x'], u_coords['y'], u), interp2d(v_coords['x'], v_coords['y'], v)
        u_mid, v_mid = interp_u(x_mid[0, :], y_mid[:, 0]), interp_v(x_mid[0, :], y_mid[:, 0])

        # Determining the departure point based on the wind at the midpoint of back trajectory
        x_dep = x_mat - dt * u_mid
        y_dep = y_mat - dt * v_mid

    x_dep, y_dep = dp_bounds(x_dep, y_dep, u_coords)

    return x_dep, y_dep

def depart_v(u, v, dt, coords, method = None):
    """
    Method for finding the departure points of the meridional velocity in the Semi-Lagrangian scheme
    using wind either at the end or midpoint of back trajectory.
    Inputs:  u, zonal velocity grid of values,
             v, meridional velocity grid of values
             dt, time step
             coords, dictionary of coordinates for eta, u and v
             method, method to find the departure points, if none use wind at the end point,
                     if mid use wind at the midpoint of back trajectory.
    Outputs: x_dep, y_dep, departure points
    """

    eta_coords, u_coords, v_coords = coords

    x_mat, y_mat = np.meshgrid(v_coords['x'], v_coords['y'])

    if method is None:
        x_dep = x_mat - dt * interp_zonal(interp_merid(u))
        y_dep = y_mat - dt * v

    elif method == 'mid':
        # Finding the midpoint of the back trajectory (we only consider half time-step)
        x_mid = x_mat - 0.5 * dt * interp_zonal(interp_merid(u))
        y_mid = y_mat - 0.5 * dt * v

        # Truncating the departure points at the border of the domain
        x_mid, y_mid = dp_bounds(x_mid, y_mid, eta_coords)

        # Computing u and v by linear interpolation in the midpoints
        interp_u, interp_v = interp2d(u_coords['x'], u_coords['y'], u), interp2d(v_coords['x'], v_coords['y'], v)
        u_mid, v_mid = interp_u(x_mid[0, :], y_mid[:, 0]), interp_v(x_mid[0, :], y_mid[:, 0])

        # Determining the departure point based on the wind at the midpoint of back trajectory
        x_dep = x_mat - dt * u_mid
        y_dep = y_mat - dt * v_mid

    x_dep, y_dep = dp_bounds(x_dep, y_dep, v_coords)

    return x_dep, y_dep

def interp_depart(array, array_coords, x_dep, y_dep):
    '''
    Method to interpolate an array to the departure points using the cubic regular grid interpolator.
    Inputs: array, grid of values we want to interpolate to
            array_coords, dictionary of coordinates for the array
            x_dep, y_dep, departure points
    Outputs: array_new, array values at the departure points
    '''
    array_grid = (array_coords['y'], array_coords['x'])
    # Interp function
    interp = RegularGridInterpolator(array_grid, array, method='linear', bounds_error=False, fill_value=None)
    array_new = interp((y_dep, x_dep))

def linear_depart(array, u, v, dt, array_coords, u_coords, v_coords):
    '''
    Method for finding the departure points in the Semi-Lagrangian scheme using
    linear interpolation
    Inputs: array, grid of values we want to interpolate to
            u, zonal velocity grid of values,
            v, meridonal velocity grid of vlaues
            dt, time step
            array_coords, dictionary of coordinates for the array
            u_coords, dictionary of coordinates for u
            vcoords, dictionary of coordinates for v
    Outputs: array, array values at the departure points
    '''

    # use interp2d for finding departure points
    # use regular grid interpolator for interpolating array to dp

    # create grid of coords
    x_mat, y_mat = np.meshgrid(array_coords['x'], array_coords['y'])
    # Finding departure points
    array_grid = (array_coords['y'], array_coords['x'])
    u_grid = (u_coords['y'], u_coords['x'])
    v_grid = (v_coords['y'], v_coords['x'])

    x_dep = x_mat - dt * interp_function(u_grid, u, array_grid, 'interp2d')
    y_dep = y_mat - dt * interp_function(v_grid, v, array_grid, 'interp2d')

    # Boundary conditions: truncating the departure points at the border of the domain
    # Set the departure point to be the on the boundary if it is outside the domain THIS CAN BE OPTIONAL
    x_dep, y_dep = dp_bounds(x_dep, y_dep, array_coords)

    # Interpolating array at the departure points
    array = interp_function(array_grid, array, (x_dep, y_dep), 'Regular')
    return array

def linear_depart_eta(array, u, v, dt, array_coords):
    '''
    Method for finding the departure points in the Semi-Lagrangian scheme using
    linear interpolation
    Inputs: array, grid of values we want to interpolate to
            u, zonal velocity grid of values,
            v, meridonal velocity grid of vlaues
            dt, time step
            array_coords, dictionary of coordinates for the array
            u_coords, dictionary of coordinates for u
            vcoords, dictionary of coordinates for v
    Outputs: array, array values at the departure points
    '''

    # use interp2d for finding departure points
    # use regular grid interpolator for interpolating array to dp

    # create grid of coords
    x_mat, y_mat = np.meshgrid(array_coords['x'], array_coords['y'])
    # Finding departure points
    array_grid = (array_coords['y'], array_coords['x'])

    x_dep = x_mat - dt * interp_zonal(u)
    y_dep = y_mat - dt * interp_merid(v)

    # Boundary conditions: truncating the departure points at the border of the domain
    # Set the departure point to be the on the boundary if it is outside the domain THIS CAN BE OPTIONAL
    # make this a separate function
    x_dep = np.where(x_dep >= array_coords['x'].min(), x_dep, array_coords['x'].min())
    x_dep = np.where(x_dep <= array_coords['x'].max(), x_dep, array_coords['x'].max())
    y_dep = np.where(y_dep >= array_coords['y'].min(), y_dep, array_coords['y'].min())
    y_dep = np.where(y_dep <= array_coords['y'].max(), y_dep, array_coords['y'].max())

    ## Interpolating array at the departure points
    # Interp function
    interp = RegularGridInterpolator(array_grid, array, bounds_error=False, fill_value=None)
    array = interp((y_dep, x_dep))
    return array

def linear_depart_u(u, v, dt, u_coords):
    '''
    Method for finding the departure points in the Semi-Lagrangian scheme using
    linear interpolation
    Inputs: array, grid of values we want to interpolate to
            u, zonal velocity grid of values,
            v, meridional velocity grid of values
            dt, time step
            array_coords, dictionary of coordinates for the array
            u_coords, dictionary of coordinates for u
            vcoords, dictionary of coordinates for v
    Outputs: array, array values at the departure points
    '''

    # use interp2d for finding departure points
    # use regular grid interpolator for interpolating array to dp

    # create grid of coords
    x_mat, y_mat = np.meshgrid(u_coords['x'], u_coords['y'])
    # Finding departure points
    array_grid = (u_coords['y'], u_coords['x'])
    x_dep = x_mat - dt * u
    y_dep = y_mat - dt * interp_merid(interp_zonal(v))
    # Boundary conditions: truncating the departure points at the border of the domain
    # Set the departure point to be the on the boundary if it is outside the domain THIS CAN BE OPTIONAL
    # make this a separate function
    x_dep = np.where(x_dep >= u_coords['x'].min(), x_dep, u_coords['x'].min())
    x_dep = np.where(x_dep <= u_coords['x'].max(), x_dep, u_coords['x'].max())
    y_dep = np.where(y_dep >= u_coords['y'].min(), y_dep, u_coords['y'].min())
    y_dep = np.where(y_dep <= u_coords['y'].max(), y_dep, u_coords['y'].max())

    ## Interpolating array at the departure points
    # Interp function
    interp = RegularGridInterpolator(array_grid, u, bounds_error=False, fill_value=None)
    u_new = interp((y_dep, x_dep))
    return u_new

def linear_depart_v(v, u, dt, v_coords):
    '''
    Method for finding the departure points in the Semi-Lagrangian scheme using
    linear interpolation
    Inputs: array, grid of values we want to interpolate to
            u, zonal velocity grid of values,
            v, meridonal velocity grid of vlaues
            dt, time step
            array_coords, dictionary of coordinates for the array
            u_coords, dictionary of coordinates for u
            vcoords, dictionary of coordinates for v
    Outputs: array, array values at the departure points
    '''

    # use interp2d for finding departure points
    # use regular grid interpolator for interpolating array to dp

    # create grid of coords
    x_mat, y_mat = np.meshgrid(v_coords['x'], v_coords['y'])

    # Finding departure points
    array_grid = (v_coords['y'], v_coords['x'])

    x_dep = x_mat - dt * interp_zonal(interp_merid(u))
    y_dep = y_mat - dt * v

    # Boundary conditions: truncating the departure points at the border of the domain
    # Set the departure point to be the on the boundary if it is outside the domain THIS CAN BE OPTIONAL
    # make this a separate function
    x_dep = np.where(x_dep >= v_coords['x'].min(), x_dep, v_coords['x'].min())
    x_dep = np.where(x_dep <= v_coords['x'].max(), x_dep, v_coords['x'].max())
    y_dep = np.where(y_dep >= v_coords['y'].min(), y_dep, v_coords['y'].min())
    y_dep = np.where(y_dep <= v_coords['y'].max(), y_dep, v_coords['y'].max())

    ## Interpolating array at the departure points
    # Interp function
    interp = RegularGridInterpolator(array_grid, v, bounds_error=False, fill_value=None)
    v_new = interp((y_dep, x_dep))
    return v_new