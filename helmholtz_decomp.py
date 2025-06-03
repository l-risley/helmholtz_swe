"""
Functions to perform the Helmholtz decomposition.
@author : Laura Risley, 2025
"""

from scipy.optimize import minimize, fmin_cg
from scipy.sparse.linalg import LinearOperator, cg, eigs
import numpy as np
from numpy.linalg import norm, cond

from parameters import *
from set_up import *
from functions import *
from plotting import *

def vel_from_helm2(sf, vp, dx, dy):
    """
    Transform stream function and velocity potential to horizontal velocity vectors, based on Helmholtz theorem.
    u = - d sf/dy + d vp/dx
    v =  d sf/dx + d vp/dy
    Boundaries are ignored for sf and vp.
    Inputs:  - sf, streamfunction matrix (ny, nx)
             - vp, velocity potential matrix (ny, nx)
             - dx, dy, spatial grid length
    Outputs: - u, v, horizontal velocity matrices (ny, nx+1), (ny+1, nx)
    """
    ny, nx = np.shape(sf)

    # y-derivative of streamfunction
    u_sf = np.zeros((ny, nx + 1))
    u_sf[0, 1:-1] = 0.5 * 1 / dy * (-sf[0, :-1] - sf[0, 1:] + sf[1, :-1] + sf[1, 1:])
    u_sf[-1, 1:-1] = 0.5 * 1 / dy * (-sf[-2, :-1] - sf[-2, 1:] + sf[-1, :-1] + sf[-1, 1:])
    u_sf[1:-1, 1:-1] = 0.25 * 1 / dy * (-sf[:-2, :-1] - sf[:-2, 1:] + sf[2:, :-1] + sf[2:, 1:])

    # x-derivative of velocity potential
    u_vp = np.zeros((ny, nx + 1))
    u_vp[:, 1:-1] = 1 / dx * (vp[:, 1:] - vp[:, :-1])
    # find u
    u = - u_sf + u_vp
    u = zonal_boundary(u)

    # x-derivative of streamfunction
    v_sf = np.zeros((ny + 1, nx))
    v_sf[1:-1, 0] = 0.5 * 1 / dx * (-sf[:-1, 0] - sf[1:, 0] + sf[:-1, 1] + sf[1:, 1])
    v_sf[1:-1, -1] = 0.5 * 1 / dx * (-sf[:-1, -2] - sf[1:, -2] + sf[:-1, -1] + sf[1:, -1])
    v_sf[1:-1, 1:-1] = 0.25 * 1 / dx * (-sf[:-1, :-2] - sf[1:, :-2] + sf[:-1, 2:] + sf[1:, 2:])

    # y-derivative of velocity potential
    v_vp = np.zeros((ny + 1, nx))
    v_vp[1:-1, :] = 1 / dy * (vp[1:, :] - vp[:-1, :])
    # find v
    v = v_sf + v_vp
    v = meridional_boundary(v)

    return u, v

def vel_from_helm_adj2(u, v, dy, dx, ny, nx):
    """
    Adjoint of the transform stream function and velocity potential to horizontal velocity vectors, based on Helmholtz theorem.
    u = - d sf/dy + d vp/dx
    v =  d sf/dx + d vp/dy
    Boundaries are ignored for sf and vp.
    Inputs:  - u, v, horizontal velocity matrices (ny, nx+1), (ny+1, nx)
             - dx, dy, spatial grid length
    Outputs: - sf, vp, streamfunction matrix, velocity potential matrix (ny, nx)
    """

    # initialise sf and vp
    sf, vp = np.zeros((ny, nx)), np.zeros((ny, nx))

    v_dx = 1 / dx * v
    v_dy = 1 / dy * v

    ## adjoint routine begins
    # adjoint of v =  d sf/dx + d vp/dy
    sf[1:, 2:] += 0.25 * v_dx[1:-1, 1:-1]
    sf[:-1, 2:] += 0.25 * v_dx[1:-1, 1:-1]
    sf[1:, :-2] += - 0.25 * v_dx[1:-1, 1:-1]
    sf[:-1, :-2] += - 0.25 * v_dx[1:-1, 1:-1]

    sf[1:, -1] += 0.5 * v_dx[1:-1, -1]
    sf[:-1, -1] += 0.5 * v_dx[1:-1, -1]
    sf[1:, -2] += - 0.5 * v_dx[1:-1, -1]
    sf[:-1, -2] += - 0.5 * v_dx[1:-1, -1]

    sf[1:, 1] += 0.5 * v_dx[1:-1, 0]
    sf[:-1, 1] += 0.5 * v_dx[1:-1, 0]
    sf[1:, 0] += - 0.5 * v_dx[1:-1, 0]
    sf[:-1, 0] += - 0.5 * v_dx[1:-1, 0]

    vp[:-1, :] += - v_dy[1:-1, :]
    vp[1:, :] += v_dy[1:-1, :]

    v = 0

    u_dy = 1 / dy * u
    u_dx = 1 / dx * u

    # adjoint of u =  -d sf/dy + d vp/dx
    sf[2:, :-1] += - 0.25 * u_dy[1:-1, 1:-1]
    sf[2:, 1:] += - 0.25 * u_dy[1:-1, 1:-1]
    sf[:-2, 1:] += 0.25 * u_dy[1:-1, 1:-1]
    sf[:-2, :-1] += 0.25 * u_dy[1:-1, 1:-1]

    sf[-1, 1:] += - 0.5 * u_dy[-1, 1:-1]
    sf[-1, :-1] += - 0.5 * u_dy[-1, 1:-1]
    sf[-2, 1:] += 0.5 * u_dy[-1, 1:-1]
    sf[-2, :-1] += 0.5 * u_dy[-1, 1:-1]

    sf[1, 1:] += - 0.5 * u_dy[0, 1:-1]
    sf[1, :-1] += - 0.5 * u_dy[0, 1:-1]
    sf[0, 1:] += 0.5 * u_dy[0, 1:-1]
    sf[0, :-1] += 0.5 * u_dy[0, 1:-1]

    vp[:, :-1] += - u_dx[:, 1:-1]
    vp[:, 1:] += u_dx[:, 1:-1]

    u = 0

    return sf, vp

def vel_from_helm_filter(sf, vp, dy, dx):
    """
    Transform streamfunction and velocity potential to horizontal velocity vectors, based on Helmholtz theorem.
    u = - d sf/dy + d vp/dx
    v =  d sf/dx + d vp/dy
    Boundaries are ignored for sf and vp.
    Apply the Shapiro filter to the streamfunction derivatives.
    Inputs:  - sf, streamfunction matrix (ny, nx)
             - vp, velocity potential matrix (ny, nx)
             - dx, dy, spatial grid length
    Outputs: - u, v, horizontal velocity matrices (ny, nx+1), (ny+1, nx)
    """
    ny, nx = np.shape(sf)

    # y-derivative of streamfunction
    u_sf = np.zeros((ny, nx + 1))
    u_sf[0, 1:-1] = 0.5 * 1 / dy * (-sf[0, :-1] - sf[0, 1:] + sf[1, :-1] + sf[1, 1:])
    u_sf[-1, 1:-1] = 0.5 * 1 / dy * (-sf[-2, :-1] - sf[-2, 1:] + sf[-1, :-1] + sf[-1, 1:])
    u_sf[1:-1, 1:-1] = 0.25 * 1 / dy * (-sf[:-2, :-1] - sf[:-2, 1:] + sf[2:, :-1] + sf[2:, 1:])

    # apply the shapiro filter to the sf derivative
    u_sf_filter = shapiro_filter_2d_u(u_sf)
    # u_sf_filter = shapiro_filter_2d_u(u_sf_filter)
    # u_sf_filter = shapiro_filter_2d_u(u_sf_filter)

    # x-derivative of velocity potential
    u_vp = np.zeros((ny, nx + 1))
    u_vp[:, 1:-1] = 1 / dx * (vp[:, 1:] - vp[:, :-1])
    # find u
    u = - u_sf_filter + u_vp
    u = zonal_boundary(u)

    # x-derivative of streamfunction
    v_sf = np.zeros((ny + 1, nx))
    v_sf[1:-1, 0] = 0.5 * 1 / dx * (-sf[:-1, 0] - sf[1:, 0] + sf[:-1, 1] + sf[1:, 1])
    v_sf[1:-1, -1] = 0.5 * 1 / dx * (-sf[:-1, -2] - sf[1:, -2] + sf[:-1, -1] + sf[1:, -1])
    v_sf[1:-1, 1:-1] = 0.25 * 1 / dx * (-sf[:-1, :-2] - sf[1:, :-2] + sf[:-1, 2:] + sf[1:, 2:])

    # apply the shapiro filter to the sf derivative
    v_sf_filter = shapiro_filter_2d_v(v_sf)
    # v_sf_filter = shapiro_filter_2d_v(v_sf_filter)
    # v_sf_filter = shapiro_filter_2d_v(v_sf_filter)

    # y-derivative of velocity potential
    v_vp = np.zeros((ny + 1, nx))
    v_vp[1:-1, :] = 1 / dy * (vp[1:, :] - vp[:-1, :])
    # find v
    v = v_sf_filter + v_vp
    v = meridional_boundary(v)

    return u, v

def vel_from_helm_filter_adj(u, v, dy, dx, ny, nx):
    """
    Adjoint of the transform streamfunction and velocity potential to horizontal velocity vectors, based on Helmholtz theorem.
    u = - d sf/dy + d vp/dx
    v =  d sf/dx + d vp/dy
    Boundaries are ignored for sf and vp.
    Apply the Shapiro filter to the streamfunction derivatives.
    Inputs:  - u, v, horizontal velocity matrices (ny, nx+1), (ny+1, nx)
             - dx, dy, spatial grid length
    Outputs: - sf, vp, streamfunction matrix, velocity potential matrix (ny, nx)
    """

    # initialise sf and vp
    v_sf, v_vp = np.zeros_like(v), np.zeros_like(v)
    u_sf, u_vp = np.zeros_like(u), np.zeros_like(u)
    sf, vp = np.zeros((ny, nx)), np.zeros((ny, nx))

    ## adjoint routine begins
    # adjoint of v =  d sf/dx + d vp/dy

    v_sf += v
    v_vp += v

    v = 0

    v_sf = shapiro_filter_2d_v_adj(v_sf)
    # v_sf = shapiro_filter_2d_v_adj(v_sf)
    # v_sf = shapiro_filter_2d_v_adj(v_sf)

    v_dx = 1 / dx * v_sf
    v_dy = 1 / dy * v_vp

    sf[1:, 2:] += 0.25 * v_dx[1:-1, 1:-1]
    sf[:-1, 2:] += 0.25 * v_dx[1:-1, 1:-1]
    sf[1:, :-2] += - 0.25 * v_dx[1:-1, 1:-1]
    sf[:-1, :-2] += - 0.25 * v_dx[1:-1, 1:-1]

    sf[1:, -1] += 0.5 * v_dx[1:-1, -1]
    sf[:-1, -1] += 0.5 * v_dx[1:-1, -1]
    sf[1:, -2] += - 0.5 * v_dx[1:-1, -1]
    sf[:-1, -2] += - 0.5 * v_dx[1:-1, -1]

    sf[1:, 1] += 0.5 * v_dx[1:-1, 0]
    sf[:-1, 1] += 0.5 * v_dx[1:-1, 0]
    sf[1:, 0] += - 0.5 * v_dx[1:-1, 0]
    sf[:-1, 0] += - 0.5 * v_dx[1:-1, 0]

    vp[:-1, :] += - v_dy[1:-1, :]
    vp[1:, :] += v_dy[1:-1, :]

    # adjoint of u =  -d sf/dy + d vp/dx

    u_sf += -u
    u_vp += u

    u = 0

    u_sf = shapiro_filter_2d_u_adj(u_sf)
    # u_sf = shapiro_filter_2d_u_adj(u_sf)
    # u_sf = shapiro_filter_2d_u_adj(u_sf)

    u_dy = 1 / dy * u_sf
    u_dx = 1 / dx * u_vp

    sf[2:, :-1] += 0.25 * u_dy[1:-1, 1:-1]
    sf[2:, 1:] += 0.25 * u_dy[1:-1, 1:-1]
    sf[:-2, 1:] += - 0.25 * u_dy[1:-1, 1:-1]
    sf[:-2, :-1] += - 0.25 * u_dy[1:-1, 1:-1]

    sf[-1, 1:] += 0.5 * u_dy[-1, 1:-1]
    sf[-1, :-1] += 0.5 * u_dy[-1, 1:-1]
    sf[-2, 1:] += - 0.5 * u_dy[-1, 1:-1]
    sf[-2, :-1] += - 0.5 * u_dy[-1, 1:-1]

    sf[1, 1:] += 0.5 * u_dy[0, 1:-1]
    sf[1, :-1] += 0.5 * u_dy[0, 1:-1]
    sf[0, 1:] += - 0.5 * u_dy[0, 1:-1]
    sf[0, :-1] += - 0.5 * u_dy[0, 1:-1]

    vp[:, :-1] += - u_dx[:, 1:-1]
    vp[:, 1:] += u_dx[:, 1:-1]

    return sf, vp

def A_operator(x, dy, dx, ny, nx, filter):
    """
    The linear operator Ax for the transform from stream function and velocity potential (x) to horizontal velocity vectors (b).
    Inputs:  - x, vector containing sf (streamfunction) and vp (velocity potential) (2* ny_cv*nx_cv = 2* ny * nx)
             - dy, dx, spatial grid length
             - ny, nx, number of sf and vp points on the grid
             - filter, either None or 'd_sf', which applies the Shapiro filter to the sf derivatives
    Outputs: - b, vector containing the horizontal velocities, u and v (ny*(nx+1) + nx*(ny+1))
    """

    # sizes of sf and vp
    ny_cv, nx_cv = ny, nx

    # split x into two equal arrays for sf and vp
    sf, vp = split_x(x, ny_cv, nx_cv)

    # apply the u-transform
    if filter is None or filter == 'vel':
        u, v = vel_from_helm2(sf, vp, dy, dx)
    if filter == 'd_sf':
        u, v = vel_from_helm_filter(sf, vp, dy, dx)

    # flatten to a vector
    u_vec = u.flatten()
    v_vec = v.flatten()

    # created one vector containing both u and v
    b = np.append(u_vec, v_vec)
    return b

def A_adjoint(b, dy, dx, ny, nx, filter):
    """
    The adjoint of linear operator Ax.
    Inputs:  - b, vector containing the horizontal velocities, u and v (ny*(nx+1) + nx*(ny+1))x
             - dy, dx, spatial grid length
             - ny, nx, number of eta points on the grid (ny, nx)
             - filter, either None or 'd_sf', which applies the Shapiro filter to the sf derivatives
    Outputs: - x, vector containing sf (streamfunction) and vp (velocity potential) (2*ny*nx)
    """
    # split b into two equal arrays for u and v
    u, v = split_b(b, ny, nx)

    if filter is None or filter == 'vel':
        sf, vp = vel_from_helm_adj2(u, v, dy, dx, ny, nx)
    if filter == 'd_sf':
        sf, vp = vel_from_helm_filter_adj(u, v, dy, dx, ny, nx)

    # flatten to a vector
    sf_vec = sf.flatten()
    vp_vec = vp.flatten()

    # created one vector containing both sf and vp
    x = np.append(sf_vec, vp_vec)
    return x

def min_method(fn, grad, x0: np.ndarray, conv):
    """
    Minimisation function for the cost function in 3D VAR
    Output minimisation value and list of functions at each interation.
    Inputs: fn, function to be minimised
           grad, gradient of the function
           x0, initial input
           conv, None - output only minimised value
                 Convergence - output minimised value, values of fn and grad-norm at each iteration
    """
    print('Began minimisation')
    gtol = 1e-05 * (norm(grad(x0), np.inf).item())
    print(f'Minimisation criteria: gradient norm < {gtol}.')
    method = 'CG'
    if conv is None:
        return minimize(fn, x0, method=method, jac=grad,
                        options={'disp': True, 'gtol': gtol})

    elif conv == 'convergence':
        all_fn = [fn(x0).item()]
        all_grad = [norm(grad(x0), np.inf).item()]

        def store(x):  # callback function
            all_fn.append(fn(x).item())
            all_grad.append(norm(grad(x), np.inf).item())

        ans = minimize(fn, x0, method=method, jac=grad, callback=store,
                       options={'disp': True, 'gtol': gtol})  # , 'maxiter': 200
        print(all_fn)
        print(all_grad)
        return ans, all_fn, all_grad

def tik_reg(tk, u, v, dy, dx, ny, nx, conv=None, filter=None):
    """
    Tikhonov's regularisation to find the horizontal velocity vectors from streamfunction and velocity potential
    Inputs: tk, regularisation parameter
            u, v, horizontal velocity vectors
            dy, dx, spatial grid length
            ny, nx, number of eta points on the grid (ny, nx)
            conv, None - output only minimised value
                  Convergence - output minimised value, values of fn and grad-norm at each iteration
    Outputs: sf, streamfunction (ny+1, nx+1)
             vp, velocity potential (ny, nx)
             :param filter:
    """

    # put u and v into a vector
    b = np.append(u.flatten(), v.flatten())

    # input x is a vector
    # costfunction
    def tik_fun(x):
        # J_a = 0.5* (b-Ax)^T(b-Ax) + a*0.5*x^Tx
        J_x = b - A_operator(x, dy, dx, ny, nx, filter)
        J = np.dot(J_x, J_x)
        J_reg = tk * np.dot(x, x)
        return 0.5 * (J + J_reg)

    # gradient
    def tik_grad(x):
        # grad_J = -A^T(b-Ax) + a*x
        # b-Ax
        J_x = b - A_operator(x, dy, dx, ny, nx, filter)
        # adjoint applied to b-Ax
        adj = A_adjoint(J_x, dy, dx, ny, nx, filter)
        return -adj + tk * x

    # sizes of sf and vp
    ny_cv, nx_cv = ny, nx

    # initial guess for minimisation
    x_0 = np.zeros(2 * ny_cv * nx_cv)

    result = min_method(tik_fun, tik_grad, x_0, conv)  # use pre-defined fn to optimise

    if conv is None:
        x_arr = np.asarray(result.x)
        sf, vp = split_x(x_arr, ny_cv, nx_cv)
        return sf, vp
    elif conv == 'convergence':
        ans, cf_list, grad_list = result
        x_arr = np.asarray(ans.x)
        sf, vp = split_x(x_arr, ny_cv, nx_cv)
        cf_array = np.asarray(cf_list)
        grad_array = np.asarray(grad_list)
        return sf, vp, cf_array, grad_array

def perform_helmholtz(d_eta, d_u, d_v, n, filter):
    """
    """
    # domain coordinates for plotting
    coords_plot = array_coords_plotting(param, dimensions)
    eta_coords_p, u_coords_p, v_coords_p = coords_plot[0], coords_plot[1], coords_plot[2]

    # set dimensions
    dx, dy, dt, nt = dimensions['dx'], dimensions['dy'], dimensions['dt'], \
                     int(dimensions['nt'])

    # plot the increments
    contour(eta_coords_p['x'], eta_coords_p['y'], d_eta, f'Elevation increment( {np.int(n+1)} days)', 'Elevation')
    contour(u_coords_p['x'], u_coords_p['y'], d_u, f'Zonal velocity increment ({np.int(n+1)} days)', 'Zonal velocity')
    contour(v_coords_p['x'], v_coords_p['y'], d_v, f'Meridional velocity increment ({np.int(n+1)} days)', 'Meridional velocity')

    ## ----------------- Find the control variables -----------------
    # set a value for the tikhonov regularisation parameters
    tk = 1e-12

    ny, nx = np.shape(d_eta)

    # ----------------- Find streamfunction and velocity potential -------------------
    # Use Tikhonov's regularisation
    ny, nx = np.shape(d_eta)

    # choice of convergence
    conv = 'convergence'
    if conv is None:
        sf, vp = tik_reg(tk, d_u, d_v, dy, dx, ny, nx, conv, None)
    elif conv == 'convergence':
        sf, vp, cf_list, grad_list = tik_reg(tk, d_u, d_v, dy, dx, ny, nx, conv, None)
        # plot the convergences
        plot_one_convergence(cf_list, 'Cost Function')
        plot_one_convergence(grad_list, 'Gradient Norm')

    # smooth the fields using a Shapiro filter (post-processing)
    if filter == 'd_sf':
        sf = shapiro_filter_2d_neumann(sf)

    # plot streamfunction and velocity potential
    contour(eta_coords_p['x'], eta_coords_p['y'], sf, f'Streamfunction increment ({np.int(n+1)} days)', 'Streamfunction')
    contour(eta_coords_p['x'], eta_coords_p['y'], vp, f'Velocity potential increment 9{np.int(n+1)} days)', 'Velocity potential')

    ## ----------------- U-transfrom to reconstruct u and v -----------------
    u_re, v_re = vel_from_helm2(sf, vp, dy, dx)

    # plot reconstructed
    contour(u_coords_p['x'], u_coords_p['y'], u_re, 'Reconstructed zonal velocity increment', 'Zonal velocity')
    contour(v_coords_p['x'], v_coords_p['y'], v_re, 'Reconstructed meridional velocity increment', 'Meridional velocity')

    # plot differences
    contour(u_coords_p['x'], u_coords_p['y'], d_u - u_re, 'Zonal velocity reconstruction error', 'Zonal velocity')
    contour(v_coords_p['x'], v_coords_p['y'], d_v - v_re, 'Meridional velocity reconstruction error', 'Meridional velocity')