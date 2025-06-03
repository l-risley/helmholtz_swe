"""
Semi-implicit semi-Lagrangian scheme used to run the discrete shallow water model.
@author : Laura Risley, 2025
"""

from parameters import *
from set_up import *
from functions import *
from plotting import *
from scipy.sparse.linalg import LinearOperator, cg

def SISL(param, initial, coords, dx: int, dy: int, dt: int, nt: int, start_day):
    """
    Semi-implicit, semi-lagrangian scheme to solve the SWEs with Dirichlet BCs.
    Pressure gradient and velocity divergence are treated implicitly.
    Inputs:  param, dictionary of parameters
             initial, initial conditions
             coords, grid coordinates for eta, u and v
             dx, dy, spatial grid length
             dt, temporal step length
             nt, number of time steps
             start_day, number of day from start time (0) that this model run begins at
    Outputs: u, horizontal velocity after nt time steps
             v, vertical velocity after nt time steps
             eta, elevation after nt time steps
             t, time array
    """
    # ----------------- Initial set-up -------------------
    # semi-implicit time-weighted parameters
    alpha_0, alpha_1, alpha_2 = 0.6, 0.6, 0.6
    print(f'Running semi-implicit, semi-lagrangian scheme with time-weighted parameters = {alpha_0, alpha_1, alpha_2}.')

    # get physical parameters from dictionary
    L_x, L_y = param['L_x'], param['L_y'],
    f_0, beta, g, gamma, H, rho = param['f_0'], param['beta'], param['g'], param['gamma'], param['H'], param['rho'],
    t_max, t_min, gtau, gtau_s = param['t_max'], param['t_min'], param['gtau'], param['gtau_s']

    # set grid coordinates
    eta_coords, u_coords, v_coords = coords

    # set initial values
    eta, u, v = initial[0], initial[1], initial[2]

    # shape of eta
    ny, nx = np.shape(eta)

    # time array
    t = np.arange(0, nt * dt, dt)

    # constants to be used in semi-implicit matrix
    b_1 = (H * dt * alpha_0) / dx
    b_2 = (H * dt * alpha_0) / dy
    c_1 = (dt * g * alpha_1 * b_1) / dx
    c_2 = (dt * g * alpha_2 * b_2) / dy

    # matrices of u and v coords
    u_x_mat, u_y_mat = np.meshgrid(u_coords['x'], u_coords['y'])
    v_x_mat, v_y_mat = np.meshgrid(v_coords['x'], v_coords['y'])

    # ----------------- Discretised elliptic equation -------------------
    def A_eta(x):
        """
        Operator Ax for our implicit terms (eta^(n+1)) at the arrival points:
        (1 + 2C_1 + 2C_2) * eta_(j, i) - C_1 * (eta_(j, i+1) + eta_(j, i-1)) - C_2 * (eta_(j+1, i) + eta_(j-1)).
        Input: x, a row vector of length ny * nx
        Output: row vector for Ax operation
        """
        # need to make x into our matrix
        x_mat = np.reshape(x, np.shape(eta))

        Ax_mat = np.empty_like(x_mat)
        Ax_mat[1:-1, 1:-1] = (1 + 2 * c_1 + 2 * c_2) * x_mat[1:-1, 1:-1] - c_1 * (x_mat[1:-1, 2:] + x_mat[1:-1, :-2]) \
                             - c_2 * (x_mat[2:, 1:-1] + x_mat[:-2, 1:-1])

        # ----------------- Boundaries -------------------
        ## The corners first
        # eta_(0,0)
        Ax_mat[0, 0] = (1 + c_1 + c_2) * x_mat[0, 0] - c_1 * x_mat[0, 1] - c_2 * x_mat[1, 0]
        # eta_(N,0)
        Ax_mat[-1, 0] = (1 + c_1 + c_2) * x_mat[-1, 0] - c_1 * x_mat[-1, 1] - c_2 * x_mat[-2, 0]
        # eta_(0,N)
        Ax_mat[0, -1] = (1 + c_1 + c_2) * x_mat[0, -1] - c_1 * x_mat[0, -2] - c_2 * x_mat[1, -1]
        # eta_(N,N)
        Ax_mat[-1, -1] = (1 + c_1 + c_2) * x_mat[-1, -1] - c_1 * x_mat[-1, -2] - c_2 * x_mat[-2, -1]
        ## The west boundary
        Ax_mat[1:-1, 0] = (1 + c_1 + 2 * c_2) * x_mat[1:-1, 0] - c_1 * x_mat[1:-1, 1] - c_2 * (
                          x_mat[:-2, 0] + x_mat[2:, 0])
        ## The east boundary
        Ax_mat[1:-1, -1] = (1 + c_1 + 2 * c_2) * x_mat[1:-1, -1] - c_1 * x_mat[1:-1, -2] - c_2 * (
                          x_mat[:-2, -1] + x_mat[2:, -1])
        ## The south boundary
        Ax_mat[0, 1:-1] = (1 + c_1 + 2 * c_2) * x_mat[0, 1:-1] - c_1 * (x_mat[0, 2:] + x_mat[0, :-2]) \
                          - c_2 * x_mat[1, 1:-1]
        ## The north boundary
        Ax_mat[-1, 1:-1] = (1 + c_1 + 2 * c_2) * x_mat[-1, 1:-1] - c_1 * (x_mat[-1, 2:] + x_mat[-1, :-2]) \
                           - c_2 * x_mat[-2, 1:-1]
        return Ax_mat.flatten()  # return a row vector

    # ----------------- RHS of the elevation equation -------------------
    def eta_rhs(eta, u, v):
        """
        Function to calculate the rhs of the surface elevation equation, at the departure points.
        eta - H*dt*(1-a_0)*(du/dx + dv/dy) - dt*(u*deta/dx + v*deta/dy))
        Inputs: eta, surface elevation (ny, nx)
                u, zonal velocity (ny, nx+1)
                v, meridional velocity (ny+1, nx)
        Output: E, rhs of the surface elevation momentum equation (ny, nx)
        """
        # departure points
        x_dep, y_dep = depart_eta(u, v, dt, coords, 'mid')
        # velocity divergence
        vel_div = - H * dt * (1 - alpha_0) * (dzdx(u, dx) + dzdy(v, dy))

        E = eta + vel_div

        # interpolate to departure points
        E = interp_depart(E, eta_coords, x_dep, y_dep)
        return E

    # ----------------- Velocities -------------------

    # removing the boundaries from the velocities
    u_coords_nb = u_coords.copy()
    v_coords_nb = v_coords.copy()
    u_coords_nb['x'] = np.arange(0.5 * dx, L_x - 0.5 * dx, dx)
    v_coords_nb['y'] = np.arange(0.5 * dy, L_y - 0.5 * dy, dy)

    # ----------------- RHS of the zonal velocity momentum equation -------------------
    def u_rhs(eta, u, v, t_hrs):
        """
        Function to calculate the rhs of the zonal velocity equation, at the departure points.
        u - dt*(u*du/dx - v*dudy) + dt(f+by)*v - dt*gamma*u + tau_x/pH - g*dt(1-a_2)*deta/dx
        Inputs: eta, surface elevation (ny, nx)
                u, zonal velocity (ny, nx+1)
                v, meridional velocity (ny+1, nx)
                t_hrs, current time difference from initial time in hours
        Output: Z, rhs of the zonal velocity momentum equation (ny, nx+1)
        """
        # departure points
        x_dep, y_dep = depart_u(u[:, 1:-1], v, dt, (eta_coords, u_coords_nb, v_coords), 'mid')
        # coriolis
        cor_v = dt * cor(f_0, beta, u_y_mat[:, 1:-1], v)
        # pressure gradient
        pres_grad = - dt * g * (1 - alpha_1) * dzdx(eta, dx)

        # wind forcing
        tau = gyre_wind_intensity(t_hrs, t_max, t_min, gtau, gtau_s)
        wind = gyre_wind_forcing(tau, u_y_mat, L_y, 'u')

        # return u - dt * gamma * u + cor_v + pres_grad + dt * wind
        Z = np.zeros_like(u)

        Z[:, 1:-1] = u[:, 1:-1] - (dt * gamma * u[:, 1:-1]) + cor_v + pres_grad + \
                     (dt * wind[:, 1:-1] / (rho * H))

        # interpolate to departure points
        Z[:, 1:-1] = interp_depart(Z[:, 1:-1], u_coords_nb, x_dep, y_dep)
        return Z

    # ----------------- RHS of the meridional velocity momentum equation -------------------
    def v_rhs(eta, u, v):
        """
        Function to calculate the rhs of the meridional velocity equation, at the departure points.
         v- dt*(u*dv/dx - v*dvdy) - dt(f+by)*u - dt*gamma*v - g*dt(1-a_3)*deta/dy
        Inputs: eta, surface elevation (ny, nx)
                u, zonal velocity (ny, nx+1)
                v, meridional velocity (ny+1, nx)
        Output: M, rhs of the zonal velocity momentum equation (ny+1, nx)
        """
        # departure points
        x_dep, y_dep = depart_v(u, v[1:-1, :], dt, (eta_coords, u_coords, v_coords_nb), 'mid')

        # coriolis
        cor_u = - dt * cor(f_0, beta, v_y_mat[1:-1, :], u)
        # pressure gradient
        pres_grad = - dt * g * ((1 - alpha_2) * dzdy(eta, dy))

        # return v - dt * gamma * v + dt * wind + cor_u + pres_grad
        M = np.zeros_like(v)
        M[1:-1, :] = v[1:-1, :] + cor_u - dt * gamma * v[1:-1, :] + pres_grad

        # interpolate to departure points
        M[1:-1, :] = interp_depart(M[1:-1, :], v_coords_nb, x_dep, y_dep)
        return M

    # ----------------- Calculation loop -------------------
    for it in range(nt):
        # calculate the current time in hours (starting from 0) for the wind forcing
        t_days = start_day + it * (dt / 86400)
        t_hrs = np.remainder(t_days, 30) * 24

        # ----------------- Create a linear operator for Ax -------------------
        A = LinearOperator((ny * nx, ny * nx), matvec=A_eta)

        # ----------------- Calculate b as a vector -------------------
        # b = E_(j,i) - b_1 * (Z_(j, i+1) - Z_(j,i)) - b_2 * (M_(j+1, i) - M_(j,i))
        Z, M = u_rhs(eta, u, v, t_hrs), v_rhs(eta, u, v)
        E = eta_rhs(eta, u, v)
        Z_diff = np.diff(Z, axis=1)
        M_diff = np.diff(M, axis=0)
        b_mat = E - (b_1 * Z_diff) - (b_2 * M_diff)
        b = b_mat.flatten()

        # ----------------- Solve to find eta_new -------------------
        eta_new, exit_code = cg(A, b)
        # reshape back into a matrix
        eta_new = np.reshape(eta_new, (ny, nx))
        print(f'The convergence of eta at time {it} is {exit_code}')

        # ----------------- Solve to find u_new and v_new using eta_new -------------------
        u_new, v_new = np.zeros_like(u), np.zeros_like(v)
        u_new[:, 1:-1] = Z[:, 1:-1] - g * dt * alpha_1 * dzdx(eta_new, dx)
        v_new[1:-1, :] = M[1:-1, :] - g * dt * alpha_2 * dzdy(eta_new, dy)

        u, v, eta = u_new, v_new, eta_new

    return eta, u, v, t
