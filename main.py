"""
Load a model output for one day using the SWES.
Perform the Helmholtz decomposition.
@author : Laura Risley, 2025
"""

import numpy as np

from helmholtz_decomp import perform_helmholtz

n = 330.0

# model output of SISL
eta = np.loadtxt(f'elevation_time_{n}_days', delimiter=",")
u = np.loadtxt(f'u_time_{n}_days', delimiter=",")
v = np.loadtxt(f'v_time_{n}_days', delimiter=",")
eta_new = np.loadtxt(f'elevation_time_{n + 1}_days', delimiter=",")
u_new = np.loadtxt(f'u_time_{n + 1}_days', delimiter=",")
v_new = np.loadtxt(f'v_time_{n + 1}_days', delimiter=",")

# find the increments
d_eta, d_u, d_v = eta_new - eta, u_new - u, v_new - v

# filter options
filter = None # No Shapiro filter is applied to streamfunction
#filter = d_sf # The Shapiro filter is applied to streamfunction

#perform the Helmholtz decomposition
perform_helmholtz(d_eta, d_u, d_v, n, filter)
