"""
Parameters for the SWEs.
@author : Laura Risley, 2025
"""

import numpy as np

param = { "L_x" : 1e6,      # computational domain in x
          "L_y" : 1e6,      # computational domain in y
          "f_0" : 1e-4,     # coriolis parameter
          "beta" : 1e-11,   # beta-plane
          "g" : 10,         # gravitational acceleration
          "gamma" : 1e-12,  # linear drag coefficient
          "rho" : 1e3,      # uniform density
          "H" : 1e3,        # resting depth of the fluid (assumed constant)
          "tau_0" : 0.2,    # wind stress constant
          "gtau" : 0.02,    # wind intensity
          "gtau_s": 0.01,   # monthly wind intensity
          "t_max": 720,     # hours until end of the month (30 days)
          "t_min": 360,     # hours until mid-month (15 days)
          "rho_a" : 1.22,   # air density
          "drag" : 1.5e-3   # drag coefficient
}

dimensions = {"dx" : 1e4,                    # spatial step length
              "dy" : 1e4,
              "dt" : 1800,                   # temporal step length in seconds
              "nt_1d": np.ceil(86400/1800),  # one day
              "nt" : 30 * np.ceil(86400/1800)
              # to run for 1 cycle, put nt = 1
}