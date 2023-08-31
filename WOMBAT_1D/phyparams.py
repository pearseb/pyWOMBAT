#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 10:06:24 2023



@author: pbuchanan
"""

import numpy as np
from scipy.interpolate import interp1d, UnivariateSpline

################################
######## Vertical grid #########
################################
z_bot = 1300.0  # bottom of the model grid
z_dz = 5.0      # grid spacing
z_zgrid = -np.arange(-z_dz/2.0, z_bot + z_dz/2, z_dz)
z_npt = len(z_zgrid)	    # Number of points in z


################################
######## Vert Diffusion ########
################################
z_depthvar_Kv = 1 
z_Kv_param  = 2.0 * 1.701e-5 # constant vertical diffusion coefficient in m^2/s
# For sigmoidal Kv, use the following parameters
z_Kv_top = 0.70 * 2.0 * 1.701e-5
z_Kv_bot = 1.00 * 2.0 * 1.701e-5
z_Kv_flex = -250
z_Kv_width = 300
z_mld = 50  # Mixed layer depth
z_tmld = 1.0/86400.0 # timescale to fully mix the MLD (once per day recommended)

if z_depthvar_Kv == 1:
    z_Kv = 0.5*(z_Kv_top + z_Kv_bot) + 0.5*(z_Kv_top - z_Kv_bot) * np.tanh((z_zgrid - z_Kv_flex)/(0.5*z_Kv_width))
else:
    z_Kv = z_Kv_param * np.ones(1,np.length(z_zgrid))


################################
########### Upwelling ##########
################################
# Choose constant (=0) or depth-dependent (=1) upwelling velocity
# depth-dependent velocity requires a forcing file (set in z_d_initialize_DepParam.m)
z_depthvar_wup = 0 
z_wup_param = 3.171e-07 * 1    # 1.8395e-7 # m/s  # note: 10 m/y = 3.1710e-07 m/s

# Get upwelling profile by applying a spline to the model output w(z) profile
if z_depthvar_wup == 1: 
   dep = np.genfromtxt('vertical_CESM_depth.txt')
   vel = -np.genfromtxt('vertical_CESM_wvelETSP.txt')
   tdepth = -np.abs(dep)
   wupfcn = interp1d(tdepth,vel)
   z_wup = wupfcn(z_zgrid)/100.0 # convert to m/s
else:
   z_wup = z_wup_param * np.ones(len(z_zgrid))
   



