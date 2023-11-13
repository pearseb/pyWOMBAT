#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 10:52:47 2023

@author: pbuchanan
"""

from numba import jit

@jit(nopython=True)
def advec_diff(dt, tra, tra_top, tra_bot, z_dz, z_wup, z_Kv):
    
    # For advection velocity and diffusion coefficient fixed in time, calculate here
    # terms for the numerical advection-diffusion solver. For time-dependent w and Kv
    # move these terms inside the time loop
    alpha = z_wup[0:-1] * dt / (2*z_dz)
    beta  = - dt / (2*z_dz) * (z_wup[0:-1] - z_wup[1::])
    gamma = z_Kv[0:-1] * dt / (z_dz)**2
    delta =   dt / (4*z_dz) * (z_Kv[0:-1] - z_Kv[1::])
    
    # Integration coefficients for the tracer at k,k+1,k-1 vertical levels:
    coeff1 = 1 + beta - 2 * gamma
    coeff2 =     alpha +  gamma - delta
    coeff3 =   - alpha +  gamma + delta
    
    #### Now calculate Explicit tracer concentrations
    #### Top boundary conditions
    #tra[1, 0] = tra_top
    #### Bottom boundary conditions
    tra[1,-1] = tra_bot
    
    #### advection and diffusion
    tra[1,0:-1] = tra[0,0:-1] * coeff1 + tra[0,1::] * coeff2 + tra[0,0:-1] * coeff3
    
    return tra


def mix_mld(dt, tra, z_zgrid, z_mld, z_tmld):
    #### mixing in the mixed layer
    mld_bool = -z_zgrid < z_mld
    tramld = sum(tra[0,mld_bool]) / len(tra[0,mld_bool])
    tra[1,mld_bool] = tra[1,mld_bool] + (tramld - tra[0,mld_bool]) * z_tmld * dt
    return tra

