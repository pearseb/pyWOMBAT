#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur Aug 31 2023

@author: pbuchanan
"""

import numpy as np
from numba import jit

#@jit(nopython=True)
def sink(tra, w, z_dz, z_zgrid):
   
    # Make det > 0.0 or equal to 0.0
    tra_loc = np.fmax(tra, 0.0)
    
    #----------------------------------------------------------------------
    # (0) Define important parameters
    #----------------------------------------------------------------------
    d2s = 86400.0
    ws = w / d2s
    
    #----------------------------------------------------------------------
    # (1) Sinking of Detritus
    #----------------------------------------------------------------------
    ### 1.1 Get rate of detritus sinking through bottom of box above (m/s * µM = mmol N m-2 s-1)
    adv_tra = np.zeros(np.shape(z_zgrid))
    for kk in np.arange(1,len(z_zgrid)):
        adv_tra[kk] = ws * tra_loc[kk-1]
    ### 1.2 Put this detritus into current box, accounting for losses to box below
    ddt_tra = np.zeros(np.shape(z_zgrid))
    for kk in np.arange(len(z_zgrid)-1):
        ddt_tra[kk] = (adv_tra[kk] - adv_tra[kk+1]) / z_dz
    ### 1.3 For final box, this only receives material from above if det == 0 at the bottom boundary
    adv_tra_bot = ws * tra_loc[-1]   # mmol N m-2 s-1
    ddt_tra[-1] = (adv_tra[-1] - adv_tra_bot) / z_dz
    ddt_tra2sed = adv_tra_bot/z_dz  # µM
    
    return [ddt_tra, ddt_tra2sed]


    
    
