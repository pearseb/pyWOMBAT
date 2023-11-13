#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur Aug 31 2023

@author: pbuchanan
"""

import numpy as np
from numba import jit

#@jit(nopython=True)
def sink(det, cal, wdet, wcal, z_dz, z_zgrid):
   
    # Make det > 0.0 or equal to 0.0
    det_loc = np.fmax(det, 0.0)
    cal_loc = np.fmax(cal, 0.0)
    
    #----------------------------------------------------------------------
    # (0) Define important parameters
    #----------------------------------------------------------------------
    d2s = 86400.0
    wsdet = wdet / d2s
    wscal = wcal / d2s
    
    #----------------------------------------------------------------------
    # (1) Sinking of Detritus
    #----------------------------------------------------------------------
    ### 1.1 Get rate of detritus sinking through bottom of box above (m/s * µM = mmol N m-2 s-1)
    adv_det = np.zeros(np.shape(z_zgrid))
    adv_cal = np.zeros(np.shape(z_zgrid))
    for kk in np.arange(1,len(z_zgrid)):
        adv_det[kk] = wsdet * det_loc[kk-1]
        adv_cal[kk] = wscal * cal_loc[kk-1]
    ### 1.2 Put this detritus into current box, accounting for losses to box below
    ddt_det = np.zeros(np.shape(z_zgrid))
    ddt_cal = np.zeros(np.shape(z_zgrid))
    for kk in np.arange(len(z_zgrid)-1):
        ddt_det[kk] = (adv_det[kk] - adv_det[kk+1]) / z_dz
        ddt_cal[kk] = (adv_cal[kk] - adv_cal[kk+1]) / z_dz
    ### 1.3 For final box, put the detritus in a sediment and account for this
    adv_det_bot = wsdet * det_loc[-1]   # mmol N m-2 s-1
    adv_cal_bot = wscal * cal_loc[-1]   # mmol N m-2 s-1
    ddt_det[-1] = (adv_det[-1] - adv_det_bot) / z_dz
    ddt_cal[-1] = (adv_cal[-1] - adv_cal_bot) / z_dz
    det2sed = adv_det_bot/z_dz  # µM
    cal2sed = adv_cal_bot/z_dz  # µM
    
    return [ddt_det, ddt_cal, det2sed, cal2sed]

    
#@jit(nopython=True)
def sinkFe(det, cal, wdet, wcal, z_dz, z_zgrid):
   
    # Make det > 0.0 or equal to 0.0
    det_loc = np.fmax(det, 0.0)
    cal_loc = np.fmax(cal, 0.0)
    
    #----------------------------------------------------------------------
    # (0) Define important parameters
    #----------------------------------------------------------------------
    d2s = 86400.0
    wsdet = wdet / d2s
    wscal = wcal / d2s
    
    #----------------------------------------------------------------------
    # (1) Sinking of Detritus
    #----------------------------------------------------------------------
    ### 1.1 Get rate of detritus sinking through bottom of box above (m/s * µM = mmol N m-2 s-1)
    adv_det = np.zeros(np.shape(z_zgrid))
    adv_cal = np.zeros(np.shape(z_zgrid))
    for kk in np.arange(1,len(z_zgrid)):
        adv_det[kk] = wsdet * det_loc[kk-1]
        adv_cal[kk] = wscal * cal_loc[kk-1]
    ### 1.2 Put this detritus into current box, accounting for losses to box below
    ddt_det = np.zeros(np.shape(z_zgrid))
    ddt_cal = np.zeros(np.shape(z_zgrid))
    for kk in np.arange(len(z_zgrid)-1):
        ddt_det[kk] = (adv_det[kk] - adv_det[kk+1]) / z_dz
        ddt_cal[kk] = (adv_cal[kk] - adv_cal[kk+1]) / z_dz
    ### 1.3 For final box, put the detritus in a sediment and account for this
    adv_det_bot = wsdet * det_loc[-1]   # mmol N m-2 s-1
    adv_cal_bot = wscal * cal_loc[-1]   # mmol N m-2 s-1
    ddt_det[-1] = (adv_det[-1] - adv_det_bot) / z_dz
    ddt_cal[-1] = (adv_cal[-1] - adv_cal_bot) / z_dz
    det2sed = adv_det_bot/z_dz  # µM
    cal2sed = adv_cal_bot/z_dz  # µM
    
    return [ddt_det, ddt_cal, det2sed, cal2sed]

    
    
