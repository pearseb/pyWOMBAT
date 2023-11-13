#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 15:44:45 2023

@author: pbuchanan
"""


import numpy as np

def massb_n(no3, phy, dia, zoo, mes, det, \
            thresh):
    tot_n0 = no3[0,:] + phy[0,:] + dia[0,:] + zoo[0,:] + mes[0,:] + det[0,:]
    tot_n1 = no3[1,:] + phy[1,:] + dia[1,:] + zoo[1,:] + mes[1,:] + det[1,:]
    
    if np.abs((np.sum(tot_n0) - np.sum(tot_n1))) > thresh:
        error = 1
    else:
        error = 0
    
    return np.sum(tot_n0), np.sum(tot_n1), error


def massb_c(dic, cal, phy, dia, zoo, mes, det, C2N, \
            thresh):
    tot_c0 = dic[0,:] + cal[0,:] + (phy[0,:] + dia[0,:] + zoo[0,:] + mes[0,:] + det[0,:]) * C2N
    tot_c1 = dic[1,:] + cal[1,:] + (phy[1,:] + dia[1,:] + zoo[1,:] + mes[1,:] + det[1,:]) * C2N
    
    if np.abs((np.sum(tot_c0) - np.sum(tot_c1))) > thresh:
        error = 1
    else:
        error = 0
    
    return np.sum(tot_c0), np.sum(tot_c1), error


def massb_f(dfe, phy, dia, zoo, mes, det, Fe2N, \
            thresh):
    tot_f0 = dfe[0,:] + (phy[0,:] + dia[0,:] + zoo[0,:] + mes[0,:] + det[0,:]) * Fe2N
    tot_f1 = dfe[1,:] + (phy[1,:] + dia[1,:] + zoo[1,:] + mes[1,:] + det[1,:]) * Fe2N
    
    if np.abs((np.sum(tot_f0) - np.sum(tot_f1))) > thresh:
        error = 1
    else:
        error = 0
    
    return np.sum(tot_f0), np.sum(tot_f1), error

