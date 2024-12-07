#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 15:44:45 2023

@author: pbuchanan
"""

import numpy as np

def massb_n_0D(no3, phy, zoo, det, seddet, C2N, \
               thresh):
    tot_n0 = no3[0] + (phy[0] + zoo[0] + det[0] + seddet[0])/C2N
    tot_n1 = no3[1] + (phy[1] + zoo[1] + det[1] + seddet[1])/C2N
    
    if np.abs(tot_n0 - tot_n1) > thresh:
        error = 1
    else:
        error = 0
    
    return tot_n0, tot_n1, error


def massb_c_0D(dic, cal, phy, zoo, det, seddet, \
               thresh):
    tot_c0 = dic[0] + cal[0] + phy[0] + zoo[0] + det[0] + seddet[0]
    tot_c1 = dic[1] + cal[1] + phy[1] + zoo[1] + det[1] + seddet[1]
    
    if np.abs(tot_c0 - tot_c1) > thresh:
        error = 1
    else:
        error = 0
    
    return tot_c0, tot_c1, error


def massb_f_0D(dfe, phyfe, zoofe, detfe, seddetfe, \
               thresh):
    tot_f0 = dfe[0] + phyfe[0] + zoofe[0] + detfe[0] + seddetfe[0]
    tot_f1 = dfe[1] + phyfe[1] + zoofe[1] + detfe[1] + seddetfe[1]
    
    if np.abs(tot_f0 - tot_f1) > thresh:
        error = 1
    else:
        error = 0
    
    return tot_f0, tot_f1, error




def massb_n_1D(no3, phy, zoo, det, seddet, C2N, \
               thresh):
    tot_n0 = no3[0,:] + (phy[0,:] + zoo[0,:] + det[0,:] + seddet[0])/C2N
    tot_n1 = no3[1,:] + (phy[1,:] + zoo[1,:] + det[1,:] + seddet[1])/C2N
    
    if np.abs((np.sum(tot_n0) - np.sum(tot_n1))) > thresh:
        error = 1
    else:
        error = 0
    
    return np.sum(tot_n0), np.sum(tot_n1), error


def massb_c_1D(dic, cal, phy, zoo, det, seddet, \
               thresh):
    tot_c0 = dic[0,:] + cal[0,:] + phy[0,:] + zoo[0,:] + det[0,:] + seddet[0]
    tot_c1 = dic[1,:] + cal[1,:] + phy[1,:] + zoo[1,:] + det[1,:] + seddet[1]
    
    if np.abs((np.sum(tot_c0) - np.sum(tot_c1))) > thresh:
        error = 1
    else:
        error = 0
    
    return np.sum(tot_c0), np.sum(tot_c1), error


def massb_f_1D(dfe, phyfe, zoofe, detfe, seddetfe, \
               thresh):
    tot_f0 = dfe[0,:] + phyfe[0,:] + zoofe[0,:] + detfe[0,:] + seddetfe[0]
    tot_f1 = dfe[1,:] + phyfe[1,:] + zoofe[1,:] + detfe[1,:] + seddetfe[1]
    
    if np.abs((np.sum(tot_f0) - np.sum(tot_f1))) > thresh:
        error = 1
    else:
        error = 0
    
    return np.sum(tot_f0), np.sum(tot_f1), error

