#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 15:44:45 2023

@author: pbuchanan
"""


import numpy as np

def massb_n(n2, no3, no2, nh4, n2o, don, rdon, phy, dia, aoa, nob, aox, nar, nai, nir, nos, fnar, fnai, fnir, fnos, zoo, mes, \
            p_aoa_CN, p_nob_CN, p_aox_CN, p_nar_CN, p_nai_CN, p_nir_CN, p_nos_CN, p_pom_CN, \
            thresh):
    tot_n0 = no3[0,:] + no2[0,:] + nh4[0,:] + n2o[0,:]*2 + n2[0,:]*2 + don[0,:] + rdon[0,:] + \
             phy[0,:] / p_pom_CN + dia[0,:] / p_pom_CN + \
             aoa[0,:] / p_aoa_CN + nob[0,:] / p_nob_CN + aox[0,:] / p_aox_CN + \
             nar[0,:] / p_nar_CN + nai[0,:] / p_nai_CN + nir[0,:] / p_nir_CN + \
             nos[0,:] / p_nos_CN + fnar[0,:] / p_nar_CN + fnai[0,:] / p_nai_CN + \
             fnir[0,:] / p_nir_CN + fnos[0,:] / p_nos_CN + zoo[0,:] / p_pom_CN + mes[0,:] / p_pom_CN
    tot_n1 = no3[1,:] + no2[1,:] + nh4[1,:] + n2o[1,:]*2 + n2[1,:]*2 + don[1,:] + rdon[1,:] + \
             phy[1,:] / p_pom_CN + dia[1,:] / p_pom_CN + \
             aoa[1,:] / p_aoa_CN + nob[1,:] / p_nob_CN + aox[1,:] / p_aox_CN + \
             nar[1,:] / p_nar_CN + nai[1,:] / p_nai_CN + nir[1,:] / p_nir_CN + \
             nos[1,:] / p_nos_CN + fnar[1,:] / p_nar_CN + fnai[1,:] / p_nai_CN + \
             fnir[1,:] / p_nir_CN + fnos[1,:] / p_nos_CN + zoo[1,:] / p_pom_CN + mes[1,:] / p_pom_CN
    
    if np.abs((np.sum(tot_n0) - np.sum(tot_n1))) > thresh:
        error = 1
    else:
        error = 0
    
    return np.sum(tot_n0), np.sum(tot_n1), error


def massb_p(po4, dop, rdop, phy, dia, aoa, nob, aox, nar, nai, nir, nos, fnar, fnai, fnir, fnos, zoo, mes, \
            p_aoa_CP, p_nob_CP, p_aox_CP, p_nar_CP, p_nai_CP, p_nir_CP, p_nos_CP, p_pom_CP, \
            thresh):
    tot_p0 = po4[0,:] + dop[0,:] + rdop[0,:] + \
             phy[0,:] / p_pom_CP + dia[0,:] / p_pom_CP + \
             aoa[0,:] / p_aoa_CP + nob[0,:] / p_nob_CP + aox[0,:] / p_aox_CP + \
             nar[0,:] / p_nar_CP + nai[0,:] / p_nai_CP + nir[0,:] / p_nir_CP + nos[0,:] / p_nos_CP + \
             fnar[0,:] / p_nar_CP + fnai[0,:] / p_nai_CP + fnir[0,:] / p_nir_CP + fnos[0,:] / p_nos_CP + \
             zoo[0,:] / p_pom_CP + mes[0,:] / p_pom_CP
    tot_p1 = po4[1,:] + dop[1,:] + rdop[1,:] + \
             phy[1,:] / p_pom_CP + dia[1,:] / p_pom_CP + \
             aoa[1,:] / p_aoa_CP + nob[1,:] / p_nob_CP + aox[1,:] / p_aox_CP + \
             nar[1,:] / p_nar_CP + nai[1,:] / p_nai_CP + nir[1,:] / p_nir_CP + nos[1,:] / p_nos_CP + \
             fnar[1,:] / p_nar_CP + fnai[1,:] / p_nai_CP + fnir[1,:] / p_nir_CP + fnos[1,:] / p_nos_CP + \
             zoo[1,:] / p_pom_CP + mes[1,:] / p_pom_CP
    
    if np.abs((np.sum(tot_p0) - np.sum(tot_p1))) > thresh:
        error = 1
    else:
        error = 0
    
    return np.sum(tot_p0), np.sum(tot_p1), error


def massb_c(dic, doc, rdoc, phy, dia, aoa, nob, aox, \
            nar, nai, nir, nos, fnar, fnai, fnir, fnos, zoo, mes, thresh):
    tot_c0 = phy[0,:] + dia[0,:] + aoa[0,:] + nob[0,:] + aox[0,:] + \
             nar[0,:] + nai[0,:] + nir[0,:] + nos[0,:] + \
             fnar[0,:]+ fnai[0,:]+ fnir[0,:]+ fnos[0,:] + \
             dic[0,:] + doc[0,:] + rdoc[0,:] + zoo[0,:] + mes[0,:]
    tot_c1 = phy[1,:] + dia[1,:] + aoa[1,:] + nob[1,:] + aox[1,:] + \
             nar[1,:] + nai[1,:] + nir[1,:] + nos[1,:] + \
             fnar[1,:]+ fnai[1,:]+ fnir[1,:]+ fnos[1,:] + \
             dic[1,:] + doc[1,:] + rdoc[1,:] + zoo[1,:] + mes[1,:]
    
    if np.abs((np.sum(tot_c0) - np.sum(tot_c1))) > thresh:
        error = 1
    else:
        error = 0
    
    return np.sum(tot_c0), np.sum(tot_c1), error
