#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 10:52:47 2023

@author: pbuchanan
"""

import numpy as np
from numba import jit

@jit(nopython=True)
def bgc_sms(dil,\
            o2, no3, dfe, phy, zoo, det, cal, alk, dic,\
            i_o2, i_no3, i_dfe, i_phy, i_zoo, i_det, i_cal, i_alk, i_dic,\
            conserving):
   
    # TEMPORARY
    o2_loc = o2[1]      # µM
    no3_loc = no3[1]    # µM
    dfe_loc = dfe[1]    # µM
    phy_loc = phy[1]    # µM N
    zoo_loc = zoo[1]    # µM N
    det_loc = det[1]    # µM N
    cal_loc = cal[1]    # µM
    alk_loc = alk[1]    # µM
    dic_loc = dic[1]    # µM
    
    
    # Make tracer values zero if negative
    o2 = np.fmax(o2_loc, 0.0)
    no3 = np.fmax(no3_loc, 0.0)
    dfe = np.fmax(dfe_loc, 0.0)
    phy = np.fmax(phy_loc, 0.0)
    zoo = np.fmax(zoo_loc, 0.0)
    det = np.fmax(det_loc, 0.0)
    cal = np.fmax(cal_loc, 0.0)
    alk = np.fmax(alk_loc, 0.0)
    dic = np.fmax(dic_loc, 0.0)
    
    
    #----------------------------------------------------------------------
    # (0) Define important parameters
    #----------------------------------------------------------------------
    #########################
    ### Time and location ###
    #########################
    d2s = 86400.0
    eps = 1e-16
    dep = 50        # depth of chemostat in water column
    ############################
    ### SW radiation and PAR ###
    ############################
    p_PAR_k = 0.05          # attenuation coefficient in 1/m
    p_PAR_in = 200          # incoming photosynthetically active radiation (W/m2)
    p_alpha = 0.256 / d2s   # Initial slope of the PI curve (mmol N m-2 per mg Chl W sec)
    p_PAR_bio = 0.43        # fraction of shortwave radiation available to phytoplankton
    ###################
    ### Temperature ###
    ###################
    tc = 12.0
    p_auto_aT = 0.27 / d2s       # linear (vertical) scaler for autotrophy (/s)
    p_auto_bT = 1.066            # base coefficient determining temperature-sensitivity of autotrophy
    p_auto_cT = 1.0              # exponential scaler for autotrophy (per ºC)
    p_hete_aT = 1.0 / d2s       # linear (vertical) heterotrophic growth scaler (/s)
    p_hete_bT = 1.066            # base coefficient determining temperature-sensitivity of heterotrophy
    p_hete_cT = 1.0              # exponential scaler for heterotrophy (per ºC)
    def Tfunc(a,b,c,t):
        return a*b**(c*t)
    Tfunc_auto = Tfunc(p_auto_aT, p_auto_bT, p_auto_cT, tc)
    Tfunc_hete = Tfunc(p_hete_aT, p_hete_bT, p_hete_cT, tc)
    ############################
    ####### Phytoplannkton  ####
    ############################
    # DIC + NO3 + dFe --> POC + O2
    p_kphy_no3 = 0.7            # µM
    p_kphy_dfe = 0.1e-3         # µM
    p_phy_lmort = 0.04 / d2s    # linear mortality of phytoplankton (/s)
    p_phy_qmort = 0.25 / d2s    # quadratic mortality of phytoplankton (1 / (µM N * s))
    p_phy_CN = 106.0/16.0       # mol/mol
    p_phy_FeC = 7.1e-5          # mol/mol (based on Fe:P of 0.0075:1 (Moore et al 2015))
    p_phy_O2C = 172.0/106.0     # mol/mol
    ##########################
    ####### Zooplannkton  ####
    ##########################
    p_zoo_grz = 1.575           # scaler for rate of zooplankton grazing
    p_zoo_capcoef = 1.6 / d2s   # prey capture efficiency coefficient (m6 / (µM N)2 * s))
    p_zoo_qmort = 0.34 / d2s    # quadratic mortality of zooplankton (1 / (µM N * s))
    p_zoo_excre = 0.01 / d2s    # rate of excretion by zooplankton (/s)
    p_zoo_assim = 0.925         # zooplankton assimilation efficiency
    ######################
    ####### Detritus  ####
    ######################
    p_det_rem = 0.048 / d2s     # remineralisation rate of detritus (/s)
    ###################
    ####### CaCO3  ####
    ###################
    p_cal_rem = 0.001714 / d2s  # remineralisation rate of CaCO3 (/s)
    p_cal_fra = 0.062           # fraction of inorganic
    ##################
    ####### Iron  ####
    ##################
    p_dfe_scav = 0.00274 / d2s  # scavenging rate of dissolved iron (/s)
    p_dfe_deep = 0.6e-3         # background/deep value of dissolved iron (µM)
    

    #----------------------------------------------------------------------
    # (1) Primary production
    #----------------------------------------------------------------------
    # 1.1 Calculate PAR availability (W/m2)
    par = p_PAR_in * np.exp(-p_PAR_k * dep) * p_PAR_bio
    # 1.2 Calculate the temperature-dependent maximum growth rate (/s)
    phy_mumax = Tfunc_auto * 1
    # 1.3 Calculate growth limitation by light
    phy_lpar = 1.0 - np.exp( (-p_alpha * par) / phy_mumax ) 
    # 1.4 Calculate and apply nutrient limitation terms
    phy_lno3 = no3/(no3+p_kphy_no3)
    phy_ldfe = dfe/(dfe+p_kphy_dfe)
    # 1.5 Calculate the realised growth rate
    phy_mu = phy_mumax * phy_lpar * np.fmin(phy_lno3, phy_ldfe)
    # 1.6 Collect terms
    phy_no3upt = phy_mu * phy                                               # f11
    phy_dicupt = phy_no3upt * p_phy_CN
    phy_dfeupt = phy_dicupt * p_phy_FeC
    phy_oxyprd = phy_dicupt * p_phy_O2C
    

    #----------------------------------------------------------------------
    # (2) Grazing by zooplankton
    #----------------------------------------------------------------------
    # 1.1 Calculate temperature-dependent maximum grazing rate
    zoo_mumax = p_zoo_grz * Tfunc_hete
    # 1.2 Calculate prey capture rate function (/s)
    zoo_capt = p_zoo_capcoef * phy * phy 
    # 1.3 Calculate the realised grazing rate of zooplankton
    zoo_mu = zoo_mumax * zoo_capt / (zoo_mumax + zoo_capt)
        # The functional form of grazing creates increasing half-saturation 
        # values as maximum growth rates increase. It ensures that zooplankton
        # always have the same affinity for phytoplankton irrespective of temp
    # 1.4 Collect terms
    zoo_phygrz = zoo_mu * zoo                                               # f21
    zoo_zooexc = zoo * p_zoo_excre * Tfunc_hete                             # f31
    
    
    #----------------------------------------------------------------------
    # (3) Mortality terms
    #----------------------------------------------------------------------
    phy_lmort = p_phy_lmort * phy * Tfunc_hete                              # f22
    phy_qmort = p_phy_qmort * phy * phy      # Not temperature-dependent?   # f23
    zoo_qmort = p_zoo_qmort * zoo * zoo      # Not temperature-dependent?   # f32
    
    
    #----------------------------------------------------------------------
    # (4) Detritus and CaCO3
    #----------------------------------------------------------------------
    det_remin = p_det_rem * det * Tfunc_hete                                # f41
    cal_remin = p_cal_rem * cal                                             # f51
    
    
    #----------------------------------------------------------------------
    # (5)  Tracer equations (tendencies)
    #----------------------------------------------------------------------
    bgc_o2 = phy_oxyprd \
             - (phy_lmort + zoo_zooexc + det_remin) * p_phy_CN * p_phy_O2C
    
    bgc_no3 = (phy_lmort + zoo_zooexc + det_remin) - phy_no3upt
    
    bgc_dfe = (phy_lmort + zoo_zooexc + det_remin) * (p_phy_CN*p_phy_FeC) \
              - phy_dfeupt \
              - p_dfe_scav * np.fmax(0.0, dfe - p_dfe_deep)
    
    bgc_phy = phy_no3upt - (phy_lmort + phy_qmort + zoo_phygrz)
    
    bgc_zoo = zoo_phygrz * p_zoo_assim - (zoo_zooexc + zoo_qmort)
    
    bgc_det = (zoo_phygrz * (1.0-p_zoo_assim) + phy_qmort + zoo_qmort) \
              - det_remin
    
    bgc_cal = (zoo_phygrz * (1.0-p_zoo_assim) + phy_qmort + zoo_qmort)*p_cal_fra*p_phy_CN \
              - cal_remin
    
    bgc_alk = -bgc_no3 - bgc_cal * 2.0
    
    bgc_dic = (bgc_no3 * p_phy_CN) - bgc_cal
    
    
    ddt_o2  = dil * (i_o2 - o2) + bgc_o2
    ddt_no3 = dil * (i_no3 - no3) + bgc_no3
    ddt_dfe = dil * (i_dfe - dfe) + bgc_dfe
    ddt_phy = dil * (i_phy - phy) + bgc_phy
    ddt_zoo = dil * (i_zoo - zoo) + bgc_zoo
    ddt_det = dil * (i_det - det) + bgc_det
    ddt_cal = dil * (i_cal - cal) + bgc_cal
    ddt_alk = dil * (i_alk - alk) + bgc_alk
    ddt_dic = dil * (i_dic - dic) + bgc_dic
    
    
    return [ddt_o2, ddt_no3, ddt_dfe, \
            ddt_phy, ddt_zoo, ddt_det, \
            ddt_cal, ddt_alk, ddt_dic, \
            phy_mu, zoo_mu, phy_lmort, phy_qmort]
             
