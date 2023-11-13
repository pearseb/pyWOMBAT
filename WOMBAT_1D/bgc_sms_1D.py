#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 10:52:47 2023

@author: pbuchanan
"""

import numpy as np
from numba import jit

@jit(nopython=True)
def bgc_sms_1P1Z(o2, no3, dfe, phy, zoo, det, cal, alk, dic,\
                 rsds, tos, tob, z_dz, z_zgrid):
   
    # TEMPORARY
    o2_loc = o2[1,:]
    no3_loc = no3[1,:]
    dfe_loc = dfe[1,:]
    phy_loc = phy[1,:]
    zoo_loc = zoo[1,:]
    det_loc = det[1,:]
    cal_loc = cal[1,:]
    alk_loc = alk[1,:]
    dic_loc = dic[1,:]
    
    # Make tracer values zero if negative
    o2_loc = np.fmax(o2_loc, 0.0)
    no3_loc = np.fmax(no3_loc, 0.0)
    dfe_loc = np.fmax(dfe_loc, 0.0)
    phy_loc = np.fmax(phy_loc, 0.0)
    zoo_loc = np.fmax(zoo_loc, 0.0)
    det_loc = np.fmax(det_loc, 0.0)
    cal_loc = np.fmax(cal_loc, 0.0)
    alk_loc = np.fmax(alk_loc, 0.0)
    dic_loc = np.fmax(dic_loc, 0.0)
    
    
    #----------------------------------------------------------------------
    # (0) Define important parameters
    #----------------------------------------------------------------------
    #########################
    ### Time and location ###
    #########################
    d2s = 86400.0
    eps = 1e-16
    ############################
    ### SW radiation and PAR ###
    ############################
    p_PAR_k = 0.05          # attenuation coefficient in 1/m
    p_alpha = 0.256 / d2s   # Initial slope of the PI curve (mmol N m-2 per mg Chl W sec)
    p_PAR_bio = 0.43        # fraction of shortwave radiation available to phytoplankton
    ###################
    ### Temperature ###
    ###################
    tc = tos - (tos - tob) * (1-np.exp(p_PAR_k * z_zgrid))
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
    p_det_w   = 0.24 / d2s      # sinking rate of detritus (m/s)
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
    # (1) Primary production of nanophytoplankton
    #----------------------------------------------------------------------
    # 1.1 Calculate the temperature-dependent maximum growth rate (/s)
    phy_mumax = Tfunc_auto * 1
    # 1.2 Calculate growth limitation by light
    par = rsds * np.exp(p_PAR_k * z_zgrid) * p_PAR_bio  # z_zgrid is already negative 
    phy_lpar = 1.0 - np.exp( (-p_alpha * par) / phy_mumax ) 
    # 1.3 Calculate and apply nutrient limitation terms
    phy_lno3 = no3_loc/(no3_loc+p_kphy_no3)
    phy_ldfe = dfe_loc/(dfe_loc+p_kphy_dfe)
    # 1.4 Calculate the realised growth rate
    phy_mu = phy_mumax * phy_lpar * np.fmin(phy_lno3, phy_ldfe)
    # 1.5 Collect terms
    phy_no3upt = phy_mu * phy_loc                                               # f11
    phy_dicupt = phy_no3upt * p_phy_CN
    phy_dfeupt = phy_dicupt * p_phy_FeC
    phy_oxyprd = phy_dicupt * p_phy_O2C


    #----------------------------------------------------------------------
    # (2) Grazing by zooplankton
    #----------------------------------------------------------------------
    # 1.1 Calculate temperature-dependent maximum grazing rate
    zoo_mumax = p_zoo_grz * Tfunc_hete
    # 1.2 Calculate prey capture rate function (/s)
    zoo_capt = p_zoo_capcoef * phy_loc * phy_loc 
    # 1.3 Calculate the realised grazing rate of zooplankton
    zoo_mu = zoo_mumax * zoo_capt / (zoo_mumax + zoo_capt)
        # The functional form of grazing creates increasing half-saturation 
        # values as maximum growth rates increase. It ensures that zooplankton
        # always have the same affinity for phytoplankton irrespective of temp
    # 1.4 Collect terms
    zoo_phygrz = zoo_mu * zoo_loc                                               # f21
    zoo_zooexc = zoo_loc * p_zoo_excre * Tfunc_hete                             # f31
    
    
    #----------------------------------------------------------------------
    # (3) Mortality terms
    #----------------------------------------------------------------------
    phy_lmort = p_phy_lmort * phy_loc * Tfunc_hete                              # f22
    phy_qmort = p_phy_qmort * phy_loc * phy_loc      # Not temperature-dependent?   # f23
    zoo_qmort = p_zoo_qmort * zoo_loc * zoo_loc      # Not temperature-dependent?   # f32
    
    
    #----------------------------------------------------------------------
    # (4) Detritus and CaCO3
    #----------------------------------------------------------------------
    det_remin = p_det_rem * det_loc * Tfunc_hete                                # f41
    cal_remin = p_cal_rem * cal_loc                                             # f51


    #----------------------------------------------------------------------
    # (5)  Tracer equations (tendencies) Sources - Sinks
    #----------------------------------------------------------------------
    ddt_o2 = phy_oxyprd \
             - (phy_lmort + zoo_zooexc + det_remin) * p_phy_CN * p_phy_O2C
    
    ddt_no3 = (phy_lmort + zoo_zooexc + det_remin) - phy_no3upt
    
    ddt_dfe = (phy_lmort + zoo_zooexc + det_remin) * (p_phy_CN*p_phy_FeC) \
              - phy_dfeupt \
              - p_dfe_scav * np.fmax(0.0, dfe_loc - p_dfe_deep)
    
    ddt_phy = phy_no3upt - (phy_lmort + phy_qmort + zoo_phygrz)
    
    ddt_zoo = zoo_phygrz * p_zoo_assim - (zoo_zooexc + zoo_qmort)
    
    ddt_det = (zoo_phygrz * (1.0-p_zoo_assim) + phy_qmort + zoo_qmort) \
              - det_remin
    
    ddt_cal = (zoo_phygrz * (1.0-p_zoo_assim) + phy_qmort + zoo_qmort)*p_cal_fra*p_phy_CN \
              - cal_remin
    
    ddt_alk = -ddt_no3 - ddt_cal * 2.0
    
    ddt_dic = (ddt_no3 * p_phy_CN) - ddt_cal


    return [ddt_o2, ddt_no3, ddt_dfe, ddt_phy, ddt_zoo, ddt_det, ddt_cal, ddt_alk, ddt_dic, \
            phy_mu, zoo_mu, phy_lmort, phy_qmort]
    

@jit(nopython=True)
def bgc_sms_1P1Z_chl(o2, no3, dfe, phy, zoo, det, cal, alk, dic, chl, \
                     p_Chl_k, par, tos, tob, z_dz, z_mld, z_zgrid):
   
    eps = 1e-16
    
    # TEMPORARY
    o2_loc = o2[1,:]    # µM O2
    no3_loc = no3[1,:]  # µM NO3
    dfe_loc = dfe[1,:]  # µM dFe
    phy_loc = phy[1,:]  # µM N
    zoo_loc = zoo[1,:]  # µM N
    det_loc = det[1,:]  # µM N
    cal_loc = cal[1,:]  # µM CaCO3
    alk_loc = alk[1,:]  # µM Eq Alk
    dic_loc = dic[1,:]  # µM DIC
    chl_loc = chl[1,:]  # mg Chl
    
    # Make tracer values zero if negative
    o2_loc = np.fmax(o2_loc, 0.0)
    no3_loc = np.fmax(no3_loc, 0.0)
    dfe_loc = np.fmax(dfe_loc, 0.0)
    phy_loc = np.fmax(phy_loc, 0.0)
    zoo_loc = np.fmax(zoo_loc, 0.0)
    det_loc = np.fmax(det_loc, 0.0)
    cal_loc = np.fmax(cal_loc, 0.0)
    alk_loc = np.fmax(alk_loc, 0.0)
    dic_loc = np.fmax(dic_loc, 0.0)
    chl_loc = np.fmax(chl_loc, 0.0)
    
    
    #----------------------------------------------------------------------
    # (0) Define important parameters
    #----------------------------------------------------------------------
    #########################
    ### Time and location ###
    #########################
    d2s = 86400.0
    ############################
    ### SW radiation and PAR ###
    ############################
    p_PAR_k = 0.05          # attenuation coefficient in 1/m
    p_alpha = 2.0           # Initial slope of the PI curve (mmol C m-2 per mg Chl W sec)
    p_PAR_bio = 0.43        # fraction of shortwave radiation available to phytoplankton
    ###################
    ### Temperature ###
    ###################
    tc = tos - (tos - tob) * (1-np.exp(p_PAR_k * z_zgrid))
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
    p_phy_lmort = 0.04 / d2s    # linear mortality of phytoplankton (basal respiration) (/s)
    p_phy_qmort = 0.25 / d2s    # quadratic mortality of phytoplankton (1 / (µM N * s))
    p_phy_CN = 106.0/16.0       # mol/mol
    p_phy_FeC = 7.1e-5          # mol/mol (based on Fe:P of 0.0075:1 (Moore et al 2015))
    p_phy_O2C = 172.0/106.0     # mol/mol
    p_phy_minchlc = 0.004       # minimum chlorophyll : Carbon ratio
    p_phy_maxchlc = 0.033       # maximum chlorophyll : Carbon ratio (reduced by cooler temperatures)
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
    p_det_w   = 0.24 / d2s      # sinking rate of detritus (m/s)
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
    
    
    #####################
    #### Chlorophyll ####
    #####################
    # i)    Find daylight hours per day depending on latitude and day of year
    def daylight_hours(day, lat):
        day_angle = 2.0 * np.pi * (day-1)/365.0
        declination = 0.409 * np.sin(2*np.pi * day/365 - 1.39)
        cos_hour_angle = -np.tan(declination) * np.tan(np.deg2rad(lat))
        cos_hour_angle = np.fmax(np.fmin(cos_hour_angle, 1), -1)
        daylight_hours = 2 * np.rad2deg(np.arccos(cos_hour_angle)) / 15
        return daylight_hours
    day = 20
    lat = -40
    p_hrday = daylight_hours(day, lat)   # hours per day in the location of interest
    # ii)   Extract the attenuation coefficients for RGB light dependent on Chl
    zchl = np.fmax(0.01, np.fmin(10.0, chl_loc))
    ichl = ( 40 + 20 * np.log10(zchl) ).astype(np.int64)
    ek_blu = p_Chl_k[ichl,1]
    ek_gre = p_Chl_k[ichl,2]
    ek_red = p_Chl_k[ichl,3]
    # iii)  Using the attenuation coefficients, estimate the amount of RGB light available
    par_blu = np.zeros(len(z_zgrid)); par_gre = np.zeros(len(z_zgrid)); par_red = np.zeros(len(z_zgrid))
    par_blu[0] = par * np.exp(-0.5 * ek_blu[0] * z_dz)
    par_gre[0] = par * np.exp(-0.5 * ek_gre[0] * z_dz)
    par_red[0] = par * np.exp(-0.5 * ek_red[0] * z_dz)
    for kk in np.arange(1,len(z_zgrid)):
        par_blu[kk] = par_blu[kk-1] * np.exp(-ek_blu[kk-1] * z_dz)
        par_gre[kk] = par_gre[kk-1] * np.exp(-ek_gre[kk-1] * z_dz)
        par_red[kk] = par_red[kk-1] * np.exp(-ek_red[kk-1] * z_dz)
    # iv)   Find light available for Phytoplankton
    par_tot = par_blu + par_gre + par_red
    par_phy = 1.85 * par_blu + 0.69 * par_gre + 0.46 * par_red
    # v)    Calculate euphotic layer depth
    par_eup = par * p_PAR_bio * 0.01  # 1% light level
    par_euk = 0
    for kk in np.arange(0,len(z_zgrid)-1):
        if (par_tot[kk] >= par_eup):
            par_euk = kk+1
    z_eup = z_zgrid[par_euk] - (z_zgrid[par_euk] - z_zgrid[par_euk-1]) * 0.5
    z_eup = np.fmax(-300.0, z_eup)  # cannot be over 300 metres deep
    # vi)    Collect mean light over the mixed layer (total and phytoplankton specific)
    k_mld = np.argmin(np.abs((-z_zgrid - z_mld))) + 1
    par_tot_mldsum = 0.0
    par_phy_mldsum = 0.0
    par_z_mldsum = 0.0
    for kk in np.arange(0,len(z_zgrid)-1):
        if (-z_zgrid[kk+1] <= z_mld):
            par_tot_mldsum = par_tot_mldsum + par_tot[kk] * z_dz 
            par_phy_mldsum = par_phy_mldsum + par_phy[kk] * z_dz
            par_z_mldsum = par_z_mldsum + z_dz
    par_tot_mld = par_tot*1
    par_phy_mld = par_phy*1
    for kk in np.arange(0,len(z_zgrid)-1):
        if (-z_zgrid[kk+1] <= z_mld):
            z1_dep = 1.0 / par_z_mldsum
            par_tot_mld[kk] = par_tot_mldsum * z1_dep
            par_phy_mld[kk] = par_phy_mldsum * z1_dep
    # vii)  Impact of daylength on phytoplankton and chlorophyll
    chl_lday = np.ones(len(z_zgrid)) * eps
    phy_lday = np.ones(len(z_zgrid)) * eps
    for kk in np.arange(0,len(z_zgrid)-1):
        if (-z_zgrid[kk] <= z_mld):
            zval = np.fmax(1.0, p_hrday) * np.fmin(1.0, -z_eup/z_mld)
            chl_lday[kk] = zval / 24.0
            phy_lday[kk] = 1.5 * zval / (12.0 + zval)
            
    
    
    #----------------------------------------------------------------------
    # (1) Primary production of nanophytoplankton
    #----------------------------------------------------------------------
    # 1.1 Calculate the temperature-dependent maximum growth rate (/s)
    phy_mumax = Tfunc_auto * 1
    # 1.2 Calculate growth limitation by light
    #       i)   take into account chlorophyll to find initial P-I slope
    chlc_ratio = chl_loc / (phy_loc * p_phy_CN * 12.0)
    pislope = np.fmax(p_alpha * chlc_ratio, p_alpha * p_phy_minchlc)
    #       ii)  find chlorophyll production rate
    chl_pislope = pislope / (phy_mumax * d2s * chl_lday)   # decrease slope if growing fast, increase if not much daylight 
    chl_lpar = ( 1.0 - np.exp( -chl_pislope * par_phy_mld ) )
    chl_pro = phy_mumax * chl_lpar
    #       iii) find phytoplankton biomass production rate
    phy_pislope = pislope / ( (1.0/d2s + p_phy_lmort) * phy_lday * d2s ) # alter slope accounting for respiration
    phy_lpar = 1.0 - np.exp( (-phy_pislope * par_phy) )
    phy_pro = phy_mumax * phy_lday * phy_lpar
    # 1.3 Calculate nutrient limitation terms
    phy_lno3 = no3_loc/(no3_loc+p_kphy_no3)
    phy_ldfe = dfe_loc/(dfe_loc+p_kphy_dfe)
    phy_lnut = np.fmin(phy_lno3, phy_ldfe)
    # 1.4 Calculate the realised growth rate of phytoplankton after light and nutrient limitation
    phy_mu = phy_pro * phy_lnut
    # 1.5 Calculate the realised growth rate of chlorophyll
    chl_pro = d2s * chl_pro * (phy_mu * p_phy_CN) * phy_lnut
    chl_pro_min = p_phy_minchlc * (phy_mu * p_phy_CN * 12)
    phy_maxchlc = (p_phy_maxchlc / (1.0 - 1.14 / 43.4 * tc)) * (1.0 - 1.14 / 43.4 * 20.0)
    phy_maxchlc = np.fmin(p_phy_maxchlc, phy_maxchlc)
    chl_mu = chl_pro_min + (phy_maxchlc - p_phy_minchlc)*12*chl_pro / (pislope * par_phy_mld/chl_lday)
    # 1.5 Collect terms for phytoplankton
    phy_no3upt = phy_mu * phy_loc                                           # f11
    phy_dicupt = phy_no3upt * p_phy_CN
    phy_dfeupt = phy_dicupt * p_phy_FeC
    phy_oxyprd = phy_dicupt * p_phy_O2C


    #----------------------------------------------------------------------
    # (2) Grazing by zooplankton
    #----------------------------------------------------------------------
    # 1.1 Calculate temperature-dependent maximum grazing rate
    zoo_mumax = p_zoo_grz * Tfunc_hete
    # 1.2 Calculate prey capture rate function (/s)
    zoo_capt = p_zoo_capcoef * phy_loc * phy_loc
    # 1.3 Calculate the realised grazing rate of zooplankton
    zoo_mu = zoo_mumax * zoo_capt / (zoo_mumax + zoo_capt)
        # The functional form of grazing creates increasing half-saturation 
        # values as maximum growth rates increase. It ensures that zooplankton
        # always have the same affinity for phytoplankton irrespective of temp
    # 1.4 Collect terms
    zoo_phygrz = zoo_mu * zoo_loc                                           # f21
    zoo_zooexc = zoo_loc * p_zoo_excre * Tfunc_hete                         # f31
    
    
    #----------------------------------------------------------------------
    # (3) Mortality terms
    #----------------------------------------------------------------------
    phy_lmort = p_phy_lmort * phy_loc * Tfunc_hete                             # f22
    phy_qmort = p_phy_qmort * phy_loc * phy_loc # Not temperature-dependent?   # f23
    zoo_qmort = p_zoo_qmort * zoo_loc * zoo_loc # Not temperature-dependent?   # f32
    
    
    #----------------------------------------------------------------------
    # (4) Detritus and CaCO3
    #----------------------------------------------------------------------
    det_remin = p_det_rem * det_loc * Tfunc_hete                          # f41
    cal_remin = p_cal_rem * cal_loc                                       # f51


    #----------------------------------------------------------------------
    # (5)  Tracer equations (tendencies) Sources - Sinks
    #----------------------------------------------------------------------
    ddt_o2 = phy_oxyprd \
             - (phy_lmort + zoo_zooexc + det_remin) * p_phy_CN * p_phy_O2C
    
    ddt_no3 = (phy_lmort + zoo_zooexc + det_remin) - phy_no3upt
    
    ddt_dfe = (phy_lmort + zoo_zooexc + det_remin) * (p_phy_CN*p_phy_FeC) \
              - phy_dfeupt \
              - p_dfe_scav * np.fmax(0.0, dfe_loc - p_dfe_deep)
    
    ddt_phy = phy_no3upt - (phy_lmort + phy_qmort + zoo_phygrz)
    
    ddt_zoo = zoo_phygrz * p_zoo_assim - (zoo_zooexc + zoo_qmort)
    
    ddt_det = (zoo_phygrz * (1.0-p_zoo_assim) + phy_qmort + zoo_qmort) \
              - det_remin
    
    ddt_cal = (zoo_phygrz * (1.0-p_zoo_assim) + phy_qmort + zoo_qmort)*p_cal_fra*p_phy_CN \
              - cal_remin
    
    ddt_alk = -ddt_no3 - ddt_cal * 2.0
    
    ddt_dic = (ddt_no3 * p_phy_CN) - ddt_cal
    
    ddt_chl = chl_mu - chlc_ratio * (phy_lmort + phy_qmort + zoo_qmort)*p_phy_CN*12 


    return [ddt_o2, ddt_no3, ddt_dfe, ddt_phy, ddt_zoo, ddt_det, \
            ddt_cal, ddt_alk, ddt_dic, ddt_chl, \
            phy_mu, zoo_mu, phy_lmort, phy_qmort, chlc_ratio]
    

@jit(nopython=True)
def bgc_sms_2P2Z_chl(o2, no3, dfe, phy, dia, zoo, mes, det, cal, alk, dic, pchl, dchl, \
                     p_Chl_k, par, tos, tob, z_dz, z_mld, z_zgrid, day, lat):
   
    eps = 1e-16
    
    # TEMPORARY
    o2_loc = o2[1,:]    # µM O2
    no3_loc = no3[1,:]  # µM NO3
    dfe_loc = dfe[1,:]  # µM dFe
    phy_loc = phy[1,:]  # µM N
    dia_loc = dia[1,:]  # µM N
    zoo_loc = zoo[1,:]  # µM N
    mes_loc = mes[1,:]  # µM N
    det_loc = det[1,:]  # µM N
    cal_loc = cal[1,:]  # µM CaCO3
    alk_loc = alk[1,:]  # µM Eq Alk
    dic_loc = dic[1,:]  # µM DIC
    pchl_loc = pchl[1,:]  # mg Chl
    dchl_loc = dchl[1,:]  # mg Chl
    
    # Make tracer values zero if negative
    o2_loc = np.fmax(o2_loc, 0.0)
    no3_loc = np.fmax(no3_loc, 0.0)
    dfe_loc = np.fmax(dfe_loc, 0.0)
    phy_loc = np.fmax(phy_loc, eps)
    dia_loc = np.fmax(dia_loc, eps)
    zoo_loc = np.fmax(zoo_loc, eps)
    mes_loc = np.fmax(mes_loc, eps)
    det_loc = np.fmax(det_loc, 0.0)
    cal_loc = np.fmax(cal_loc, 0.0)
    alk_loc = np.fmax(alk_loc, 0.0)
    dic_loc = np.fmax(dic_loc, 0.0)
    pchl_loc = np.fmax(pchl_loc, eps*106./16.*0.02)
    dchl_loc = np.fmax(dchl_loc, eps*106./16.*0.02)
    
    
    #----------------------------------------------------------------------
    # (0) Define important parameters
    #----------------------------------------------------------------------
    #########################
    ### Time and location ###
    #########################
    d2s = 86400.0
    ############################
    ### SW radiation and PAR ###
    ############################
    p_PAR_k = 0.05          # attenuation coefficient in 1/m
    p_alpha = 2.0           # Initial slope of the PI curve (mmol C m-2 per mg Chl W sec)
    p_PAR_bio = 0.43        # fraction of shortwave radiation available to phytoplankton
    ###################
    ### Temperature ###
    ###################
    tc = tos - (tos - tob) * (1-np.exp(p_PAR_k * z_zgrid))
    p_auto_aT = 1.0 / d2s       # linear (vertical) scaler for autotrophy (/s)
    p_auto_bT = 1.066            # base coefficient determining temperature-sensitivity of autotrophy
    p_auto_cT = 1.0              # exponential scaler for autotrophy (per ºC)
    p_hete_aT = 0.95 / d2s       # linear (vertical) heterotrophic growth scaler (/s)
    p_hete_bT = 1.075            # base coefficient determining temperature-sensitivity of heterotrophy
    p_hete_cT = 1.00              # exponential scaler for heterotrophy (per ºC)
    def Tfunc(a,b,c,t):
        return a*b**(c*t)
    Tfunc_auto = Tfunc(p_auto_aT, p_auto_bT, p_auto_cT, tc)
    Tfunc_hete = Tfunc(p_hete_aT, p_hete_bT, p_hete_cT, tc)
    ###########################
    ####### Stoichiometry  ####
    ###########################
    p_CN = 106.0/16.0       # mol/mol
    p_FeC = 7.1e-5          # mol/mol (based on Fe:P of 0.0075:1 (Moore et al 2015))
    p_O2C = 172.0/106.0     # mol/mol
    #################################
    ####### Nano-Phytoplannkton  ####
    #################################
    # DIC + NO3 + dFe --> POC + O2
    p_kphy_no3 = 0.1            # µM
    p_kphy_dfe = 0.05e-3         # µM
    p_phy_lmort = 0.04 / d2s    # linear mortality of phytoplankton (basal respiration) (/s)
    p_phy_qmort = 0.25 / d2s    # quadratic mortality of phytoplankton (1 / (µM C * s))
    p_phy_minchlc = 0.004       # minimum chlorophyll : Carbon ratio
    p_phy_maxchlc = 0.033       # maximum chlorophyll : Carbon ratio (reduced by cooler temperatures)
    #####################
    ####### Diatoms  ####
    #####################
    # DIC + NO3 + dFe --> POC + O2
    p_kdia_no3 = 1.0            # µM
    p_kdia_dfe = 0.1e-3         # µM
    p_dia_lmort = 0.04 / d2s    # linear mortality of phytoplankton (basal respiration) (/s)
    p_dia_qmort = 0.25 / d2s    # quadratic mortality of phytoplankton (1 / (µM C * s))
    p_dia_minchlc = 0.004       # minimum chlorophyll : Carbon ratio
    p_dia_maxchlc = 0.05       # maximum chlorophyll : Carbon ratio (reduced by cooler temperatures)
    ##########################
    ####### Zooplannkton  ####
    ##########################
    p_zoo_grz = 1.0           # scaler for rate of zooplankton grazing
    p_zoo_capcoef = 1.0 / d2s   # prey capture efficiency coefficient (m6 / (µM C)2 * s))
    p_zoo_qmort = 0.1 / d2s    # quadratic mortality of zooplankton (1 / (µM C * s))
    p_zoo_excre = 0.01 / d2s    # rate of excretion by zooplankton (/s)
    p_zoo_assim = 0.925         # zooplankton assimilation efficiency
    ###############################
    ####### Meso-Zooplannkton  ####
    ###############################
    p_mes_grz = 0.25           # scaler for rate of zooplankton grazing
    p_mes_capcoef = 0.2 / d2s   # prey capture efficiency coefficient (m6 / (µM C)2 * s))
    p_mes_qmort = 0.5 / d2s    # quadratic mortality of zooplankton (1 / (µM C * s))
    p_mes_excre = 0.01 / d2s  # rate of excretion by zooplankton (/s)
    p_mes_assim = 0.925         # zooplankton assimilation efficiency
    ######################
    ####### Detritus  ####
    ######################
    p_det_rem = 0.048 / d2s     # remineralisation rate of detritus (/s)
    p_det_w   = 0.24 / d2s      # sinking rate of detritus (m/s)
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
    
    
    #####################
    #### Chlorophyll ####
    #####################
    # i)    Find daylight hours per day depending on latitude and day of year
    def daylight_hours(day, lat):
        day_angle = 2.0 * np.pi * (day-1)/365.0
        declination = 0.409 * np.sin(2*np.pi * day/365 - 1.39)
        cos_hour_angle = -np.tan(declination) * np.tan(np.deg2rad(lat))
        cos_hour_angle = np.fmax(np.fmin(cos_hour_angle, 1), -1)
        daylight_hours = 2 * np.rad2deg(np.arccos(cos_hour_angle)) / 15
        return daylight_hours
    p_hrday = daylight_hours(day, lat)   # hours per day in the location of interest
    # ii)   Extract the attenuation coefficients for RGB light dependent on Chl
    zchl = np.fmax(0.01, np.fmin(10.0, (pchl_loc+dchl_loc) ))
    ichl = ( 40 + 20 * np.log10(zchl) ).astype(np.int64)
    ek_blu = p_Chl_k[ichl,1]
    ek_gre = p_Chl_k[ichl,2]
    ek_red = p_Chl_k[ichl,3]
    # iii)  Using the attenuation coefficients, estimate the amount of RGB light available
    par_blu = np.zeros(len(z_zgrid)); par_gre = np.zeros(len(z_zgrid)); par_red = np.zeros(len(z_zgrid))
    par_blu[0] = par * 0.4 * 1./3 * np.exp(-0.5 * ek_blu[0] * z_dz)
    par_gre[0] = par * 0.4 * 1./3 * np.exp(-0.5 * ek_gre[0] * z_dz)
    par_red[0] = par * 0.4 * 1./3 * np.exp(-0.5 * ek_red[0] * z_dz)
    for kk in np.arange(1,len(z_zgrid)):
        par_blu[kk] = par_blu[kk-1] * np.exp(-ek_blu[kk-1] * z_dz)
        par_gre[kk] = par_gre[kk-1] * np.exp(-ek_gre[kk-1] * z_dz)
        par_red[kk] = par_red[kk-1] * np.exp(-ek_red[kk-1] * z_dz)
    # iv)   Find light available for Phytoplankton
    par_tot = par_blu + par_gre + par_red
    par_phy = 1.85 * par_blu + 0.69 * par_gre + 0.46 * par_red
    par_dia = 1.62 * par_blu + 0.74 * par_gre + 0.63 * par_red
    # v)    Calculate euphotic layer depth
    par_eup = par * p_PAR_bio * 0.01  # 1% light level
    par_euk = 0
    for kk in np.arange(0,len(z_zgrid)-1):
        if (par_tot[kk] >= par_eup):
            par_euk = kk+1
    z_eup = z_zgrid[par_euk] - (z_zgrid[par_euk] - z_zgrid[par_euk-1]) * 0.5
    z_eup = np.fmax(-300.0, z_eup)  # cannot be over 300 metres deep
    # vi)    Collect mean light over the mixed layer (total and phytoplankton specific)
    k_mld = np.argmin(np.abs((-z_zgrid - z_mld))) + 1
    par_tot_mldsum = 0.0
    par_phy_mldsum = 0.0
    par_dia_mldsum = 0.0
    par_z_mldsum = 0.0
    for kk in np.arange(0,len(z_zgrid)-1):
        if (-z_zgrid[kk+1] <= z_mld):
            par_tot_mldsum = par_tot_mldsum + par_tot[kk] * z_dz 
            par_phy_mldsum = par_phy_mldsum + par_phy[kk] * z_dz
            par_dia_mldsum = par_dia_mldsum + par_dia[kk] * z_dz
            par_z_mldsum = par_z_mldsum + z_dz
    par_tot_mld = par_tot*1
    par_phy_mld = par_phy*1
    par_dia_mld = par_dia*1
    for kk in np.arange(0,len(z_zgrid)-1):
        if (-z_zgrid[kk+1] <= z_mld):
            z1_dep = 1.0 / par_z_mldsum
            par_tot_mld[kk] = par_tot_mldsum * z1_dep
            par_phy_mld[kk] = par_phy_mldsum * z1_dep
            par_dia_mld[kk] = par_dia_mldsum * z1_dep
    # vii)  Impact of daylength on phytoplankton and chlorophyll
    chl_lday = np.ones(len(z_zgrid)) * 0.01
    lday = np.ones(len(z_zgrid)) * 0.01
    for kk in np.arange(0,len(z_zgrid)-1):
        zval = np.fmax(1.0, p_hrday)
        if (-z_zgrid[kk] <= z_mld):
            zval = zval * np.fmin(1.0, -z_eup/z_mld)
        chl_lday[kk] = zval / 24.0
        lday[kk] = 1.5 * zval / (12.0 + zval)
    
    
    #----------------------------------------------------------------------
    # (1) Primary production of nanophytoplankton
    #----------------------------------------------------------------------
    # 1.1 Calculate the temperature-dependent maximum growth rate (/s)
    phy_mumax = Tfunc_auto * 1
    dia_mumax = Tfunc_auto * 1.1
    # 1.2 Calculate growth limitation by light
    #       i)   take into account chlorophyll to find initial P-I slope
    phy_chlc_ratio = pchl_loc / (phy_loc * p_CN * 12.0)
    dia_chlc_ratio = dchl_loc / (dia_loc * p_CN * 12.0)
    phy_pislope = np.fmax(p_alpha * phy_chlc_ratio, p_alpha * p_phy_minchlc)
    dia_pislope = np.fmax(p_alpha * dia_chlc_ratio, p_alpha * p_dia_minchlc)
    #       ii)  find chlorophyll production rate
    phy_chl_pislope = phy_pislope / (phy_mumax * d2s * chl_lday)   # decrease slope if growing fast, increase if not much daylight 
    dia_chl_pislope = dia_pislope / (dia_mumax * d2s * chl_lday)   # decrease slope if growing fast, increase if not much daylight 
    phy_chl_lpar = ( 1.0 - np.exp( -phy_chl_pislope * par_phy_mld ) )
    dia_chl_lpar = ( 1.0 - np.exp( -dia_chl_pislope * par_dia_mld ) )
    phy_chl_pro = phy_mumax * phy_chl_lpar
    dia_chl_pro = dia_mumax * dia_chl_lpar
    #       iii) find phytoplankton biomass production rate
    phy_pislope2 = phy_pislope / ( (1.0/d2s + p_phy_lmort) * lday * d2s ) # alter slope accounting for respiration
    dia_pislope2 = dia_pislope / ( (1.0/d2s + p_dia_lmort) * lday * d2s ) # alter slope accounting for respiration
    phy_lpar = 1.0 - np.exp( (-phy_pislope2 * par_phy) )
    dia_lpar = 1.0 - np.exp( (-dia_pislope2 * par_dia) )
    phy_pro = phy_mumax * lday * phy_lpar
    dia_pro = dia_mumax * lday * dia_lpar
    
    print("Chl:C ratio =", phy_chlc_ratio[0], phy_chlc_ratio[10])
    print("PI slope =", phy_pislope[0], phy_pislope[10])
    print("PI slope (Chl) =", phy_chl_pislope[0], phy_chl_pislope[10])
    print("Light-limited phytoplankton production =", phy_pro[0]*d2s, phy_pro[10]*d2s)
    print("Light-limited chlorophyll production =", phy_chl_pro[0]*d2s, phy_chl_pro[10]*d2s)
    
    # 1.3 Calculate nutrient limitation terms
    phy_lno3 = no3_loc/(no3_loc+p_kphy_no3)
    phy_ldfe = dfe_loc/(dfe_loc+p_kphy_dfe)
    phy_lnut = np.fmin(phy_lno3, phy_ldfe)
    dia_lno3 = no3_loc/(no3_loc+p_kdia_no3)
    dia_ldfe = dfe_loc/(dfe_loc+p_kdia_dfe)
    dia_lnut = np.fmin(dia_lno3, dia_ldfe)
    # 1.4 Calculate the realised growth rate of phytoplankton after light and nutrient limitation
    phy_mu = phy_pro * phy_lnut
    dia_mu = dia_pro * dia_lnut
    # 1.5 Calculate the realised growth rate of chlorophyll
    phy_chl_pro = d2s * phy_mu * phy_loc * p_CN * 12 * (phy_chl_pro * phy_lnut)
    dia_chl_pro = d2s * dia_mu * dia_loc * p_CN * 12 * (dia_chl_pro * dia_lnut)
    phy_chl_pro_min = p_phy_minchlc * phy_mu * phy_loc * p_CN * 12
    dia_chl_pro_min = p_dia_minchlc * dia_mu * dia_loc * p_CN * 12
    phy_maxchlc = (p_phy_maxchlc / (1.0 - 1.14 / 43.4 * tc)) * (1.0 - 1.14 / 43.4 * 20.0)
    phy_maxchlc = np.fmin(p_phy_maxchlc, phy_maxchlc)
    dia_maxchlc = (p_dia_maxchlc / (1.0 - 1.14 / 43.4 * tc)) * (1.0 - 1.14 / 43.4 * 20.0)
    dia_maxchlc = np.fmin(p_dia_maxchlc, dia_maxchlc)
    phy_chl_mu = phy_chl_pro_min + (phy_maxchlc - p_phy_minchlc)*phy_chl_pro / (phy_pislope * par_phy_mld/chl_lday)
    dia_chl_mu = dia_chl_pro_min + (dia_maxchlc - p_dia_minchlc)*dia_chl_pro / (dia_pislope * par_dia_mld/chl_lday)
    # 1.5 Collect terms for phytoplankton
    phy_no3upt = phy_mu * phy_loc                                           # f11
    phy_dicupt = phy_no3upt * p_CN
    phy_dfeupt = phy_dicupt * p_FeC
    phy_oxyprd = phy_dicupt * p_O2C
    dia_no3upt = dia_mu * dia_loc                                           # f11
    dia_dicupt = dia_no3upt * p_CN
    dia_dfeupt = dia_dicupt * p_FeC
    dia_oxyprd = dia_dicupt * p_O2C
    
    print("Nutrient-limited phytoplankton production =", phy_mu[0]*d2s, phy_mu[10]*d2s)
    print("Nutrient-limited chlorophyll production =", phy_chl_pro[0]*d2s, phy_chl_pro[10]*d2s)
    print("Min chlorophyll production =", phy_chl_pro_min[0]*d2s, phy_chl_pro_min[10]*d2s)
    print("Max chlorophyll:C =", phy_maxchlc[0], phy_maxchlc[10])
    print("Realised chlorophyll production =", phy_chl_mu[0]*d2s, phy_chl_mu[10]*d2s)
    
    
    #----------------------------------------------------------------------
    # (2) Grazing by zooplankton
    #----------------------------------------------------------------------
    # 1.1 Calculate temperature-dependent maximum grazing rate
    zoo_mumax = p_zoo_grz * Tfunc_hete
    mes_mumax = p_mes_grz * Tfunc_hete
    # 1.2 Calculate prey capture rate function (/s)
    zoo_capt = p_zoo_capcoef * (phy_loc + det_loc) * (phy_loc + det_loc)
    mes_capt = p_mes_capcoef * (phy_loc + dia_loc + det_loc + zoo_loc) * (phy_loc + dia_loc + det_loc + zoo_loc)
    # 1.3 Calculate the realised grazing rate of zooplankton
    zoo_mu = zoo_mumax * zoo_capt / (zoo_mumax + zoo_capt)
    mes_mu = mes_mumax * mes_capt / (mes_mumax + mes_capt)
        # The functional form of grazing creates increasing half-saturation 
        # values as maximum growth rates increase. It ensures that zooplankton
        # always have the same affinity for phytoplankton irrespective of temp
    # 1.4 Collect terms
    zoo_phygrz = zoo_mu * zoo_loc * (phy_loc/(phy_loc+det_loc+eps))         # f21
    zoo_detgrz = zoo_mu * zoo_loc * (det_loc/(phy_loc+det_loc+eps))         # f21
    zoo_zooexc = zoo_loc * p_zoo_excre * Tfunc_hete                         # f31
    mes_phygrz = mes_mu * mes_loc * (phy_loc/(phy_loc+dia_loc+det_loc+zoo_loc+eps))         # f21
    mes_diagrz = mes_mu * mes_loc * (dia_loc/(phy_loc+dia_loc+det_loc+zoo_loc+eps))         # f21
    mes_detgrz = mes_mu * mes_loc * (det_loc/(phy_loc+dia_loc+det_loc+zoo_loc+eps))         # f21
    mes_zoogrz = mes_mu * mes_loc * (zoo_loc/(phy_loc+dia_loc+det_loc+zoo_loc+eps))         # f21
    mes_mesexc = mes_loc * p_mes_excre * Tfunc_hete                         # f31
    
    
    #----------------------------------------------------------------------
    # (3) Mortality terms
    #----------------------------------------------------------------------
    phy_lmort = p_phy_lmort * phy_loc * Tfunc_hete                             # f22
    phy_qmort = p_phy_qmort * phy_loc * phy_loc # Not temperature-dependent?   # f23
    dia_lmort = p_dia_lmort * dia_loc * Tfunc_hete                             # f22
    dia_qmort = p_dia_qmort * dia_loc * dia_loc # Not temperature-dependent?   # f23
    zoo_qmort = p_zoo_qmort * zoo_loc * zoo_loc # Not temperature-dependent?   # f32
    mes_qmort = p_mes_qmort * mes_loc * mes_loc # Not temperature-dependent?   # f32
    
    
    #----------------------------------------------------------------------
    # (4) Detritus and CaCO3
    #----------------------------------------------------------------------
    det_remin = p_det_rem * det_loc * Tfunc_hete * d2s                    # f41
    cal_remin = p_cal_rem * cal_loc                                       # f51
    
    
    #----------------------------------------------------------------------
    # (5)  Tracer equations (tendencies) Sources - Sinks
    #----------------------------------------------------------------------
    ddt_o2 = phy_oxyprd + dia_oxyprd \
             - (phy_lmort + dia_lmort + zoo_zooexc + mes_mesexc + det_remin) * p_CN * p_O2C
             
    ddt_no3 = (phy_lmort + dia_lmort + zoo_zooexc + mes_mesexc + det_remin) \
              - phy_no3upt - dia_no3upt
    
    ddt_dfe = (phy_lmort + dia_lmort + zoo_zooexc + mes_mesexc + det_remin) * (p_CN*p_FeC) \
              - phy_dfeupt - dia_dfeupt \
              - p_dfe_scav * np.fmax(0.0, dfe_loc - p_dfe_deep)
    
    ddt_phy = phy_no3upt - (phy_lmort + phy_qmort + zoo_phygrz + mes_phygrz)
    ddt_dia = dia_no3upt - (dia_lmort + dia_qmort + mes_diagrz)
    
    ddt_zoo = (zoo_phygrz + zoo_detgrz) * p_zoo_assim - (zoo_zooexc + zoo_qmort) - mes_zoogrz
    ddt_mes = (mes_phygrz + mes_diagrz + mes_detgrz + mes_zoogrz) * p_mes_assim - (mes_mesexc + mes_qmort)
    
    ddt_det = ((zoo_phygrz + zoo_detgrz) * (1.0-p_zoo_assim) \
              + (mes_phygrz + mes_diagrz + mes_detgrz + mes_zoogrz) * (1.0-p_mes_assim) \
              + phy_qmort + dia_qmort + zoo_qmort + mes_qmort ) \
              - det_remin - zoo_detgrz - mes_detgrz
    
    ddt_cal = ( zoo_phygrz * (1.0-p_zoo_assim) + mes_phygrz * (1.0-p_mes_assim) \
              + phy_qmort + zoo_qmort + mes_qmort ) * p_cal_fra*p_CN \
              - cal_remin
    
    ddt_alk = -ddt_no3 - ddt_cal * 2.0
    
    ddt_dic = (ddt_no3 * p_CN) - ddt_cal
    
    ddt_pchl = phy_chl_mu - phy_chlc_ratio * (phy_lmort + phy_qmort + zoo_phygrz + mes_phygrz)*p_CN*12 
    ddt_dchl = dia_chl_mu - dia_chlc_ratio * (dia_lmort + dia_qmort + mes_diagrz)*p_CN*12 

    
    #----------------------------------------------------------------------
    # (6)  Diagnostics for evaluation
    #----------------------------------------------------------------------
    pgi_zoo = (zoo_mu * zoo_loc) / (phy_loc + det_loc)
    pgi_mes = (mes_mu * mes_loc) / (phy_loc + dia_loc + det_loc + zoo_loc)
    

    return [ddt_o2, ddt_no3, ddt_dfe, ddt_phy, ddt_dia, ddt_zoo, ddt_mes, ddt_det, \
            ddt_cal, ddt_alk, ddt_dic, ddt_pchl, ddt_dchl, \
            pgi_zoo, pgi_mes, \
            phy_mu, dia_mu, zoo_mu, mes_mu, phy_lmort, dia_lmort, phy_qmort, dia_qmort, phy_chlc_ratio, dia_chlc_ratio]


@jit(nopython=True)
def bgc_sms_2P2Z_C(o2, no3, dfe, phy, dia, zoo, mes, det, cal, alk, dic, pchl, dchl, \
                   p_Chl_k, par, tos, tob, z_dz, z_mld, z_zgrid, day, lat):
   
    eps = 1e-16
    
    # TEMPORARY
    o2_loc = o2[1,:]    # µM O2
    no3_loc = no3[1,:]  # µM NO3
    dfe_loc = dfe[1,:]  # µM dFe
    phy_loc = phy[1,:]  # µM C
    dia_loc = dia[1,:]  # µM C
    zoo_loc = zoo[1,:]  # µM C
    mes_loc = mes[1,:]  # µM C
    det_loc = det[1,:]  # µM C
    cal_loc = cal[1,:]  # µM CaCO3
    alk_loc = alk[1,:]  # µM Eq Alk
    dic_loc = dic[1,:]  # µM DIC
    pchl_loc = pchl[1,:]  # mg Chl
    dchl_loc = dchl[1,:]  # mg Chl
    
    # Make tracer values zero if negative
    o2_loc = np.fmax(o2_loc, 0.0)
    no3_loc = np.fmax(no3_loc, 0.0)
    dfe_loc = np.fmax(dfe_loc, 0.0)
    phy_loc = np.fmax(phy_loc, eps)
    dia_loc = np.fmax(dia_loc, eps)
    zoo_loc = np.fmax(zoo_loc, eps)
    mes_loc = np.fmax(mes_loc, eps)
    det_loc = np.fmax(det_loc, 0.0)
    cal_loc = np.fmax(cal_loc, 0.0)
    alk_loc = np.fmax(alk_loc, 0.0)
    dic_loc = np.fmax(dic_loc, 0.0)
    pchl_loc = np.fmax(pchl_loc, eps*0.02)
    dchl_loc = np.fmax(dchl_loc, eps*0.02)
    
    
    #----------------------------------------------------------------------
    # (0) Define important parameters
    #----------------------------------------------------------------------
    #########################
    ### Time and location ###
    #########################
    d2s = 86400.0
    ############################
    ### SW radiation and PAR ###
    ############################
    p_PAR_k = 0.05          # attenuation coefficient in 1/m
    p_alpha = 2.0           # Initial slope of the PI curve (mmol C m-2 per mg Chl W sec)
    p_PAR_bio = 0.43        # fraction of shortwave radiation available to phytoplankton
    ###################
    ### Temperature ###
    ###################
    tc = tos - (tos - tob) * (1-np.exp(p_PAR_k * z_zgrid))
    p_auto_aT = 1.0 / d2s       # linear (vertical) scaler for autotrophy (/s)
    p_auto_bT = 1.066            # base coefficient determining temperature-sensitivity of autotrophy
    p_auto_cT = 1.0              # exponential scaler for autotrophy (per ºC)
    p_hete_aT = 0.95 / d2s       # linear (vertical) heterotrophic growth scaler (/s)
    p_hete_bT = 1.075            # base coefficient determining temperature-sensitivity of heterotrophy
    p_hete_cT = 1.00              # exponential scaler for heterotrophy (per ºC)
    def Tfunc(a,b,c,t):
        return a*b**(c*t)
    Tfunc_auto = Tfunc(p_auto_aT, p_auto_bT, p_auto_cT, tc)
    Tfunc_hete = Tfunc(p_hete_aT, p_hete_bT, p_hete_cT, tc)
    ###########################
    ####### Stoichiometry  ####
    ###########################
    p_CN = 106.0/16.0       # mol/mol
    p_FeC = 7.1e-5          # mol/mol (based on Fe:P of 0.0075:1 (Moore et al 2015))
    p_O2C = 172.0/106.0     # mol/mol
    #################################
    ####### Nano-Phytoplannkton  ####
    #################################
    # DIC + NO3 + dFe --> POC + O2
    p_kphy_no3 = 0.1            # µM
    p_kphy_dfe = 0.05e-3         # µM
    p_phy_lmort = 0.04 / d2s    # linear mortality of phytoplankton (basal respiration) (/s)
    p_phy_qmort = 0.25 / d2s    # quadratic mortality of phytoplankton (1 / (µM C * s))
    p_phy_minchlc = 0.004       # minimum chlorophyll : Carbon ratio
    p_phy_maxchlc = 0.033       # maximum chlorophyll : Carbon ratio (reduced by cooler temperatures)
    #####################
    ####### Diatoms  ####
    #####################
    # DIC + NO3 + dFe --> POC + O2
    p_kdia_no3 = 1.0            # µM
    p_kdia_dfe = 0.1e-3         # µM
    p_dia_lmort = 0.04 / d2s    # linear mortality of phytoplankton (basal respiration) (/s)
    p_dia_qmort = 0.25 / d2s    # quadratic mortality of phytoplankton (1 / (µM C * s))
    p_dia_minchlc = 0.004       # minimum chlorophyll : Carbon ratio
    p_dia_maxchlc = 0.05       # maximum chlorophyll : Carbon ratio (reduced by cooler temperatures)
    ##########################
    ####### Zooplannkton  ####
    ##########################
    p_zoo_grz = 1.0           # scaler for rate of zooplankton grazing
    p_zoo_capcoef = 1.0 / d2s   # prey capture efficiency coefficient (m6 / (µM C)2 * s))
    p_zoo_qmort = 0.1 / d2s    # quadratic mortality of zooplankton (1 / (µM C * s))
    p_zoo_excre = 0.01 / d2s    # rate of excretion by zooplankton (/s)
    p_zoo_assim = 0.925         # zooplankton assimilation efficiency
    ###############################
    ####### Meso-Zooplannkton  ####
    ###############################
    p_mes_grz = 0.25           # scaler for rate of zooplankton grazing
    p_mes_capcoef = 0.1 / d2s   # prey capture efficiency coefficient (m6 / (µM C)2 * s))
    p_mes_qmort = 0.5 / d2s    # quadratic mortality of zooplankton (1 / (µM C * s))
    p_mes_excre = 0.01 / d2s  # rate of excretion by zooplankton (/s)
    p_mes_assim = 0.925         # zooplankton assimilation efficiency
    ######################
    ####### Detritus  ####
    ######################
    p_det_rem = 0.048 / d2s     # remineralisation rate of detritus (/s)
    p_det_w   = 0.24 / d2s      # sinking rate of detritus (m/s)
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
    
    
    #####################
    #### Chlorophyll ####
    #####################
    # i)    Find daylight hours per day depending on latitude and day of year
    def daylight_hours(day, lat):
        day_angle = 2.0 * np.pi * (day-1)/365.0
        declination = 0.409 * np.sin(2*np.pi * day/365 - 1.39)
        cos_hour_angle = -np.tan(declination) * np.tan(np.deg2rad(lat))
        cos_hour_angle = np.fmax(np.fmin(cos_hour_angle, 1), -1)
        daylight_hours = 2 * np.rad2deg(np.arccos(cos_hour_angle)) / 15
        return daylight_hours
    p_hrday = daylight_hours(day, lat)   # hours per day in the location of interest
    # ii)   Extract the attenuation coefficients for RGB light dependent on Chl
    zchl = np.fmax(0.01, np.fmin(10.0, (pchl_loc+dchl_loc) ))
    ichl = ( 40 + 20 * np.log10(zchl) ).astype(np.int64)
    ek_blu = p_Chl_k[ichl,1]
    ek_gre = p_Chl_k[ichl,2]
    ek_red = p_Chl_k[ichl,3]
    # iii)  Using the attenuation coefficients, estimate the amount of RGB light available
    par_blu = np.zeros(len(z_zgrid)); par_gre = np.zeros(len(z_zgrid)); par_red = np.zeros(len(z_zgrid))
    par_blu[0] = par * 0.4 * 1./3 * np.exp(-0.5 * ek_blu[0] * z_dz)
    par_gre[0] = par * 0.4 * 1./3 * np.exp(-0.5 * ek_gre[0] * z_dz)
    par_red[0] = par * 0.4 * 1./3 * np.exp(-0.5 * ek_red[0] * z_dz)
    for kk in np.arange(1,len(z_zgrid)):
        par_blu[kk] = par_blu[kk-1] * np.exp(-ek_blu[kk-1] * z_dz)
        par_gre[kk] = par_gre[kk-1] * np.exp(-ek_gre[kk-1] * z_dz)
        par_red[kk] = par_red[kk-1] * np.exp(-ek_red[kk-1] * z_dz)
    # iv)   Find light available for Phytoplankton
    par_tot = par_blu + par_gre + par_red
    par_phy = 1.85 * par_blu + 0.69 * par_gre + 0.46 * par_red
    par_dia = 1.62 * par_blu + 0.74 * par_gre + 0.63 * par_red
    # v)    Calculate euphotic layer depth
    par_eup = par * p_PAR_bio * 0.01  # 1% light level
    par_euk = 0
    for kk in np.arange(0,len(z_zgrid)-1):
        if (par_tot[kk] >= par_eup):
            par_euk = kk+1
    z_eup = z_zgrid[par_euk] - (z_zgrid[par_euk] - z_zgrid[par_euk-1]) * 0.5
    z_eup = np.fmax(-300.0, z_eup)  # cannot be over 300 metres deep
    # vi)    Collect mean light over the mixed layer (total and phytoplankton specific)
    k_mld = np.argmin(np.abs((-z_zgrid - z_mld))) + 1
    par_tot_mldsum = 0.0
    par_phy_mldsum = 0.0
    par_dia_mldsum = 0.0
    par_z_mldsum = 0.0
    for kk in np.arange(0,len(z_zgrid)-1):
        if (-z_zgrid[kk+1] <= z_mld):
            par_tot_mldsum = par_tot_mldsum + par_tot[kk] * z_dz 
            par_phy_mldsum = par_phy_mldsum + par_phy[kk] * z_dz
            par_dia_mldsum = par_dia_mldsum + par_dia[kk] * z_dz
            par_z_mldsum = par_z_mldsum + z_dz
    par_tot_mld = par_tot*1
    par_phy_mld = par_phy*1
    par_dia_mld = par_dia*1
    for kk in np.arange(0,len(z_zgrid)-1):
        if (-z_zgrid[kk+1] <= z_mld):
            z1_dep = 1.0 / par_z_mldsum
            par_tot_mld[kk] = par_tot_mldsum * z1_dep
            par_phy_mld[kk] = par_phy_mldsum * z1_dep
            par_dia_mld[kk] = par_dia_mldsum * z1_dep
    # vii)  Impact of daylength on phytoplankton and chlorophyll
    chl_lday = np.ones(len(z_zgrid)) * 0.01
    lday = np.ones(len(z_zgrid)) * 0.01
    for kk in np.arange(0,len(z_zgrid)-1):
        zval = np.fmax(1.0, p_hrday)
        if (-z_zgrid[kk] <= z_mld):
            zval = zval * np.fmin(1.0, -z_eup/z_mld)
        chl_lday[kk] = zval / 24.0
        lday[kk] = 1.5 * zval / (12.0 + zval)
            
    
    
    #----------------------------------------------------------------------
    # (1) Primary production of nanophytoplankton
    #----------------------------------------------------------------------
    # 1.1 Calculate the temperature-dependent maximum growth rate (/s)
    phy_mumax = Tfunc_auto * 1
    dia_mumax = Tfunc_auto * 1.1
    # 1.2 Calculate growth limitation by light
    #       i)   take into account chlorophyll to find initial P-I slope
    phy_chlc_ratio = pchl_loc / (phy_loc * 12.0)
    dia_chlc_ratio = dchl_loc / (dia_loc * 12.0)
    phy_pislope = np.fmax(p_alpha * phy_chlc_ratio, p_alpha * p_phy_minchlc)
    dia_pislope = np.fmax(p_alpha * dia_chlc_ratio, p_alpha * p_dia_minchlc)
    #       ii)  find chlorophyll production rate
    phy_chl_pislope = phy_pislope / (phy_mumax * d2s * chl_lday)   # decrease slope if growing fast, increase if not much daylight 
    dia_chl_pislope = dia_pislope / (dia_mumax * d2s * chl_lday)   # decrease slope if growing fast, increase if not much daylight 
    phy_chl_lpar = ( 1.0 - np.exp( -phy_chl_pislope * par_phy_mld ) )
    dia_chl_lpar = ( 1.0 - np.exp( -dia_chl_pislope * par_dia_mld ) )
    phy_chl_pro = phy_mumax * phy_chl_lpar
    dia_chl_pro = dia_mumax * dia_chl_lpar
    #       iii) find phytoplankton biomass production rate
    phy_pislope2 = phy_pislope / ( (1.0/d2s + p_phy_lmort) * lday * d2s ) # alter slope accounting for respiration
    dia_pislope2 = dia_pislope / ( (1.0/d2s + p_dia_lmort) * lday * d2s ) # alter slope accounting for respiration
    phy_lpar = 1.0 - np.exp( (-phy_pislope2 * par_phy) )
    dia_lpar = 1.0 - np.exp( (-dia_pislope2 * par_dia) )
    phy_pro = phy_mumax * lday * phy_lpar
    dia_pro = dia_mumax * lday * dia_lpar
    
    print("Chl:C ratio =", phy_chlc_ratio[0], phy_chlc_ratio[10])
    print("PI slope =", phy_pislope[0], phy_pislope[10])
    print("PI slope (Chl) =", phy_chl_pislope[0], phy_chl_pislope[10])
    print("Light-limited phytoplankton production =", phy_pro[0]*d2s, phy_pro[10]*d2s)
    print("Light-limited chlorophyll production =", phy_chl_pro[0]*d2s, phy_chl_pro[10]*d2s)
    
    # 1.3 Calculate nutrient limitation terms
    phy_lno3 = no3_loc/(no3_loc+p_kphy_no3)
    phy_ldfe = dfe_loc/(dfe_loc+p_kphy_dfe)
    phy_lnut = np.fmin(phy_lno3, phy_ldfe)
    dia_lno3 = no3_loc/(no3_loc+p_kdia_no3)
    dia_ldfe = dfe_loc/(dfe_loc+p_kdia_dfe)
    dia_lnut = np.fmin(dia_lno3, dia_ldfe)
    # 1.4 Calculate the realised growth rate of phytoplankton after light and nutrient limitation
    phy_mu = phy_pro * phy_lnut
    dia_mu = dia_pro * dia_lnut
    # 1.5 Calculate the realised growth rate of chlorophyll
    phy_chl_pro = d2s * phy_mu * phy_loc * (phy_chl_pro * phy_lnut)
    dia_chl_pro = d2s * dia_mu * dia_loc * (dia_chl_pro * dia_lnut)
    phy_chl_pro_min = p_phy_minchlc * phy_mu * phy_loc * 12
    dia_chl_pro_min = p_dia_minchlc * dia_mu * dia_loc * 12
    phy_maxchlc = (p_phy_maxchlc / (1.0 - 1.14 / 43.4 * tc)) * (1.0 - 1.14 / 43.4 * 20.0)
    phy_maxchlc = np.fmin(p_phy_maxchlc, phy_maxchlc)
    dia_maxchlc = (p_dia_maxchlc / (1.0 - 1.14 / 43.4 * tc)) * (1.0 - 1.14 / 43.4 * 20.0)
    dia_maxchlc = np.fmin(p_dia_maxchlc, dia_maxchlc)
    phy_chl_mu = phy_chl_pro_min + (phy_maxchlc - p_phy_minchlc)*12*phy_chl_pro / (phy_pislope * par_phy_mld/chl_lday)
    dia_chl_mu = dia_chl_pro_min + (dia_maxchlc - p_dia_minchlc)*12*dia_chl_pro / (dia_pislope * par_dia_mld/chl_lday)
    # 1.5 Collect terms for phytoplankton
    phy_dicupt = phy_mu * phy_loc                                           # f11
    phy_no3upt = phy_dicupt / p_CN                                          # f11
    phy_dfeupt = phy_dicupt * p_FeC
    phy_oxyprd = phy_dicupt * p_O2C
    dia_dicupt = dia_mu * dia_loc                                           # f11
    dia_no3upt = dia_dicupt / p_CN                                           # f11
    dia_dfeupt = dia_dicupt * p_FeC
    dia_oxyprd = dia_dicupt * p_O2C
    
    print("Nutrient-limited phytoplankton production =", phy_mu[0]*d2s, phy_mu[10]*d2s)
    print("Nutrient-limited chlorophyll production =", phy_chl_pro[0]*d2s, phy_chl_pro[10]*d2s)
    print("Min chlorophyll production =", phy_chl_pro_min[0]*d2s, phy_chl_pro_min[10]*d2s)
    print("Max chlorophyll:C =", phy_maxchlc[0], phy_maxchlc[10])
    print("Realised chlorophyll production =", phy_chl_mu[0]*d2s, phy_chl_mu[10]*d2s)
    
    
    #----------------------------------------------------------------------
    # (2) Grazing by zooplankton
    #----------------------------------------------------------------------
    # 1.1 Calculate temperature-dependent maximum grazing rate
    zoo_mumax = p_zoo_grz * Tfunc_hete
    mes_mumax = p_mes_grz * Tfunc_hete
    # 1.2 Calculate prey capture rate function (/s)
    zoo_capt = p_zoo_capcoef * (phy_loc + det_loc) * (phy_loc + det_loc)
    mes_capt = p_mes_capcoef * (phy_loc + dia_loc + det_loc + zoo_loc) * (phy_loc + dia_loc + det_loc + zoo_loc)
    # 1.3 Calculate the realised grazing rate of zooplankton
    zoo_mu = zoo_mumax * zoo_capt / (zoo_mumax + zoo_capt)
    mes_mu = mes_mumax * mes_capt / (mes_mumax + mes_capt)
        # The functional form of grazing creates increasing half-saturation 
        # values as maximum growth rates increase. It ensures that zooplankton
        # always have the same affinity for phytoplankton irrespective of temp
    # 1.4 Collect terms
    zoo_phygrz = zoo_mu * zoo_loc * (phy_loc/(phy_loc+det_loc+eps))         # f21
    zoo_detgrz = zoo_mu * zoo_loc * (det_loc/(phy_loc+det_loc+eps))         # f21
    zoo_zooexc = zoo_loc * p_zoo_excre * Tfunc_hete                         # f31
    mes_phygrz = mes_mu * mes_loc * (phy_loc/(phy_loc+dia_loc+det_loc+zoo_loc+eps))         # f21
    mes_diagrz = mes_mu * mes_loc * (dia_loc/(phy_loc+dia_loc+det_loc+zoo_loc+eps))         # f21
    mes_detgrz = mes_mu * mes_loc * (det_loc/(phy_loc+dia_loc+det_loc+zoo_loc+eps))         # f21
    mes_zoogrz = mes_mu * mes_loc * (zoo_loc/(phy_loc+dia_loc+det_loc+zoo_loc+eps))         # f21
    mes_mesexc = mes_loc * p_mes_excre * Tfunc_hete                         # f31
    
    
    #----------------------------------------------------------------------
    # (3) Mortality terms
    #----------------------------------------------------------------------
    phy_lmort = p_phy_lmort * phy_loc * Tfunc_hete                             # f22
    phy_qmort = p_phy_qmort * phy_loc * phy_loc # Not temperature-dependent?   # f23
    dia_lmort = p_dia_lmort * dia_loc * Tfunc_hete                             # f22
    dia_qmort = p_dia_qmort * dia_loc * dia_loc # Not temperature-dependent?   # f23
    zoo_qmort = p_zoo_qmort * zoo_loc * zoo_loc # Not temperature-dependent?   # f32
    mes_qmort = p_mes_qmort * mes_loc * mes_loc # Not temperature-dependent?   # f32
    
    
    #----------------------------------------------------------------------
    # (4) Detritus and CaCO3
    #----------------------------------------------------------------------
    det_remin = p_det_rem * det_loc * Tfunc_hete * d2s                    # f41
    cal_remin = p_cal_rem * cal_loc                                       # f51


    #----------------------------------------------------------------------
    # (5)  Tracer equations (tendencies) Sources - Sinks
    #----------------------------------------------------------------------
    ddt_o2 = phy_oxyprd + dia_oxyprd \
             - (phy_lmort + dia_lmort + zoo_zooexc + mes_mesexc + det_remin) * p_O2C
             
    ddt_no3 = (phy_lmort + dia_lmort + zoo_zooexc + mes_mesexc + det_remin) / p_CN \
              - phy_no3upt - dia_no3upt
    
    ddt_dfe = (phy_lmort + dia_lmort + zoo_zooexc + mes_mesexc + det_remin) * p_FeC \
              - phy_dfeupt - dia_dfeupt \
              - p_dfe_scav * np.fmax(0.0, dfe_loc - p_dfe_deep)
    
    ddt_phy = phy_dicupt - (phy_lmort + phy_qmort + zoo_phygrz + mes_phygrz)
    ddt_dia = dia_dicupt - (dia_lmort + dia_qmort + mes_diagrz)
    
    ddt_zoo = (zoo_phygrz + zoo_detgrz) * p_zoo_assim - (zoo_zooexc + zoo_qmort) - mes_zoogrz
    ddt_mes = (mes_phygrz + mes_diagrz + mes_detgrz + mes_zoogrz) * p_mes_assim - (mes_mesexc + mes_qmort)
    
    ddt_det = ((zoo_phygrz + zoo_detgrz) * (1.0-p_zoo_assim) \
              + (mes_phygrz + mes_diagrz + mes_detgrz + mes_zoogrz) * (1.0-p_mes_assim) \
              + phy_qmort + dia_qmort + zoo_qmort + mes_qmort ) \
              - det_remin - zoo_detgrz - mes_detgrz
    
    ddt_cal = ( zoo_phygrz * (1.0-p_zoo_assim) + mes_phygrz * (1.0-p_mes_assim) \
              + phy_qmort + zoo_qmort + mes_qmort ) * p_cal_fra \
              - cal_remin
    
    ddt_alk = -ddt_no3 - ddt_cal * 2.0
    
    ddt_dic = (ddt_no3 * p_CN) - ddt_cal
    
    ddt_pchl = phy_chl_mu - phy_chlc_ratio * (phy_lmort + phy_qmort + zoo_phygrz + mes_phygrz)*12 
    ddt_dchl = dia_chl_mu - dia_chlc_ratio * (dia_lmort + dia_qmort + mes_diagrz)*12 

    
    #----------------------------------------------------------------------
    # (6)  Diagnostics for evaluation
    #----------------------------------------------------------------------
    pgi_zoo = (zoo_mu * zoo_loc) / (phy_loc + det_loc)
    pgi_mes = (mes_mu * mes_loc) / (phy_loc + dia_loc + det_loc + zoo_loc)
    

    return [ddt_o2, ddt_no3, ddt_dfe, ddt_phy, ddt_dia, ddt_zoo, ddt_mes, ddt_det, \
            ddt_cal, ddt_alk, ddt_dic, ddt_pchl, ddt_dchl, \
            pgi_zoo, pgi_mes, \
            phy_mu, dia_mu, zoo_mu, mes_mu, phy_lmort, dia_lmort, phy_qmort, dia_qmort, phy_chlc_ratio, dia_chlc_ratio]
    

@jit(nopython=True)
def bgc_sms_2P2Z_varFe(o2, no3, dfe, \
                       phy, dia, zoo, mes, \
                       det, cal, alk, dic, \
                       pchl, dchl, \
                       phyfe, diafe, detfe, zoofe, mesfe, \
                       p_Chl_k, par, tos, tob, z_dz, z_mld, z_zgrid, day, lat):
   
    eps = 1e-16
    
    # TEMPORARY
    o2_loc = o2[1,:]    # µM O2
    no3_loc = no3[1,:]  # µM NO3
    dfe_loc = dfe[1,:]  # µM dFe
    phy_loc = phy[1,:]  # µM C
    dia_loc = dia[1,:]  # µM C
    zoo_loc = zoo[1,:]  # µM C
    mes_loc = mes[1,:]  # µM C
    det_loc = det[1,:]  # µM C
    cal_loc = cal[1,:]  # µM CaCO3
    alk_loc = alk[1,:]  # µM Eq Alk
    dic_loc = dic[1,:]  # µM DIC
    pchl_loc = pchl[1,:]  # mg Chl
    dchl_loc = dchl[1,:]  # mg Chl
    phyfe_loc = phyfe[1,:]  # µM Fe
    diafe_loc = diafe[1,:]  # µM Fe
    detfe_loc = detfe[1,:]  # µM Fe
    zoofe_loc = zoofe[1,:]  # µM Fe
    mesfe_loc = mesfe[1,:]  # µM Fe
    
    # Make tracer values zero if negative
    o2_loc = np.fmax(o2_loc, 0.0)
    no3_loc = np.fmax(no3_loc, 0.0)
    dfe_loc = np.fmax(dfe_loc, 0.0)
    phy_loc = np.fmax(phy_loc, eps)
    dia_loc = np.fmax(dia_loc, eps)
    zoo_loc = np.fmax(zoo_loc, eps)
    mes_loc = np.fmax(mes_loc, eps)
    det_loc = np.fmax(det_loc, 0.0)
    cal_loc = np.fmax(cal_loc, 0.0)
    alk_loc = np.fmax(alk_loc, 0.0)
    dic_loc = np.fmax(dic_loc, 0.0)
    
    
    #----------------------------------------------------------------------
    # (0) Define important parameters
    #----------------------------------------------------------------------
    #########################
    ### Time and location ###
    #########################
    d2s = 86400.0
    ############################
    ### SW radiation and PAR ###
    ############################
    p_PAR_k = 0.05          # attenuation coefficient in 1/m
    p_alpha = 2.0           # Initial slope of the PI curve (mmol C m-2 per mg Chl W sec)
    p_PAR_bio = 0.43        # fraction of shortwave radiation available to phytoplankton
    ###################
    ### Temperature ###
    ###################
    tc = tos - (tos - tob) * (1-np.exp(p_PAR_k * z_zgrid))
    p_auto_aT = 1.0 / d2s       # linear (vertical) scaler for autotrophy (/s)
    p_auto_bT = 1.066            # base coefficient determining temperature-sensitivity of autotrophy
    p_auto_cT = 1.0              # exponential scaler for autotrophy (per ºC)
    p_hete_aT = 0.95 / d2s       # linear (vertical) heterotrophic growth scaler (/s)
    p_hete_bT = 1.075            # base coefficient determining temperature-sensitivity of heterotrophy
    p_hete_cT = 1.00              # exponential scaler for heterotrophy (per ºC)
    def Tfunc(a,b,c,t):
        return a*b**(c*t)
    Tfunc_auto = Tfunc(p_auto_aT, p_auto_bT, p_auto_cT, tc)
    Tfunc_hete = Tfunc(p_hete_aT, p_hete_bT, p_hete_cT, tc)
    ###########################
    ####### Stoichiometry  ####
    ###########################
    p_CN = 106.0/16.0       # mol/mol
    p_O2C = 172.0/106.0     # mol/mol
    #################################
    ####### Nano-Phytoplannkton  ####
    #################################
    # DIC + NO3 + dFe --> POC + O2
    p_kphy_no3 = 0.1            # µM
    p_kphy_dfe = 0.05e-3         # µM
    p_phy_lmort = 0.04 / d2s    # linear mortality of phytoplankton (basal respiration) (/s)
    p_phy_qmort = 0.25 / d2s    # quadratic mortality of phytoplankton (1 / (µM C * s))
    p_phy_minChlC = 0.004       # minimum chlorophyll : Carbon ratio
    p_phy_maxChlC = 0.033       # maximum chlorophyll : Carbon ratio (reduced by cooler temperatures)
    #####################
    ####### Diatoms  ####
    #####################
    # DIC + NO3 + dFe --> POC + O2
    p_kdia_no3 = 1.0            # µM
    p_kdia_dfe = 0.1e-3         # µM
    p_dia_lmort = 0.04 / d2s    # linear mortality of phytoplankton (basal respiration) (/s)
    p_dia_qmort = 0.25 / d2s    # quadratic mortality of phytoplankton (1 / (µM C * s))
    p_dia_minChlC = 0.004       # minimum chlorophyll : Carbon ratio
    p_dia_maxChlC = 0.05       # maximum chlorophyll : Carbon ratio (reduced by cooler temperatures)
    ##########################
    ####### Zooplannkton  ####
    ##########################
    p_zoo_grz = 1.0           # scaler for rate of zooplankton grazing
    p_zoo_capcoef = 1.0 / d2s   # prey capture efficiency coefficient (m6 / (µM C)2 * s))
    p_zoo_qmort = 0.1 / d2s    # quadratic mortality of zooplankton (1 / (µM C * s))
    p_zoo_excre = 0.01 / d2s    # rate of excretion by zooplankton (/s)
    p_zoo_assim = 0.925         # zooplankton assimilation efficiency
    ###############################
    ####### Meso-Zooplannkton  ####
    ###############################
    p_mes_grz = 0.25           # scaler for rate of zooplankton grazing
    p_mes_capcoef = 0.1 / d2s   # prey capture efficiency coefficient (m6 / (µM C)2 * s))
    p_mes_qmort = 0.5 / d2s    # quadratic mortality of zooplankton (1 / (µM C * s))
    p_mes_excre = 0.01 / d2s  # rate of excretion by zooplankton (/s)
    p_mes_assim = 0.925         # zooplankton assimilation efficiency
    ######################
    ####### Detritus  ####
    ######################
    p_det_rem = 0.048 / d2s     # remineralisation rate of detritus (/s)
    p_det_w   = 0.24 / d2s      # sinking rate of detritus (m/s)
    ###################
    ####### CaCO3  ####
    ###################
    p_cal_rem = 0.001714 / d2s  # remineralisation rate of CaCO3 (/s)
    p_cal_fra = 0.062           # fraction of inorganic
    ##################
    ####### Iron  ####
    ##################
    p_phy_maxFeC = 40e-6        # Maximum Fe:C quota for nanophytoplankton
    p_dia_maxFeC = 40e-6        # Maximum Fe:C quota for diatoms
    p_phy_optFeC = 7e-6         # Optimal Fe:C quota for nanophytoplankton
    p_dia_optFeC = 7e-6         # Optimal Fe:C quota for diatoms
    p_fe_coef1 = 0.0016 / 55.85
    p_fe_coef2 = 1.21e-5 * 14.0 / 55.85 / 7.625 * 0.5 * 1.5
    p_fe_coef3 = 1.15e-4 * 14.0 / 55.85 / 7.625 * 0.5
    p_dfe_scav = 0.00274 / d2s  # scavenging rate of dissolved iron (/s)
    p_dfe_deep = 0.2e-3         # background/deep value of dissolved iron (µM)
    p_zoo_optFeC = 60.0e-6         # mol/mol (Baines et al. 2016)
    p_mes_optFeC = 20.0e-6         # mol/mol (Baines et al. 2016)
    
    
    # Initialise the Chl and Fe in biomass
    pchl_loc = np.fmax(pchl_loc, eps*p_phy_minChlC)
    dchl_loc = np.fmax(dchl_loc, eps*p_dia_minChlC)
    phyfe_loc = np.fmax(phyfe_loc, eps*p_phy_optFeC)
    diafe_loc = np.fmax(diafe_loc, eps*p_dia_optFeC)
    detfe_loc = np.fmax(detfe_loc, eps*p_phy_optFeC)
    zoofe_loc = np.fmax(zoofe_loc, eps*p_zoo_optFeC)
    mesfe_loc = np.fmax(mesfe_loc, eps*p_mes_optFeC)
    
    # Get important ratios
    phy_ChlC = pchl_loc / (phy_loc * 12.0)
    dia_ChlC = dchl_loc / (dia_loc * 12.0)
    phy_FeC = phyfe_loc / phy_loc
    dia_FeC = diafe_loc / dia_loc
    det_FeC = detfe_loc / det_loc
    zoo_FeC = zoofe_loc / zoo_loc
    mes_FeC = mesfe_loc / mes_loc
    
    
    #####################
    #### Chlorophyll ####
    #####################
    # i)    Find daylight hours per day depending on latitude and day of year
    def daylight_hours(day, lat):
        day_angle = 2.0 * np.pi * (day-1)/365.0
        declination = 0.409 * np.sin(2*np.pi * day/365 - 1.39)
        cos_hour_angle = -np.tan(declination) * np.tan(np.deg2rad(lat))
        cos_hour_angle = np.fmax(np.fmin(cos_hour_angle, 1), -1)
        daylight_hours = 2 * np.rad2deg(np.arccos(cos_hour_angle)) / 15
        return daylight_hours
    p_hrday = daylight_hours(day, lat)   # hours per day in the location of interest
    # ii)   Extract the attenuation coefficients for RGB light dependent on Chl
    zchl = np.fmax(0.01, np.fmin(10.0, (pchl_loc+dchl_loc) ))
    ichl = ( 40 + 20 * np.log10(zchl) ).astype(np.int64)
    ek_blu = p_Chl_k[ichl,1]
    ek_gre = p_Chl_k[ichl,2]
    ek_red = p_Chl_k[ichl,3]
    # iii)  Using the attenuation coefficients, estimate the amount of RGB light available
    par_blu = np.zeros(len(z_zgrid)); par_gre = np.zeros(len(z_zgrid)); par_red = np.zeros(len(z_zgrid))
    par_blu[0] = par * 0.4 * 1./3 * np.exp(-0.5 * ek_blu[0] * z_dz)
    par_gre[0] = par * 0.4 * 1./3 * np.exp(-0.5 * ek_gre[0] * z_dz)
    par_red[0] = par * 0.4 * 1./3 * np.exp(-0.5 * ek_red[0] * z_dz)
    for kk in np.arange(1,len(z_zgrid)):
        par_blu[kk] = par_blu[kk-1] * np.exp(-ek_blu[kk-1] * z_dz)
        par_gre[kk] = par_gre[kk-1] * np.exp(-ek_gre[kk-1] * z_dz)
        par_red[kk] = par_red[kk-1] * np.exp(-ek_red[kk-1] * z_dz)
    # iv)   Find light available for Phytoplankton
    par_tot = par_blu + par_gre + par_red
    par_phy = 1.85 * par_blu + 0.69 * par_gre + 0.46 * par_red
    par_dia = 1.62 * par_blu + 0.74 * par_gre + 0.63 * par_red
    # v)    Calculate euphotic layer depth
    par_eup = par * p_PAR_bio * 0.01  # 1% light level
    par_euk = 0
    for kk in np.arange(0,len(z_zgrid)-1):
        if (par_tot[kk] >= par_eup):
            par_euk = kk+1
    z_eup = z_zgrid[par_euk] - (z_zgrid[par_euk] - z_zgrid[par_euk-1]) * 0.5
    z_eup = np.fmax(-300.0, z_eup)  # cannot be over 300 metres deep
    # vi)    Collect mean light over the mixed layer (total and phytoplankton specific)
    k_mld = np.argmin(np.abs((-z_zgrid - z_mld))) + 1
    par_tot_mldsum = 0.0
    par_phy_mldsum = 0.0
    par_dia_mldsum = 0.0
    par_z_mldsum = 0.0
    for kk in np.arange(0,len(z_zgrid)-1):
        if (-z_zgrid[kk+1] <= z_mld):
            par_tot_mldsum = par_tot_mldsum + par_tot[kk] * z_dz 
            par_phy_mldsum = par_phy_mldsum + par_phy[kk] * z_dz
            par_dia_mldsum = par_dia_mldsum + par_dia[kk] * z_dz
            par_z_mldsum = par_z_mldsum + z_dz
    par_tot_mld = par_tot*1
    par_phy_mld = par_phy*1
    par_dia_mld = par_dia*1
    for kk in np.arange(0,len(z_zgrid)-1):
        if (-z_zgrid[kk+1] <= z_mld):
            z1_dep = 1.0 / par_z_mldsum
            par_tot_mld[kk] = par_tot_mldsum * z1_dep
            par_phy_mld[kk] = par_phy_mldsum * z1_dep
            par_dia_mld[kk] = par_dia_mldsum * z1_dep
    # vii)  Impact of daylength on phytoplankton and chlorophyll
    chl_lday = np.ones(len(z_zgrid)) * 0.01
    lday = np.ones(len(z_zgrid)) * 0.01
    for kk in np.arange(0,len(z_zgrid)-1):
        zval = np.fmax(1.0, p_hrday)
        if (-z_zgrid[kk] <= z_mld):
            zval = zval * np.fmin(1.0, -z_eup/z_mld)
        chl_lday[kk] = zval / 24.0
        lday[kk] = 1.5 * zval / (12.0 + zval)
            
    
    
    #----------------------------------------------------------------------
    # (1) Primary production of nanophytoplankton
    #----------------------------------------------------------------------
    # 1.1 Calculate the temperature-dependent maximum growth rate (/s)
    phy_mumax = Tfunc_auto * 1
    dia_mumax = Tfunc_auto * 1.1
    # 1.2 Calculate growth limitation by light
    #       i)   take into account chlorophyll to find initial P-I slope
    phy_pislope = np.fmax(p_alpha * phy_ChlC, p_alpha * p_phy_minChlC)
    dia_pislope = np.fmax(p_alpha * dia_ChlC, p_alpha * p_dia_minChlC)
    #       ii)  find chlorophyll production rate
    phy_chl_pislope = phy_pislope / (phy_mumax * d2s * chl_lday)   # decrease slope if growing fast, increase if not much daylight 
    dia_chl_pislope = dia_pislope / (dia_mumax * d2s * chl_lday)   # decrease slope if growing fast, increase if not much daylight 
    phy_chl_lpar = ( 1.0 - np.exp( -phy_chl_pislope * par_phy_mld ) )
    dia_chl_lpar = ( 1.0 - np.exp( -dia_chl_pislope * par_dia_mld ) )
    phy_chl_pro = phy_mumax * phy_chl_lpar
    dia_chl_pro = dia_mumax * dia_chl_lpar
    #       iii) find phytoplankton biomass production rate
    phy_pislope2 = phy_pislope / ( (1.0/d2s + p_phy_lmort) * lday * d2s ) # alter slope accounting for respiration
    dia_pislope2 = dia_pislope / ( (1.0/d2s + p_dia_lmort) * lday * d2s ) # alter slope accounting for respiration
    phy_lpar = 1.0 - np.exp( (-phy_pislope2 * par_phy) )
    dia_lpar = 1.0 - np.exp( (-dia_pislope2 * par_dia) )
    phy_pro = phy_mumax * lday * phy_lpar
    dia_pro = dia_mumax * lday * dia_lpar
    
    #print("Chl:C ratio =", phy_ChlC[0], phy_ChlC[10])
    #print("PI slope =", phy_pislope[0], phy_pislope[10])
    #print("PI slope (Chl) =", phy_chl_pislope[0], phy_chl_pislope[10])
    #print("Light-limited phytoplankton production =", phy_pro[0]*d2s, phy_pro[10]*d2s)
    #print("Light-limited chlorophyll production =", phy_chl_pro[0]*d2s, phy_chl_pro[10]*d2s)
    
    # 1.3 Calculate nutrient limitation terms
    phy_lno3 = no3_loc/(no3_loc+p_kphy_no3)
    dia_lno3 = no3_loc/(no3_loc+p_kdia_no3)
    phy_FeC = phyfe_loc / phy_loc
    phy_minFeC = p_fe_coef1 * pchl_loc / phy_loc + \
                 p_fe_coef2 * phy_lno3 + \
                 p_fe_coef3 * phy_lno3
    phy_ldfe = np.fmax(0.0, (phy_FeC - phy_minFeC) / p_phy_optFeC)
    dia_FeC = diafe_loc / dia_loc
    dia_minFeC = p_fe_coef1 * dchl_loc / dia_loc + \
                 p_fe_coef2 * dia_lno3 + \
                 p_fe_coef3 * dia_lno3
    dia_ldfe = np.fmax(0.0, (dia_FeC - dia_minFeC) / p_dia_optFeC) 
    phy_lnut = np.fmin(phy_lno3, phy_ldfe)
    dia_lnut = np.fmin(dia_lno3, dia_ldfe)
    # 1.4 Calculate the realised growth rate of phytoplankton after light and nutrient limitation
    phy_mu = phy_pro * phy_lnut
    dia_mu = dia_pro * dia_lnut
    # 1.5 Calculate the realised growth rate of chlorophyll
    phy_chl_pro = d2s * phy_mu * phy_loc * (phy_chl_pro * phy_lnut)
    dia_chl_pro = d2s * dia_mu * dia_loc * (dia_chl_pro * dia_lnut)
    phy_chl_pro_min = p_phy_minChlC * phy_mu * phy_loc * 12
    dia_chl_pro_min = p_dia_minChlC * dia_mu * dia_loc * 12
    phy_maxChlC = (p_phy_maxChlC / (1.0 - 1.14 / 43.4 * tc)) * (1.0 - 1.14 / 43.4 * 20.0)
    phy_maxChlC = np.fmin(p_phy_maxChlC, phy_maxChlC)
    dia_maxChlC = (p_dia_maxChlC / (1.0 - 1.14 / 43.4 * tc)) * (1.0 - 1.14 / 43.4 * 20.0)
    dia_maxChlC = np.fmin(p_dia_maxChlC, dia_maxChlC)
    phy_chl_mu = phy_chl_pro_min + (phy_maxChlC - p_phy_minChlC)*12*phy_chl_pro / (phy_pislope * par_phy_mld/chl_lday)
    dia_chl_mu = dia_chl_pro_min + (dia_maxChlC - p_dia_minChlC)*12*dia_chl_pro / (dia_pislope * par_dia_mld/chl_lday)
    # 1.5 Collect terms for phytoplankton
    phy_dicupt = phy_mu * phy_loc                                           # f11
    phy_no3upt = phy_dicupt / p_CN                                          # f11
    phy_oxyprd = phy_dicupt * p_O2C
    dia_dicupt = dia_mu * dia_loc                                           # f11
    dia_no3upt = dia_dicupt / p_CN                                           # f11
    dia_oxyprd = dia_dicupt * p_O2C
    
    #print("Nutrient-limited phytoplankton production =", phy_mu[0]*d2s, phy_mu[10]*d2s)
    #print("Nutrient-limited chlorophyll production =", phy_chl_pro[0]*d2s, phy_chl_pro[10]*d2s)
    #print("Min chlorophyll production =", phy_chl_pro_min[0]*d2s, phy_chl_pro_min[10]*d2s)
    #print("Max chlorophyll:C =", phy_maxChlC[0], phy_maxChlC[10])
    #print("Realised chlorophyll production =", phy_chl_mu[0]*d2s, phy_chl_mu[10]*d2s)
    
    
    # Collect terms for Iron uptake by phytoplankton and diatoms
    phy_maxQfe = phy_loc * p_phy_maxFeC
    zmax = np.fmax(0.0, (1.0 - phyfe_loc/phy_maxQfe) / np.abs(1.05 - phyfe_loc/phy_maxQfe))
    phy_dfeupt = phy_loc * phy_mumax * p_phy_maxFeC * \
                 dfe_loc / ( dfe_loc + p_kphy_dfe ) * zmax * \
                 (4.0 - 4.5 * phy_ldfe / ( phy_ldfe + 0.5 ))
    dia_maxQfe = dia_loc * p_dia_maxFeC
    zmax = np.fmax(0.0, (1.0 - diafe_loc/dia_maxQfe) / np.abs(1.05 - diafe_loc/dia_maxQfe))
    dia_dfeupt = dia_loc * dia_mumax * p_dia_maxFeC * \
                 dfe_loc / ( dfe_loc + p_kdia_dfe ) * zmax * \
                 (4.0 - 4.5 * dia_ldfe / ( dia_ldfe + 0.5 ))
    
    
    #----------------------------------------------------------------------
    # (2) Grazing by zooplankton
    #----------------------------------------------------------------------
    # 1.1 Calculate temperature-dependent maximum grazing rate
    zoo_mumax = p_zoo_grz * Tfunc_hete
    mes_mumax = p_mes_grz * Tfunc_hete
    # 1.2 Calculate prey capture rate function (/s)
    zoo_capt = p_zoo_capcoef * (phy_loc + det_loc) * (phy_loc + det_loc)
    mes_capt = p_mes_capcoef * (phy_loc + dia_loc + det_loc + zoo_loc) * (phy_loc + dia_loc + det_loc + zoo_loc)
    # 1.3 Calculate the realised grazing rate of zooplankton
    zoo_mu = zoo_mumax * zoo_capt / (zoo_mumax + zoo_capt)
    mes_mu = mes_mumax * mes_capt / (mes_mumax + mes_capt)
        # The functional form of grazing creates increasing half-saturation 
        # values as maximum growth rates increase. It ensures that zooplankton
        # always have the same affinity for phytoplankton irrespective of temp
    # 1.4 Collect terms
    zoo_phygrz = zoo_mu * zoo_loc * (phy_loc/(phy_loc+det_loc+eps))         # f21
    zoo_detgrz = zoo_mu * zoo_loc * (det_loc/(phy_loc+det_loc+eps))         # f21
    zoo_zooexc = zoo_loc * p_zoo_excre * Tfunc_hete * d2s                         # f31
    mes_phygrz = mes_mu * mes_loc * (phy_loc/(phy_loc+dia_loc+det_loc+zoo_loc+eps))         # f21
    mes_diagrz = mes_mu * mes_loc * (dia_loc/(phy_loc+dia_loc+det_loc+zoo_loc+eps))         # f21
    mes_detgrz = mes_mu * mes_loc * (det_loc/(phy_loc+dia_loc+det_loc+zoo_loc+eps))         # f21
    mes_zoogrz = mes_mu * mes_loc * (zoo_loc/(phy_loc+dia_loc+det_loc+zoo_loc+eps))         # f21
    mes_mesexc = mes_loc * p_mes_excre * Tfunc_hete * d2s                        # f31
    # 1.5 Calculate if extra Fe is needed or excreted by zooplankton
    zoo_phygrz_extrafe = zoo_phygrz*phy_FeC - zoo_phygrz*p_zoo_optFeC
    zoo_detgrz_extrafe = zoo_detgrz*det_FeC - zoo_detgrz*p_zoo_optFeC
    mes_phygrz_extrafe = mes_phygrz*phy_FeC - mes_phygrz*p_mes_optFeC
    mes_diagrz_extrafe = mes_diagrz*dia_FeC - mes_diagrz*p_mes_optFeC
    mes_detgrz_extrafe = mes_detgrz*det_FeC - mes_detgrz*p_mes_optFeC
    mes_zoogrz_extrafe = mes_zoogrz*zoo_FeC - mes_zoogrz*p_mes_optFeC
    # if negative (Fe needed), then increase the grazing rates of zooplankton by up to 2-fold
    for kk in np.arange(0,len(z_zgrid)-1):
        if zoo_phygrz_extrafe[kk] < 0.0:
            zoo_Fethirst = np.fmin(2.0, (zoo_phygrz[kk]+zoo_detgrz[kk])*p_zoo_optFeC \
                                        / ( zoo_phygrz[kk]*phy_FeC[kk] + zoo_detgrz[kk]*det_FeC[kk] ) )
            zoo_mumax[kk] = p_zoo_grz * Tfunc_hete[kk] * zoo_Fethirst
            zoo_mu[kk] = zoo_mumax[kk] * zoo_capt[kk] / (zoo_mumax[kk] + zoo_capt[kk])
            zoo_phygrz[kk] = zoo_mu[kk] * zoo_loc[kk] * (phy_loc[kk]/(phy_loc[kk]+det_loc[kk]+eps)) 
            zoo_detgrz[kk] = zoo_mu[kk] * zoo_loc[kk] * (det_loc[kk]/(phy_loc[kk]+det_loc[kk]+eps))
            zoo_phygrz_extrafe[kk] = zoo_phygrz[kk]*phy_FeC[kk] - zoo_phygrz[kk]*p_zoo_optFeC
            zoo_detgrz_extrafe[kk] = zoo_detgrz[kk]*det_FeC[kk] - zoo_detgrz[kk]*p_zoo_optFeC
        if mes_phygrz_extrafe[kk] < 0.0:
            mes_Fethirst = np.fmin(2.0, 
                                   (mes_phygrz[kk]+mes_diagrz[kk]+mes_detgrz[kk]+mes_zoogrz[kk])*p_zoo_optFeC \
                                   / ( mes_phygrz[kk]*phy_FeC[kk] + mes_diagrz[kk]*dia_FeC[kk] + \
                                       mes_detgrz[kk]*det_FeC[kk] + mes_zoogrz[kk]*p_zoo_optFeC) )
            mes_mumax[kk] = p_mes_grz * Tfunc_hete[kk] * mes_Fethirst
            mes_mu[kk] = mes_mumax[kk] * mes_capt[kk] / (mes_mumax[kk] + mes_capt[kk])
            mes_phygrz[kk] = mes_mu[kk] * mes_loc[kk] * (phy_loc[kk]/(phy_loc[kk]+dia_loc[kk]+det_loc[kk]+zoo_loc[kk]+eps))         # f21
            mes_diagrz[kk] = mes_mu[kk] * mes_loc[kk] * (dia_loc[kk]/(phy_loc[kk]+dia_loc[kk]+det_loc[kk]+zoo_loc[kk]+eps))         # f21
            mes_detgrz[kk] = mes_mu[kk] * mes_loc[kk] * (det_loc[kk]/(phy_loc[kk]+dia_loc[kk]+det_loc[kk]+zoo_loc[kk]+eps))         # f21
            mes_zoogrz[kk] = mes_mu[kk] * mes_loc[kk] * (zoo_loc[kk]/(phy_loc[kk]+dia_loc[kk]+det_loc[kk]+zoo_loc[kk]+eps))         # f21
            mes_phygrz_extrafe[kk] = mes_phygrz[kk]*phy_FeC[kk] - mes_phygrz[kk]*p_mes_optFeC
            mes_diagrz_extrafe[kk] = mes_diagrz[kk]*dia_FeC[kk] - mes_diagrz[kk]*p_mes_optFeC
            mes_detgrz_extrafe[kk] = mes_detgrz[kk]*det_FeC[kk] - mes_detgrz[kk]*p_mes_optFeC
            mes_zoogrz_extrafe[kk] = mes_zoogrz[kk]*zoo_FeC[kk] - mes_zoogrz[kk]*p_mes_optFeC
    
    
    #----------------------------------------------------------------------
    # (3) Mortality terms
    #----------------------------------------------------------------------
    phy_lmort = p_phy_lmort * phy_loc * Tfunc_hete                             # f22
    phy_qmort = p_phy_qmort * phy_loc * phy_loc # Not temperature-dependent?   # f23
    dia_lmort = p_dia_lmort * dia_loc * Tfunc_hete                             # f22
    dia_qmort = p_dia_qmort * dia_loc * dia_loc # Not temperature-dependent?   # f23
    zoo_qmort = p_zoo_qmort * zoo_loc * zoo_loc # Not temperature-dependent?   # f32
    mes_qmort = p_mes_qmort * mes_loc * mes_loc # Not temperature-dependent?   # f32
    
    
    #----------------------------------------------------------------------
    # (4) Detritus and CaCO3
    #----------------------------------------------------------------------
    det_remin = p_det_rem * det_loc * Tfunc_hete * d2s                    # f41
    cal_remin = p_cal_rem * cal_loc                                       # f51


    #----------------------------------------------------------------------
    # (5)  Tracer equations (tendencies) Sources - Sinks
    #----------------------------------------------------------------------
    ddt_o2 = phy_oxyprd + dia_oxyprd \
             - (phy_lmort + dia_lmort + zoo_zooexc + mes_mesexc + det_remin) * p_O2C
             
    ddt_no3 = (phy_lmort + dia_lmort + zoo_zooexc + mes_mesexc + det_remin) / p_CN \
              - phy_no3upt - dia_no3upt
    
    ddt_dfe = phy_lmort*phy_FeC + dia_lmort*dia_FeC \
              + zoo_zooexc*zoo_FeC + mes_mesexc*mes_FeC \
              + det_remin*det_FeC \
              - phy_dfeupt - dia_dfeupt \
              - p_dfe_scav * np.fmax(0.0, dfe_loc - p_dfe_deep)
    
    ddt_phy = phy_dicupt - (phy_lmort + phy_qmort + zoo_phygrz + mes_phygrz)
    ddt_dia = dia_dicupt - (dia_lmort + dia_qmort + mes_diagrz)
    ddt_pchl = phy_chl_mu - phy_ChlC * (phy_lmort + phy_qmort + zoo_phygrz + mes_phygrz)*12 
    ddt_dchl = dia_chl_mu - dia_ChlC * (dia_lmort + dia_qmort + mes_diagrz)*12 
    ddt_phyfe = phy_dfeupt - phy_FeC * (phy_lmort + phy_qmort + zoo_phygrz + mes_phygrz)
    ddt_diafe = dia_dfeupt - dia_FeC * (dia_lmort + dia_qmort + mes_diagrz)

    ddt_zoo = (zoo_phygrz + zoo_detgrz) * p_zoo_assim - (zoo_zooexc + zoo_qmort) - mes_zoogrz
    ddt_mes = (mes_phygrz + mes_diagrz + mes_detgrz + mes_zoogrz) * p_mes_assim - (mes_mesexc + mes_qmort)
    ddt_zoofe = (zoo_phygrz*phy_FeC + zoo_detgrz*det_FeC) * p_zoo_assim \
                - (zoo_zooexc + zoo_qmort + mes_zoogrz)*zoo_FeC
    ddt_mesfe = (mes_phygrz*phy_FeC + mes_diagrz*dia_FeC + mes_detgrz*det_FeC + mes_zoogrz*zoo_FeC) * p_mes_assim \
                - (mes_mesexc + mes_qmort)*mes_FeC
                
    ddt_det = (zoo_phygrz + zoo_detgrz) * (1.0-p_zoo_assim) \
              + (mes_phygrz + mes_diagrz + mes_detgrz + mes_zoogrz) * (1.0-p_mes_assim) \
              + phy_qmort + dia_qmort + zoo_qmort + mes_qmort \
              - det_remin - zoo_detgrz - mes_detgrz
    ddt_detfe = (zoo_phygrz*phy_FeC + zoo_detgrz*det_FeC) * (1.0-p_zoo_assim) \
                + (mes_phygrz*phy_FeC + mes_diagrz*dia_FeC + mes_detgrz*det_FeC + mes_zoogrz*zoo_FeC) * (1.0-p_mes_assim) \
                + phy_qmort*phy_FeC + dia_qmort*dia_FeC + zoo_qmort*zoo_FeC + mes_qmort*mes_FeC \
                - (det_remin + zoo_detgrz + mes_detgrz) * det_FeC \
                + p_dfe_scav * np.fmax(0.0, dfe_loc - p_dfe_deep)

    
    ddt_cal = ( zoo_phygrz * (1.0-p_zoo_assim) + mes_phygrz * (1.0-p_mes_assim) \
              + phy_qmort + zoo_qmort + mes_qmort ) * p_cal_fra \
              - cal_remin
    
    ddt_alk = -ddt_no3 - ddt_cal * 2.0
    
    ddt_dic = (ddt_no3 * p_CN) - ddt_cal
    
    #----------------------------------------------------------------------
    # (6)  Diagnostics for evaluation
    #----------------------------------------------------------------------
    pgi_zoo = (zoo_mu * zoo_loc) / (phy_loc + det_loc)
    pgi_mes = (mes_mu * mes_loc) / (phy_loc + dia_loc + det_loc + zoo_loc)
    

    return [ddt_o2, ddt_no3, ddt_dfe, ddt_phy, ddt_dia, ddt_zoo, ddt_mes, ddt_det, \
            ddt_cal, ddt_alk, ddt_dic, ddt_pchl, ddt_dchl, \
            ddt_phyfe, ddt_diafe, ddt_detfe, ddt_zoofe, ddt_mesfe, \
            pgi_zoo, pgi_mes, \
            phy_mu, dia_mu, zoo_mu, mes_mu, phy_lmort, dia_lmort, phy_qmort, dia_qmort, \
            phy_ChlC, dia_ChlC, phy_FeC*1e6, dia_FeC*1e6, det_FeC*1e6, zoo_FeC*1e6, mes_FeC*1e6]
    

