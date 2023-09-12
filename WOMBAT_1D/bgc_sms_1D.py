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
            phy_lday[kk] = 1.5 * zval / (12.0 * zval)
            
    
    
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
    


#@jit(nopython=True)
#def bgc_sms_2P2Z(o2, no3, dfe, phy, dia, zoo, mes, det, cal, alk, dic,\
#                 rsds, tos, tob, z_dz, z_mld, z_zgrid):
  
