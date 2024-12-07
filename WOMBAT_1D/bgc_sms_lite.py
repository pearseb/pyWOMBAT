#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 10:52:47 2023

@author: pbuchanan
"""

import numpy as np
from numba import jit


@jit(nopython=True)
def bgc_sms_0D(o2, no3, dfe, phy, zoo, det, cal, alk, dic, \
               pchl, phyfe, zoofe, detfe, \
               p_Chl_k, par, tc, day, lat, z_mld):
   
    eps = 1e-16
    
    # TEMPORARY
    o2_loc = o2[1]    # µM O2
    no3_loc = no3[1]  # µM NO3
    dfe_loc = dfe[1]  # µM dFe
    phy_loc = phy[1]  # µM N
    zoo_loc = zoo[1]  # µM N
    det_loc = det[1]  # µM N
    cal_loc = cal[1]  # µM CaCO3
    alk_loc = alk[1]  # µM Eq Alk
    dic_loc = dic[1]  # µM DIC
    pchl_loc = pchl[1]  # mg Chl
    phyfe_loc = phyfe[1]  # µM Fe
    zoofe_loc = zoofe[1]  # µM Fe
    detfe_loc = detfe[1]  # µM Fe
    
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
    pchl_loc = np.fmax(pchl_loc, 0.0)
    phyfe_loc = np.fmax(phyfe_loc, 0.0)
    zoofe_loc = np.fmax(zoofe_loc, 0.0)
    detfe_loc = np.fmax(detfe_loc, 0.0)
    
    
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
    p_alpha = 2.5           # Initial slope of the PI curve (mmol C m-2 per mg Chl W sec)
    p_PAR_bio = 0.43        # fraction of shortwave radiation available to phytoplankton
    ###################
    ### Temperature ###
    ###################
    tck = tc + 273.15
    p_auto_aT = 1.0 / d2s        # linear (vertical) scaler for autotrophy (/s)
    p_auto_bT = 1.066            # base coefficient determining temperature-sensitivity of autotrophy
    p_auto_cT = 1.0              # exponential scaler for autotrophy (per ºC)
    p_hete_aT = 1.0 / d2s        # linear (vertical) heterotrophic growth scaler (/s)
    p_hete_bT = 1.075            # base coefficient determining temperature-sensitivity of heterotrophy
    p_hete_cT = 1.0              # exponential scaler for heterotrophy (per ºC)
    def Tfunc(a,b,c,t):
        return a*b**(c*t)
    Tfunc_auto = Tfunc(p_auto_aT, p_auto_bT, p_auto_cT, tc)
    Tfunc_hete = Tfunc(p_hete_aT, p_hete_bT, p_hete_cT, tc)
    ############################
    ####### Phytoplannkton  ####
    ############################
    # DIC + NO3 + dFe --> POC + O2
    p_phy_kn = 0.7             # µM
    p_phy_kfe = 0.1*1e-3       # µM
    p_phy_biothresh = 1.0      # µM threshold for blooms
    p_phy_lmort = 0.025        # linear mortality of phytoplankton (basal respiration) (/s)
    p_phy_qmort = 0.1 / d2s    # quadratic mortality of phytoplankton (1 / (µM N * s))
    p_phy_CN = 122.0/16.0       # mol/mol
    p_phy_FeC = 7.1e-5          # mol/mol (based on Fe:P of 0.0075:1 (Moore et al 2015))
    p_phy_O2C = 172.0/122.0     # mol/mol
    p_phy_minchlc = 0.004       # minimum chlorophyll : Carbon ratio
    p_phy_maxchlc = 0.033       # maximum chlorophyll : Carbon ratio (reduced by cooler temperatures)
    p_phy_optFeC = 7e-6         # optimal Fe:C quota of phytoplankton
    p_phy_maxFeC = 40e-6        # maximum Fe:C quota of phytoplankton
    
    ##########################
    ####### Zooplannkton  ####
    ##########################
    p_zoo_grz = 3.0             # scaler for rate of zooplankton grazing
    p_zoo_capcoef = 1.0 / d2s   # prey capture efficiency coefficient (m6 / (µM N)2 * s))
    p_zoo_prefdet = 0.1
    p_zoo_qmort = 0.5 / d2s    # quadratic mortality of zooplankton (1 / (µM N * day))
    p_zoo_excre = 0.05         # rate of excretion by zooplankton (/day)
    p_zoo_assim = 0.7         # zooplankton assimilation efficiency
    ######################
    ####### Detritus  ####
    ######################
    p_det_rem = 0.048           # remineralisation rate of detritus (/s)
    ###################
    ####### CaCO3  ####
    ###################
    p_cal_rem = 0.001714 / d2s  # remineralisation rate of CaCO3 (/s)
    p_cal_fra = 0.062           # fraction of inorganic
    
    
    
    #----------------------------------------------------------------------
    # (1) Light field
    #----------------------------------------------------------------------
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
    zchl = np.fmax(0.01, np.fmin(10.0, pchl_loc))
    ichl = int( 40 + 20 * np.log10(zchl) )
    ek_blu = p_Chl_k[ichl,1]
    ek_gre = p_Chl_k[ichl,2]
    ek_red = p_Chl_k[ichl,3]
    # iii)  Using the attenuation coefficients, estimate the amount of RGB light available
    par_blu = par * np.exp(-0.5 * ek_blu * 50) * p_PAR_bio
    par_gre = par * np.exp(-0.5 * ek_gre * 50) * p_PAR_bio
    par_red = par * np.exp(-0.5 * ek_red * 50) * p_PAR_bio
    # iv)   Find light available for Phytoplankton
    par_tot = par_blu + par_gre + par_red
    par_phy = 1.85 * par_blu + 0.68 * par_gre + 0.46 * par_red
    # v)    Calculate euphotic layer depth
    z_eup = 150.0
    # vi)    Collect mean light over the mixed layer (total and phytoplankton specific)
    par_tot_mld = par_tot*1
    par_phy_mld = par_phy*1
    zval = np.fmax(1.0, p_hrday) * np.fmin(1.0, z_eup/z_mld)
    chl_lday = zval / 24.0
    phy_lday = 1.5 * zval / (12.0 + zval)

    
    #----------------------------------------------------------------------
    # (2) Dissolve iron chemistry
    #----------------------------------------------------------------------
    # Determine equilibrium fractionation of total dFe into Fe` and L-Fe
    ligand = 0.7 * 1e-9   # nM * mol per nM
    fe_keq = 10**( 17.27 - 1565.7 / tck )
    fe_III = ( -( 1. + fe_keq * ligand - fe_keq * dfe_loc*1e-6 ) + \
              ( ( 1. + fe_keq * ligand - fe_keq * dfe_loc*1e-6 )**2 \
                 + 4. * dfe_loc*1e-6 * fe_keq)**0.5 ) / ( 2. * fe_keq ) * 1e6
    fe_lig = np.fmax(0.0, dfe_loc - fe_III)
    
    # Precipitation of Fe` (creation of nanoparticles)
    sal = 35.0
    zval = 19.924 * sal / ( 1000. - 1.005 * sal)
    fesol1 = 10**(-13.486 - 0.1856*zval**0.5 + 0.3073*zval + 5254.0/np.fmax(tck, 278.15) )
    fesol2 = 10**(2.517 - 0.8885*zval**0.5 + 0.2139*zval - 1320.0/np.fmax(tck, 278.15) )
    fesol3 = 10**(0.4511 - 0.3305*zval**0.5 - 1996.0/np.fmax(tck, 278.15) )
    fesol4 = 10**(-0.2965 - 0.7881*zval**0.5 - 4086.0/np.fmax(tck, 278.15) )
    fesol5 = 10**(4.4466 - 0.8505*zval**0.5 - 7980.0/np.fmax(tck, 278.15) )
    hp = 10**(-7.9)
    fe3sol = fesol1 * ( hp**3 + fesol2 * hp**2 + fesol3 * hp + fesol4 + fesol5 / hp ) *1e6
    precip = np.fmax(0.0, (fe_III-fe3sol) ) * 0.01/d2s
    
    # Scavenging of Fe` --> Det
    partic = (det_loc + cal_loc)
    scaven = fe_III * (3e-5 + 0.005 * partic) / d2s 
    scadet = scaven * (det_loc+eps) / (partic+eps)

    # Coagulation of colloidal Fe (nM) into small and large particles (uM) 
    fe_col = fe_lig * 0.5
    doc_loc = 40e-6     # mol/L
    zval = ( 0.369 * 0.3 * doc_loc + 102.4 * det_loc*1e-6 ) + ( 114. * 0.3 * doc_loc )
    fe2det = fe_col * zval / d2s
    
    precip = precip*0.
    scaven = scaven*0.
    scadet = scadet*0.
    fe2det = fe2det*0.
    
    
    #----------------------------------------------------------------------
    # (3) Primary production of nanophytoplankton
    #----------------------------------------------------------------------
    # 1.1 Calculate the temperature-dependent maximum growth rate (/s)
    phy_mumax = Tfunc_auto * 1
    # 1.2 Nutrient limitation terms
    phy_conc1 = np.fmax(0.0, phy_loc - p_phy_biothresh)
    phy_conc2 = phy_loc - phy_conc1
    phy_k_nit = np.fmax(p_phy_kn, 
                        ( phy_conc2 * p_phy_kn + phy_conc1 * 3 * p_phy_kn ) / ( phy_loc + eps ) )
    phy_k_dfe = np.fmax(p_phy_kfe, 
                        ( phy_conc2 * p_phy_kfe + phy_conc1 * 3 * p_phy_kfe ) / ( phy_loc + eps ) )
    
    phy_limnit = no3_loc / (no3_loc + phy_k_nit)
    phy_minFeC = 0.0016 / 55.85 * pchl_loc/(phy_loc*12+eps) + \
                 1.21e-5 * 14.0 / 55.85 / 7.625 * 0.5 * 1.5 * phy_limnit + \
                 1.15e-4 * 14.0 / 55.85 / 7.625 * 0.5 * phy_limnit
    phy_limdfe = np.fmin(1.0, np.fmax(0.0, ( phyfe_loc/(phy_loc+eps) - phy_minFeC ) / p_phy_optFeC ))
    phy_limnut = np.fmin(phy_limnit, phy_limdfe)
    
    # 1.3 Calculate growth limitation by light
    chlc_ratio = pchl_loc / (phy_loc * 12.0 + eps)
    phy_pisl = np.fmax(p_alpha * chlc_ratio, p_alpha * p_phy_minchlc)
    # Original
    #phy_pisl2 = phy_pisl / ( (1.0/d2s + p_phy_lmort) * phy_lday * d2s ) # alter slope accounting for respiration
    #phy_lpar = 1.0 - np.exp( (-phy_pisl2 * par_phy) )
    # New
    phy_pisl2 = phy_pisl / ( (1.0 + p_phy_lmort) * np.fmax(p_hrday, 1.0)/24.0 ) # alter slope accounting for respiration
    phy_lpar = (1.0 - np.exp( -phy_pisl2 * par_phy ) ) * phy_lday
    phy_pro = phy_mumax * phy_lpar
    # 1.4 Apply nutrient limitation to phytoplankton growth to get realised growth rate
    phy_mu = phy_pro * phy_limnut
    # 1.5 Growth rate in chlorophyll
    chl_pisl = phy_pisl / (phy_mumax * d2s * np.fmin(1.0, p_hrday)/24.0)   # decrease slope if growing fast, increase if not much daylight 
    chl_lpar = ( 1.0 - np.exp( -chl_pisl * par_phy_mld ) ) * chl_lday
    chl_pro = phy_mumax * chl_lpar * phy_limnut
    chl_mumin = phy_mu * phy_loc * 12 * p_phy_minchlc
    chl_maxq = np.fmin(p_phy_maxchlc, (p_phy_maxchlc / (1.0 - 1.14/43.4 * tc)) * (1.0 - 1.14/43.4 * 20))
    chl_mu = phy_mu * phy_loc * 12 * d2s * chl_pro
    if chl_mu > 0.0:
        chl_mu = chl_mumin + ( (chl_maxq - p_phy_minchlc) * chl_mu ) / ( phy_pisl * par_phy_mld/chl_lday )
    # 1.6 Collect terms for phytoplankton
    phy_dicupt = phy_mu * phy_loc                                           # f11
    phy_no3upt = phy_dicupt / p_phy_CN
    phy_oxyprd = phy_dicupt * p_phy_O2C


    #----------------------------------------------------------------------
    # (4) Dissolved Iron uptake by phytoplankton
    #----------------------------------------------------------------------
    phy_maxQFe = phy_loc * p_phy_maxFeC
    phy_Feupt_upreg   = (4.0 - 4.5 * phy_limdfe / (phy_limdfe + 0.5))
    phy_Feupt_downreg = np.fmax(0.0, (1.0 - phyfe_loc/(phy_maxQFe+eps)) \
                                / np.abs(1.05 - phyfe_loc/(phy_maxQFe+eps)) )
    phy_dfeupt = phy_loc * phy_mumax * p_phy_maxFeC * \
                 dfe_loc / (dfe_loc + phy_k_dfe) * phy_Feupt_downreg * phy_Feupt_upreg

    #----------------------------------------------------------------------
    # (5) Grazing by zooplankton
    #----------------------------------------------------------------------
    # 1.1 Calculate temperature-dependent maximum grazing rate
    zoo_mumax = p_zoo_grz * Tfunc_hete
    # 1.2 Calculate prey capture rate function (/s)
    prey = phy_loc + det_loc*p_zoo_prefdet
    zoo_capt = p_zoo_capcoef * prey * prey
    # 1.3 Calculate the realised grazing rate of zooplankton
    zoo_mu = zoo_mumax * zoo_capt / (zoo_mumax + zoo_capt)
        # The functional form of grazing creates increasing half-saturation 
        # values as maximum growth rates increase. It ensures that zooplankton
        # always have the same affinity for phytoplankton irrespective of temp
    # 1.4 Collect terms
    zoo_phygrz = zoo_mu * zoo_loc * phy_loc / (prey + eps)                  # f21
    zoo_detgrz = zoo_mu * zoo_loc * det_loc*p_zoo_prefdet / (prey + eps)    # f21
    zoo_zooexc = zoo_loc * p_zoo_excre * Tfunc_hete                         # f31
    
    
    #----------------------------------------------------------------------
    # (6) Mortality terms
    #----------------------------------------------------------------------
    phy_lmort = p_phy_lmort * phy_loc * Tfunc_hete * np.fmin(1.0, np.fmax(0.0, phy_limnut/0.3))
    phy_qmort = p_phy_qmort * phy_loc * phy_loc # Not temperature-dependent?   # f23
    zoo_qmort = p_zoo_qmort * zoo_loc * zoo_loc # Not temperature-dependent?   # f32
    
    
    #----------------------------------------------------------------------
    # (7) Detritus and CaCO3
    #----------------------------------------------------------------------
    det_remin = p_det_rem * det_loc * Tfunc_hete                          # f41
    cal_remin = p_cal_rem * cal_loc                                       # f51


    #----------------------------------------------------------------------
    # (8)  Tracer equations (tendencies) Sources - Sinks
    #----------------------------------------------------------------------
    
    # ratios 
    phyFeC = phyfe_loc / (phy_loc + eps)
    zooFeC = zoofe_loc / (zoo_loc + eps)
    detFeC = detfe_loc / (det_loc + eps)
    
    
    ddt_o2 = phy_oxyprd - (phy_lmort + zoo_zooexc + det_remin) * p_phy_O2C
    
    ddt_no3 = (phy_lmort + zoo_zooexc + det_remin)/p_phy_CN - phy_no3upt
    
    ddt_dfe = (phy_lmort*phyFeC + zoo_zooexc*zooFeC + det_remin*detFeC) - phy_dfeupt \
              - precip - scaven - fe2det
              
    ddt_phy = phy_dicupt - (phy_lmort + phy_qmort + zoo_phygrz)
    ddt_phyfe = phy_dfeupt - (phy_lmort + phy_qmort + zoo_phygrz)*phyFeC
    
    ddt_zoo = (zoo_phygrz + zoo_detgrz)*p_zoo_assim - (zoo_zooexc + zoo_qmort)
    ddt_zoofe = (zoo_phygrz*phyFeC + zoo_detgrz*detFeC)*p_zoo_assim - (zoo_zooexc + zoo_qmort)*zooFeC
    
    ddt_det = (zoo_phygrz + zoo_detgrz)*(1.0-p_zoo_assim) + phy_qmort + zoo_qmort - det_remin - zoo_detgrz
    ddt_detfe = (zoo_phygrz*phyFeC + zoo_detgrz*detFeC)*(1.0-p_zoo_assim) \
                + phy_qmort*phyFeC + zoo_qmort*zooFeC \
                - (det_remin + zoo_detgrz)*detFeC \
                + scadet + fe2det
    
    ddt_cal = (zoo_phygrz*(1.0-p_zoo_assim) + phy_qmort + zoo_qmort)*p_cal_fra - cal_remin
    
    ddt_alk = -ddt_no3 - ddt_cal * 2.0
    
    ddt_dic = (ddt_no3 * p_phy_CN) - ddt_cal
    
    ddt_pchl = chl_mu - chlc_ratio * (phy_lmort + phy_qmort + zoo_qmort)*12 


    return [ddt_o2, ddt_no3, ddt_dfe, ddt_phy, ddt_zoo, ddt_det, \
            ddt_cal, ddt_alk, ddt_dic, ddt_pchl, ddt_phyfe, ddt_zoofe, ddt_detfe, \
            phy_mu, zoo_mu, phy_lmort, phy_qmort, \
            chlc_ratio, phyFeC*1e6, zooFeC*1e6, detFeC*1e6]
    

@jit(nopython=True)
def bgc_sms_1D(o2, no3, dfe, phy, zoo, det, cal, alk, dic, pchl, phyfe, zoofe, detfe, \
               p_Chl_k, par, tos, tob, \
               day, lat, z_dz, z_mld, z_zgrid):
   
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
    pchl_loc = pchl[1,:]  # mg Chl
    phyfe_loc = phyfe[1,:]  # µM Fe
    zoofe_loc = zoofe[1,:]  # µM Fe
    detfe_loc = detfe[1,:]  # µM Fe
    
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
    pchl_loc = np.fmax(pchl_loc, 0.0)
    phyfe_loc = np.fmax(phyfe_loc, 0.0)
    zoofe_loc = np.fmax(zoofe_loc, 0.0)
    detfe_loc = np.fmax(detfe_loc, 0.0)
    
    
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
    p_alpha = 2.25          # Initial slope of the PI curve (mmol C m-2 per mg Chl W sec)
    p_PAR_bio = 0.43        # fraction of shortwave radiation available to phytoplankton
    ###################
    ### Temperature ###
    ###################
    tc = tos - (tos - tob) * (1-np.exp(0.2*p_PAR_k * z_zgrid))
    tck = tc + 273.15
    p_auto_aT = 1.0 / d2s       # linear (vertical) scaler for autotrophy (/s)
    p_auto_bT = 1.050            # base coefficient determining temperature-sensitivity of autotrophy
    p_auto_cT = 1.0              # exponential scaler for autotrophy (per ºC)
    p_hete_aT = 1.0 / d2s        # linear (vertical) heterotrophic growth scaler (/s)
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
    p_phy_kn = 2.0             # µM
    p_phy_kfe = 2.5*1e-3       # µM
    p_phy_biomin = 1e-5        # µM threshold for minimum biomass
    p_phy_biothresh = 0.6      # µM threshold for blooms
    p_phy_lmort = 0.005        # linear mortality of phytoplankton (basal respiration) (/day)
    p_phy_qmort = 0.05 / d2s   # quadratic mortality of phytoplankton (1 / (µM N * s))
    p_phy_CN = 122.0/16.0      # mol/mol
    p_phy_O2C = 172.0/122.0    # mol/mol
    p_phy_minchlc = 0.004      # minimum chlorophyll : Carbon ratio
    p_phy_optchlc = 0.036      # optimal chlorophyll : Carbon ratio (NOT reduced by cooler temperatures)
    p_phy_optFeC = 10e-6       # optimal Fe:C quota of phytoplankton
    p_phy_maxFeC = 50e-6       # maximum Fe:C quota of phytoplankton
    
    ##########################
    ####### Zooplannkton  ####
    ##########################
    p_zoo_grz = 3.0             # scaler for rate of zooplankton grazing
    p_zoo_biomin = 1e-5         # µM threshold for minimum biomass
    p_zoo_epsmin = 0.025 / d2s  # prey capture efficiency coefficient (m6 / (µM N)2 * s))
    p_zoo_epsmax = 0.25 / d2s   # prey capture efficiency coefficient (m6 / (µM N)2 * s))
    p_zoo_prefdet = 0.25
    p_zoo_qmort = 0.9 / d2s     # quadratic mortality of zooplankton (1 / (µM N * day))
    p_zoo_respi = 0.05          # rate of excretion by zooplankton (/day)
    p_zoo_assim = 0.6           # zooplankton assimilation efficiency
    p_zoo_kzoo = 0.25           # zooplankton half-saturation coefficient to reduce linear losses due to respiration
    ######################
    ####### Detritus  ####
    ######################
    p_det_rem = 0.090          # remineralisation rate of detritus (/day)
    ###################
    ####### CaCO3  ####
    ###################
    p_cal_rem = 0.00525 / d2s   # remineralisation rate of CaCO3 (/s)
    p_cal_fra = 0.080           # fraction of PIC produced per biomass gain in phytoplankton
    
    
    
    #----------------------------------------------------------------------
    # (1) Light field
    #----------------------------------------------------------------------
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
    zchl = np.fmax(0.01, np.fmin(10.0, pchl_loc))
    ichl = ( 40 + 20 * np.log10(zchl) ).astype(np.int64)
    ek_blu = p_Chl_k[ichl,1]
    ek_gre = p_Chl_k[ichl,2]
    ek_red = p_Chl_k[ichl,3]
    # iii)  Using the attenuation coefficients, estimate the amount of RGB light available
    par_blu = np.zeros(len(z_zgrid)); par_gre = np.zeros(len(z_zgrid)); par_red = np.zeros(len(z_zgrid))
    par_blu[0] = par * np.exp(-0.5 * ek_blu[0] * z_dz) * p_PAR_bio
    par_gre[0] = par * np.exp(-0.5 * ek_gre[0] * z_dz) * p_PAR_bio
    par_red[0] = par * np.exp(-0.5 * ek_red[0] * z_dz) * p_PAR_bio
    for kk in np.arange(1,len(z_zgrid)):
        par_blu[kk] = par_blu[kk-1] * np.exp(-ek_blu[kk-1] * z_dz)
        par_gre[kk] = par_gre[kk-1] * np.exp(-ek_gre[kk-1] * z_dz)
        par_red[kk] = par_red[kk-1] * np.exp(-ek_red[kk-1] * z_dz)
    # iv)   Find light available for Phytoplankton
    par_tot = par_blu + par_gre + par_red
    par_phy = 1.0 * par_blu + 1.0 * par_gre + 1.0 * par_red
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
        zval = np.fmax(1.0, p_hrday)
        if (-z_zgrid[kk] <= z_mld):
            zval = zval * np.fmin(1.0, -z_eup/z_mld)
        chl_lday[kk] = zval / 24.0
        phy_lday[kk] = 1.5 * zval / (12.0 + zval)
    
    
    #----------------------------------------------------------------------
    # (2) Dissolved iron chemistry
    #----------------------------------------------------------------------
    # Determine equilibrium fractionation of total dFe into Fe` and L-Fe
    ligand = 0.7 * 1e-9   # nM * mol per nM
    fe_keq = 10**( 17.27 - 1565.7 / tck ) 
    fe_III = ( -( 1. + fe_keq * ligand - fe_keq * dfe_loc*1e-6 ) + \
              ( ( 1. + fe_keq * ligand - fe_keq * dfe_loc*1e-6 )**2 \
                 + 4. * dfe_loc*1e-6 * fe_keq)**0.5 ) / ( 2. * fe_keq ) * 1e6
    fe_lig = np.fmax(0.0, dfe_loc - fe_III)
    
    # Precipitation of Fe` (creation of nanoparticles)
    sal = 35.0
    zval = 19.924 * sal / ( 1000. - 1.005 * sal)
    fesol1 = 10**(-13.486 - 0.1856*zval**0.5 + 0.3073*zval + 5254.0/np.fmax(tck, 278.15) )
    fesol2 = 10**(2.517 - 0.8885*zval**0.5 + 0.2139*zval - 1320.0/np.fmax(tck, 278.15) )
    fesol3 = 10**(0.4511 - 0.3305*zval**0.5 - 1996.0/np.fmax(tck, 278.15) )
    fesol4 = 10**(-0.2965 - 0.7881*zval**0.5 - 4086.0/np.fmax(tck, 278.15) )
    fesol5 = 10**(4.4466 - 0.8505*zval**0.5 - 7980.0/np.fmax(tck, 278.15) )
    hp = 10**(-7.9)
    fe3sol = fesol1 * ( hp**3 + fesol2 * hp**2 + fesol3 * hp + fesol4 + fesol5 / hp ) *1e6
    precip = np.fmax(0.0, (fe_III-fe3sol) ) * 0.01/d2s
    
    # Scavenging of Fe` --> Det
    partic = (det_loc + cal_loc)
    scaven = fe_III * (3e-5 + 0.5 * partic) / d2s 
    scadet = scaven * (det_loc+eps) / (partic+eps)

    # Coagulation of colloidal Fe (nM) into small and large particles (uM) 
    fe_col = fe_lig * 0.5
    doc_loc = 40e-6     # mol/L
    zval = ( 0.369 * 0.3 * doc_loc + 102.4 * det_loc*1e-6 ) + ( 114. * 0.3 * doc_loc )
    zval[-z_zgrid > z_mld] = zval[-z_zgrid > z_mld]*0.01
    fe2det = fe_col * zval / d2s
    
    #precip = precip*0.
    #scaven = scaven*0.
    #scadet = scadet*0.
    #fe2det = fe2det*0.

    
    #----------------------------------------------------------------------
    # (3) Primary production of nanophytoplankton
    #----------------------------------------------------------------------
    # 1.1 Calculate the temperature-dependent maximum growth rate (/s)
    phy_mumax = Tfunc_auto * 1
    # 1.2 Nutrient limitation terms
    phy_k_nit = p_phy_kn * np.fmax(0.1, phy_loc - p_phy_biothresh)**(0.37)
    phy_k_dfe = p_phy_kfe * np.fmax(0.1, phy_loc - p_phy_biothresh)**(0.37)
    
    phy_limnit = no3_loc / (no3_loc + phy_k_nit)
    phy_minFeC = 0.0016 / 55.85 * pchl_loc/(phy_loc+eps) + \
                 1.21e-5 * 14.0 / 55.85 / 7.625 * 0.5 * 1.5 * phy_limnit + \
                 1.15e-4 * 14.0 / 55.85 / 7.625 * 0.5 * phy_limnit
    phy_limdfe = np.fmin(1.0, np.fmax(0.0, ( phyfe_loc/(phy_loc+eps) - phy_minFeC ) / p_phy_optFeC ))
    phy_limnut = np.fmin(phy_limnit, phy_limdfe)
    
    # 1.3 Calculate growth limitation by light
    chlc_ratio = np.zeros(len(z_zgrid))
    for kk in np.arange(1,len(z_zgrid)):
        if (phy_loc[kk] > 0.0):
            chlc_ratio[kk] = pchl_loc[kk] / (phy_loc[kk] * 12.0)
        else:
            chlc_ratio[kk] = p_phy_minchlc*1.0
    
    phy_pisl = np.fmax(p_alpha * chlc_ratio, p_alpha * p_phy_minchlc)
    phy_pisl2 = phy_pisl / ( (1.0 + p_phy_lmort) * np.fmax(p_hrday, 1.0)/24.0 ) # alter slope accounting for respiration
    phy_lpar = (1.0 - np.exp( -phy_pisl2 * par_phy ) ) * phy_lday
    phy_pro = phy_mumax * phy_lpar
    # 1.4 Apply nutrient limitation to phytoplankton growth to get realised growth rate
    phy_mu = phy_pro * phy_limnut
    # 1.5 Growth rate in chlorophyll
    chl_pisl = phy_pisl / (phy_mumax * d2s * np.fmax(p_hrday, 1.0)/24.0 * (1.0-phy_limnut) )   # decrease slope if growing fast, increase if not much daylight 
    chl_lpar = ( 1.0 - np.exp( -chl_pisl * par_phy_mld ) ) * phy_lday
    chl_mumin = p_phy_minchlc * phy_mu * phy_loc * 12
    chl_muopt = p_phy_optchlc * phy_mu * phy_loc * 12
    chl_mu = (chl_muopt - chl_mumin) * chl_lpar * phy_limnut
    for kk in np.arange(0,len(z_zgrid)-1):
        if (phy_pisl[kk]*par_phy_mld[kk]) > 0.0:
            chl_mu[kk] = chl_mumin[kk] + chl_mu[kk] / (phy_pisl[kk] * par_phy_mld[kk]) 
    # 1.6 Collect terms for phytoplankton
    phy_dicupt = phy_mu * phy_loc                                           # f11
    phy_no3upt = phy_dicupt / p_phy_CN
    phy_oxyprd = phy_dicupt * p_phy_O2C


    #----------------------------------------------------------------------
    # (4) Dissolved Iron uptake by phytoplankton
    #----------------------------------------------------------------------
    phy_maxQFe = phy_loc * p_phy_maxFeC
    phy_Feupt_upreg   = (4.0 - 4.5 * phy_limdfe / (phy_limdfe + 0.5))
    phy_Feupt_downreg = np.fmax(0.0, (1.0 - phyfe_loc/(phy_maxQFe+eps)) \
                                / np.abs(1.05 - phyfe_loc/(phy_maxQFe+eps)) )
    phy_dfeupt = phy_loc * phy_mumax * p_phy_maxFeC * \
                 dfe_loc / (dfe_loc + phy_k_dfe) * phy_Feupt_downreg * phy_Feupt_upreg

    #----------------------------------------------------------------------
    # (5) Grazing by zooplankton
    #----------------------------------------------------------------------
    # 1.1 Calculate temperature-dependent maximum grazing rate
    zoo_mumax = p_zoo_grz * Tfunc_hete
    # 1.2 Calculate prey capture rate function (/s)
    prey = np.fmax(0.0, phy_loc) + np.fmax(0.0, det_loc)*p_zoo_prefdet
    zoo_epsmin = p_zoo_epsmin * (1.5 * np.tanh(0.2*(tc-15.0)) + 2.5)
    zoo_epsilon = zoo_epsmin + (p_zoo_epsmax - zoo_epsmin) / (1.0 + np.exp(-(tc-10.0))) \
                               / (1.0 + np.exp(3.0 * (prey - 2.0*p_phy_biothresh)))
    zoo_capt = zoo_epsilon * prey * prey
    # 1.3 Calculate the realised grazing rate of zooplankton
    zoo_mu = zoo_mumax * zoo_capt / (zoo_mumax + zoo_capt)
        # The functional form of grazing creates increasing half-saturation 
        # values as maximum growth rates increase. It ensures that zooplankton
        # always have the same affinity for phytoplankton irrespective of temp
    # 1.4 Collect terms
    zoo_phygrz = np.zeros(len(z_zgrid))
    zoo_detgrz = np.zeros(len(z_zgrid))
    zoo_phygrz[prey > p_phy_biomin] = zoo_mu[prey > p_phy_biomin] * zoo_loc[prey > p_phy_biomin] * \
                                      phy_loc[prey > p_phy_biomin] / prey[prey > p_phy_biomin]                  # f21
    zoo_detgrz[prey > p_phy_biomin] = zoo_mu[prey > p_phy_biomin] * zoo_loc[prey > p_phy_biomin] * \
                                      det_loc[prey > p_phy_biomin]*p_zoo_prefdet / prey[prey > p_phy_biomin]    # f21
    
    #----------------------------------------------------------------------
    # (6) Mortality terms
    #----------------------------------------------------------------------
    phy_lmort = np.zeros(len(z_zgrid))
    phy_qmort = np.zeros(len(z_zgrid))
    zoo_zoores = np.zeros(len(z_zgrid))
    zoo_qmort = np.zeros(len(z_zgrid))
    phy_lmort[phy_loc > p_phy_biomin] = p_phy_lmort * phy_loc[phy_loc > p_phy_biomin] * \
                                        Tfunc_hete[phy_loc > p_phy_biomin]
    phy_qmort[phy_loc > p_phy_biomin] = p_phy_qmort * phy_loc[phy_loc > p_phy_biomin]**2.0 # Not temperature-dependent?   # f23
    zoo_zoores[zoo_loc > p_zoo_biomin] = zoo_loc[zoo_loc > p_zoo_biomin] * p_zoo_respi * \
                                         Tfunc_hete[zoo_loc > p_zoo_biomin] * \
                                         zoo_loc[zoo_loc > p_zoo_biomin] / (zoo_loc[zoo_loc > p_zoo_biomin]+p_zoo_kzoo)            # f31
    zoo_qmort[zoo_loc > p_zoo_biomin] = p_zoo_qmort * zoo_loc[zoo_loc > p_zoo_biomin]**2.0 # Not temperature-dependent?   # f32
    
    #----------------------------------------------------------------------
    # (7) Detritus and CaCO3
    #----------------------------------------------------------------------
    det_remin = p_det_rem * det_loc * Tfunc_hete                          # f41
    det_remin[-z_zgrid > 180.0] = det_remin[-z_zgrid > 180.0]*0.5
    cal_remin = p_cal_rem * cal_loc                                       # f51


    #----------------------------------------------------------------------
    # (8)  Tracer equations (tendencies) Sources - Sinks
    #----------------------------------------------------------------------
    
    # ratios 
    phyFeC = np.zeros(len(z_zgrid))
    zooFeC = np.zeros(len(z_zgrid))
    detFeC = np.zeros(len(z_zgrid))
    phyFeC[phy_loc>0.0] = phyfe_loc[phy_loc>0.0] / phy_loc[phy_loc>0.0]
    zooFeC[zoo_loc>0.0] = zoofe_loc[zoo_loc>0.0] / zoo_loc[zoo_loc>0.0]
    detFeC[det_loc>0.0] = detfe_loc[det_loc>0.0] / det_loc[det_loc>0.0]
    
    
    ddt_o2 = phy_oxyprd - (phy_lmort + zoo_zoores + det_remin + \
                           (1.0-p_zoo_assim)*0.75*(zoo_phygrz+zoo_detgrz)) * p_phy_O2C
    
    ddt_no3 = (phy_lmort + zoo_zoores + det_remin + \
               (1.0-p_zoo_assim)*0.75*(zoo_phygrz+zoo_detgrz))/p_phy_CN - phy_no3upt
    
    ddt_dfe = (phy_lmort*phyFeC + zoo_zoores*zooFeC + det_remin*detFeC +
               (1.0-p_zoo_assim)*0.75*(zoo_phygrz*phyFeC+zoo_detgrz*detFeC)) - phy_dfeupt \
              - precip - scaven - fe2det
              
    ddt_phy = phy_dicupt - (phy_lmort + phy_qmort + zoo_phygrz)
    ddt_phyfe = phy_dfeupt - (phy_lmort + phy_qmort + zoo_phygrz)*phyFeC
    
    ddt_zoo = (zoo_phygrz + zoo_detgrz)*p_zoo_assim - (zoo_zoores + zoo_qmort)
    ddt_zoofe = (zoo_phygrz*phyFeC + zoo_detgrz*detFeC)*p_zoo_assim - (zoo_zoores + zoo_qmort)*zooFeC
    
    ddt_det = (zoo_phygrz + zoo_detgrz)*(1.0-p_zoo_assim)*0.25 + \
              phy_qmort + zoo_qmort - det_remin - zoo_detgrz
    ddt_detfe = (zoo_phygrz*phyFeC + zoo_detgrz*detFeC)*(1.0-p_zoo_assim)*0.25 \
                + phy_qmort*phyFeC + zoo_qmort*zooFeC \
                - (det_remin + zoo_detgrz)*detFeC \
                + scadet + fe2det
    
    ddt_cal = (zoo_phygrz*(1.0-p_zoo_assim)*0.25 + phy_qmort + zoo_qmort)*p_cal_fra - cal_remin
        
    ddt_dic = ( det_remin + cal_remin + \
                phy_lmort + zoo_zoores + \
                (1.0-p_zoo_assim)*0.75*(zoo_phygrz+zoo_detgrz) ) - phy_dicupt \
              - ( (1.0-p_zoo_assim)*0.25*zoo_phygrz + phy_qmort + zoo_qmort ) * p_cal_fra
    
    ddt_pchl = chl_mu - chlc_ratio * (phy_lmort + phy_qmort + zoo_qmort)*12 
    
    ddt_alk = -ddt_no3 - ddt_cal * 2.0
    
    
    return [ddt_o2, ddt_no3, ddt_dfe, ddt_phy, ddt_zoo, ddt_det, \
            ddt_cal, ddt_alk, ddt_dic, ddt_pchl, ddt_phyfe, ddt_zoofe, ddt_detfe, \
            phy_mu, zoo_mu, phy_lmort, phy_qmort, \
            chlc_ratio, phyFeC*1e6, zooFeC*1e6, detFeC*1e6]
    


### Attempt at integrating new zooplankton metabolic model of Anderson et al. 2021 L&O Letters
@jit(nopython=True)
def bgc_sms_1Dtmp(o2, no3, dfe, phy, zoo, det, cal, alk, dic, pchl, phyfe, zoofe, detfe, \
                  p_Chl_k, par, tos, tob, \
                  day, lat, z_dz, z_mld, z_zgrid):
   
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
    pchl_loc = pchl[1,:]  # mg Chl
    phyfe_loc = phyfe[1,:]  # µM Fe
    zoofe_loc = zoofe[1,:]  # µM Fe
    detfe_loc = detfe[1,:]  # µM Fe
    
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
    pchl_loc = np.fmax(pchl_loc, 0.0)
    phyfe_loc = np.fmax(phyfe_loc, 0.0)
    zoofe_loc = np.fmax(zoofe_loc, 0.0)
    detfe_loc = np.fmax(detfe_loc, 0.0)
    
    
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
    p_alpha = 2.5           # Initial slope of the PI curve (mmol C m-2 per mg Chl W sec)
    p_PAR_bio = 0.43        # fraction of shortwave radiation available to phytoplankton
    ###################
    ### Temperature ###
    ###################
    tc = tos - (tos - tob) * (1-np.exp(p_PAR_k * z_zgrid))
    tck = tc + 273.15
    p_auto_aT = 1.0 / d2s        # linear (vertical) scaler for autotrophy (/s)
    p_auto_bT = 1.066            # base coefficient determining temperature-sensitivity of autotrophy
    p_auto_cT = 1.0              # exponential scaler for autotrophy (per ºC)
    p_hete_aT = 1.0 / d2s        # linear (vertical) heterotrophic growth scaler (/s)
    p_hete_bT = 1.075            # base coefficient determining temperature-sensitivity of heterotrophy
    p_hete_cT = 1.0              # exponential scaler for heterotrophy (per ºC)
    def Tfunc(a,b,c,t):
        return a*b**(c*t)
    Tfunc_auto = Tfunc(p_auto_aT, p_auto_bT, p_auto_cT, tc)
    Tfunc_hete = Tfunc(p_hete_aT, p_hete_bT, p_hete_cT, tc)
    ############################
    ####### Phytoplannkton  ####
    ############################
    # DIC + NO3 + dFe --> POC + O2
    p_phy_kn = 0.7             # µM
    p_phy_kfe = 0.1*1e-3       # µM
    p_phy_biothresh = 1.0      # µM threshold for blooms
    p_phy_lmort = 0.025        # linear mortality of phytoplankton (basal respiration) (/s)
    p_phy_qmort = 0.1 / d2s    # quadratic mortality of phytoplankton (1 / (µM N * s))
    p_phy_CN = 122.0/16.0       # mol/mol
    p_phy_FeC = 7.1e-5          # mol/mol (based on Fe:P of 0.0075:1 (Moore et al 2015))
    p_phy_O2C = 172.0/122.0     # mol/mol
    p_phy_minchlc = 0.004       # minimum chlorophyll : Carbon ratio
    p_phy_maxchlc = 0.040       # maximum chlorophyll : Carbon ratio (reduced by cooler temperatures)
    p_phy_optFeC = 7e-6         # optimal Fe:C quota of phytoplankton
    p_phy_maxFeC = 40e-6        # maximum Fe:C quota of phytoplankton
    
    ##########################
    ####### Zooplannkton  ####
    ##########################
    p_zoo_grz = 3.0             # scaler for rate of zooplankton grazing
    p_zoo_capcoef = 1.0 / d2s   # prey capture efficiency coefficient (m6 / (µM N)2 * s))
    p_zoo_prefdet = 0.1
    p_zoo_qmort = 0.5 / d2s    # quadratic mortality of zooplankton (1 / (µM N * day))
    p_zoo_excre = 0.05         # rate of excretion by zooplankton (/day)
    p_zoo_assim = 0.7         # zooplankton assimilation efficiency
    # Anderson et al. 2021 
    p_zoo_betaV = 0.62      # absorption efficiency of protein
    p_zoo_betaH = 0.62      # absorption efficiency of carbohydrate
    p_protein_CN = 3.7
    p_zoo_CN = 5.9
    ######################
    ####### Detritus  ####
    ######################
    p_det_rem = 0.048           # remineralisation rate of detritus (/s)
    ###################
    ####### CaCO3  ####
    ###################
    p_cal_rem = 0.001714 / d2s  # remineralisation rate of CaCO3 (/s)
    p_cal_fra = 0.062           # fraction of inorganic
    
    
    
    #----------------------------------------------------------------------
    # (1) Light field
    #----------------------------------------------------------------------
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
    zchl = np.fmax(0.01, np.fmin(10.0, pchl_loc))
    ichl = ( 40 + 20 * np.log10(zchl) ).astype(np.int64)
    ek_blu = p_Chl_k[ichl,1]
    ek_gre = p_Chl_k[ichl,2]
    ek_red = p_Chl_k[ichl,3]
    # iii)  Using the attenuation coefficients, estimate the amount of RGB light available
    par_blu = np.zeros(len(z_zgrid)); par_gre = np.zeros(len(z_zgrid)); par_red = np.zeros(len(z_zgrid))
    par_blu[0] = par * np.exp(-0.5 * ek_blu[0] * z_dz) * p_PAR_bio
    par_gre[0] = par * np.exp(-0.5 * ek_gre[0] * z_dz) * p_PAR_bio
    par_red[0] = par * np.exp(-0.5 * ek_red[0] * z_dz) * p_PAR_bio
    for kk in np.arange(1,len(z_zgrid)):
        par_blu[kk] = par_blu[kk-1] * np.exp(-ek_blu[kk-1] * z_dz)
        par_gre[kk] = par_gre[kk-1] * np.exp(-ek_gre[kk-1] * z_dz)
        par_red[kk] = par_red[kk-1] * np.exp(-ek_red[kk-1] * z_dz)
    # iv)   Find light available for Phytoplankton
    par_tot = par_blu + par_gre + par_red
    par_phy = 1.85 * par_blu + 0.68 * par_gre + 0.46 * par_red
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
        zval = np.fmax(1.0, p_hrday)
        if (-z_zgrid[kk] <= z_mld):
            zval = zval * np.fmin(1.0, -z_eup/z_mld)
        chl_lday[kk] = zval / 24.0
        phy_lday[kk] = 1.5 * zval / (12.0 + zval)
    
    
    #----------------------------------------------------------------------
    # (2) Dissolve iron chemistry
    #----------------------------------------------------------------------
    # Determine equilibrium fractionation of total dFe into Fe` and L-Fe
    ligand = 0.7 * 1e-9   # nM * mol per nM
    fe_keq = 10**( 17.27 - 1565.7 / tck ) 
    fe_III = ( -( 1. + fe_keq * ligand - fe_keq * dfe_loc*1e-6 ) + \
              ( ( 1. + fe_keq * ligand - fe_keq * dfe_loc*1e-6 )**2 \
                 + 4. * dfe_loc*1e-6 * fe_keq)**0.5 ) / ( 2. * fe_keq ) * 1e6
    fe_lig = np.fmax(0.0, dfe_loc - fe_III)
    
    # Precipitation of Fe` (creation of nanoparticles)
    sal = 35.0
    zval = 19.924 * sal / ( 1000. - 1.005 * sal)
    fesol1 = 10**(-13.486 - 0.1856*zval**0.5 + 0.3073*zval + 5254.0/np.fmax(tck, 278.15) )
    fesol2 = 10**(2.517 - 0.8885*zval**0.5 + 0.2139*zval - 1320.0/np.fmax(tck, 278.15) )
    fesol3 = 10**(0.4511 - 0.3305*zval**0.5 - 1996.0/np.fmax(tck, 278.15) )
    fesol4 = 10**(-0.2965 - 0.7881*zval**0.5 - 4086.0/np.fmax(tck, 278.15) )
    fesol5 = 10**(4.4466 - 0.8505*zval**0.5 - 7980.0/np.fmax(tck, 278.15) )
    hp = 10**(-7.9)
    fe3sol = fesol1 * ( hp**3 + fesol2 * hp**2 + fesol3 * hp + fesol4 + fesol5 / hp ) *1e6
    precip = np.fmax(0.0, (fe_III-fe3sol) ) * 0.01/d2s
    
    # Scavenging of Fe` --> Det
    partic = (det_loc + cal_loc)
    scaven = fe_III * (3e-5 + 0.005 * partic) / d2s 
    scadet = scaven * (det_loc+eps) / (partic+eps)

    # Coagulation of colloidal Fe (nM) into small and large particles (uM) 
    fe_col = fe_lig * 0.5
    doc_loc = 40e-6     # mol/L
    zval = ( 0.369 * 0.3 * doc_loc + 102.4 * det_loc*1e-6 ) + ( 114. * 0.3 * doc_loc )
    zval[-z_zgrid > z_mld] = zval[-z_zgrid > z_mld]*0.01
    fe2det = fe_col * zval / d2s
    
    #precip = precip*0.
    #scaven = scaven*0.
    #scadet = scadet*0.
    #fe2det = fe2det*0.

    
    #----------------------------------------------------------------------
    # (3) Primary production of nanophytoplankton
    #----------------------------------------------------------------------
    # 1.1 Calculate the temperature-dependent maximum growth rate (/s)
    phy_mumax = Tfunc_auto * 1
    # 1.2 Nutrient limitation terms
    phy_conc1 = np.fmax(0.0, phy_loc - p_phy_biothresh)
    phy_conc2 = phy_loc - phy_conc1
    phy_k_nit = np.fmax(p_phy_kn, 
                        ( phy_conc2 * p_phy_kn + phy_conc1 * 3 * p_phy_kn ) / ( phy_loc + eps ) )
    phy_k_dfe = np.fmax(p_phy_kfe, 
                        ( phy_conc2 * p_phy_kfe + phy_conc1 * 3 * p_phy_kfe ) / ( phy_loc + eps ) )
    
    phy_limnit = no3_loc / (no3_loc + phy_k_nit)
    phy_minFeC = 0.0016 / 55.85 * pchl_loc/(phy_loc+eps) + \
                 1.21e-5 * 14.0 / 55.85 / 7.625 * 0.5 * 1.5 * phy_limnit + \
                 1.15e-4 * 14.0 / 55.85 / 7.625 * 0.5 * phy_limnit
    phy_limdfe = np.fmin(1.0, np.fmax(0.0, ( phyfe_loc/(phy_loc+eps) - phy_minFeC ) / p_phy_optFeC ))
    phy_limnut = np.fmin(phy_limnit, phy_limdfe)
    
    # 1.3 Calculate growth limitation by light
    chlc_ratio = pchl_loc / (phy_loc * 12.0 + eps)
    phy_pisl = np.fmax(p_alpha * chlc_ratio, p_alpha * p_phy_minchlc)
    # Original
    #phy_pisl2 = phy_pisl / ( (1.0/d2s + p_phy_lmort) * phy_lday * d2s ) # alter slope accounting for respiration
    #phy_lpar = 1.0 - np.exp( (-phy_pisl2 * par_phy) )
    # New
    phy_pisl2 = phy_pisl / ( (1.0 + p_phy_lmort) * np.fmax(p_hrday, 1.0)/24.0 ) # alter slope accounting for respiration
    phy_lpar = (1.0 - np.exp( -phy_pisl2 * par_phy ) ) * phy_lday
    phy_pro = phy_mumax * phy_lpar
    # 1.4 Apply nutrient limitation to phytoplankton growth to get realised growth rate
    phy_mu = phy_pro * phy_limnut
    # 1.5 Growth rate in chlorophyll
    chl_pisl = phy_pisl / (phy_mumax * d2s * np.fmax(p_hrday, 1.0)/24.0 * (1.0-phy_limnut) )   # decrease slope if growing fast, increase if not much daylight 
    chl_lpar = ( 1.0 - np.exp( -chl_pisl * par_phy_mld ) ) * phy_lday
    chl_pro = phy_mumax * chl_lpar * phy_limnut
    chl_maxq = np.fmin(p_phy_maxchlc, (p_phy_maxchlc / (1.0 - 1.14/43.4 * tc)) * (1.0 - 1.14/43.4 * 20))
    chl_maxq = p_phy_maxchlc * 1
    chl_mumin = phy_mu * phy_loc * 12 * p_phy_minchlc   # units of mg Chl m-3 s-1
    #chl_mu = phy_mu * phy_loc * 12 * d2s * chl_pro
    #for kk in np.arange(0,len(z_zgrid)-1):
    #    if chl_mu[kk] > 0.0:
    #        chl_mu[kk] = chl_mumin[kk] + ( (chl_maxq[kk] - p_phy_minchlc) * chl_mu[kk] ) \
    #                                   / ( phy_pisl[kk] * par_phy_mld[kk]/phy_lday[kk] )
    chl_mumax = phy_mu * phy_loc * 12 * chl_maxq   # units of mg Chl m-3 s-1
    chl_mu = (chl_mumax - chl_mumin) * chl_lpar * phy_limnut
    for kk in np.arange(0,len(z_zgrid)-1):
        if par_phy_mld[kk] > 0.0:
            chl_mu[kk] = chl_mumin[kk] + chl_mu[kk] / (phy_pisl[kk] * par_phy_mld[kk]) 
    #for kk in np.arange(0,len(z_zgrid)-1):
    #    if chl_mu[kk] > 0.0:
    #        chl_mu[kk] = chl_mumin[kk] + ( (chl_maxq[kk] - p_phy_minchlc) * chl_mu[kk] ) \
    #                                   / ( phy_pisl[kk] * par_phy_mld[kk]/phy_lday[kk] )
    # 1.6 Collect terms for phytoplankton
    phy_dicupt = phy_mu * phy_loc                                           # f11
    phy_no3upt = phy_dicupt / p_phy_CN
    phy_oxyprd = phy_dicupt * p_phy_O2C


    #----------------------------------------------------------------------
    # (4) Dissolved Iron uptake by phytoplankton
    #----------------------------------------------------------------------
    phy_maxQFe = phy_loc * p_phy_maxFeC
    phy_Feupt_upreg   = (4.0 - 4.5 * phy_limdfe / (phy_limdfe + 0.5))
    phy_Feupt_downreg = np.fmax(0.0, (1.0 - phyfe_loc/(phy_maxQFe+eps)) \
                                / np.abs(1.05 - phyfe_loc/(phy_maxQFe+eps)) )
    phy_dfeupt = phy_loc * phy_mumax * p_phy_maxFeC * \
                 dfe_loc / (dfe_loc + phy_k_dfe) * phy_Feupt_downreg * phy_Feupt_upreg

    #----------------------------------------------------------------------
    # (5) Grazing by zooplankton
    #----------------------------------------------------------------------
    # 1.1 Calculate temperature-dependent maximum grazing rate
    zoo_mumax = p_zoo_grz * Tfunc_hete
    # 1.2 Calculate prey capture rate function (/s)
    prey = phy_loc + det_loc*p_zoo_prefdet
    zoo_capt = p_zoo_capcoef * prey * prey
    # 1.3 Calculate the realised grazing rate of zooplankton
    zoo_mu = zoo_mumax * zoo_capt / (zoo_mumax + zoo_capt)
        # The functional form of grazing creates increasing half-saturation 
        # values as maximum growth rates increase. It ensures that zooplankton
        # always have the same affinity for phytoplankton irrespective of temp
    # 1.4 Collect terms
    zoo_phygrz = zoo_mu * zoo_loc * phy_loc / (prey + eps)                  # f21
    zoo_detgrz = zoo_mu * zoo_loc * det_loc*p_zoo_prefdet / (prey + eps)    # f21
    zoo_zooexc = zoo_loc * p_zoo_excre * Tfunc_hete                         # f31
    
    
    #----------------------------------------------------------------------
    # (6) Mortality terms
    #----------------------------------------------------------------------
    phy_lmort = p_phy_lmort * phy_loc * Tfunc_hete * np.fmin(1.0, np.fmax(0.0, phy_limnut/0.3))
    phy_qmort = p_phy_qmort * phy_loc * phy_loc # Not temperature-dependent?   # f23
    zoo_qmort = p_zoo_qmort * zoo_loc * zoo_loc # Not temperature-dependent?   # f32
    
    
    #----------------------------------------------------------------------
    # (7) Detritus and CaCO3
    #----------------------------------------------------------------------
    det_remin = p_det_rem * det_loc * Tfunc_hete                          # f41
    cal_remin = p_cal_rem * cal_loc                                       # f51


    #----------------------------------------------------------------------
    # (8)  Tracer equations (tendencies) Sources - Sinks
    #----------------------------------------------------------------------
    
    # ratios 
    phyFeC = phyfe_loc / (phy_loc + eps)
    zooFeC = zoofe_loc / (zoo_loc + eps)
    detFeC = detfe_loc / (det_loc + eps)
    
    
    ddt_o2 = phy_oxyprd - (phy_lmort + zoo_zooexc + det_remin) * p_phy_O2C
    
    ddt_no3 = (phy_lmort + zoo_zooexc + det_remin)/p_phy_CN - phy_no3upt
    
    ddt_dfe = (phy_lmort*phyFeC + zoo_zooexc*zooFeC + det_remin*detFeC) - phy_dfeupt \
              - precip - scaven - fe2det
              
    ddt_phy = phy_dicupt - (phy_lmort + phy_qmort + zoo_phygrz)
    ddt_phyfe = phy_dfeupt - (phy_lmort + phy_qmort + zoo_phygrz)*phyFeC
    
    ddt_zoo = (zoo_phygrz + zoo_detgrz)*p_zoo_assim - (zoo_zooexc + zoo_qmort)
    ddt_zoofe = (zoo_phygrz*phyFeC + zoo_detgrz*detFeC)*p_zoo_assim - (zoo_zooexc + zoo_qmort)*zooFeC
    
    ddt_det = (zoo_phygrz + zoo_detgrz)*(1.0-p_zoo_assim) + phy_qmort + zoo_qmort - det_remin - zoo_detgrz
    ddt_detfe = (zoo_phygrz*phyFeC + zoo_detgrz*detFeC)*(1.0-p_zoo_assim) \
                + phy_qmort*phyFeC + zoo_qmort*zooFeC \
                - (det_remin + zoo_detgrz)*detFeC \
                + scadet + fe2det
    
    ddt_cal = (zoo_phygrz*(1.0-p_zoo_assim) + phy_qmort + zoo_qmort)*p_cal_fra - cal_remin
    
    ddt_alk = -ddt_no3 - ddt_cal * 2.0
    
    ddt_dic = (ddt_no3 * p_phy_CN) - ddt_cal
    
    ddt_pchl = chl_mu - chlc_ratio * (phy_lmort + phy_qmort + zoo_qmort)*12 


    return [ddt_o2, ddt_no3, ddt_dfe, ddt_phy, ddt_zoo, ddt_det, \
            ddt_cal, ddt_alk, ddt_dic, ddt_pchl, ddt_phyfe, ddt_zoofe, ddt_detfe, \
            phy_mu, zoo_mu, phy_lmort, phy_qmort, \
            chlc_ratio, phyFeC*1e6, zooFeC*1e6, detFeC*1e6]
 
