#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 10:52:47 2023

@author: pbuchanan
"""

import numpy as np
from numba import jit

@jit(nopython=True)
def bgc_sms(o2, n2, no3, no2, nh4, n2o, po4, dic, \
            doc, dop, don, rdoc, rdop, rdon, \
            phy, dia, \
            aoa, nob, aox, \
            nar, nai, nir, nos, \
            fnar, fnai, fnir, fnos, \
            zoo, mes, \
            z_dz, z_zgrid, t_poc_flux_top, \
            conserving, darwin_arch):
   
    # TEMPORARY
    o2_loc = o2[1,:]
    n2_loc = n2[1,:]
    no3_loc = no3[1,:]
    no2_loc = no2[1,:]
    nh4_loc = nh4[1,:]
    n2o_loc = n2o[1,:]
    po4_loc = po4[1,:]
    dic_loc = dic[1,:]
    doc_loc = doc[1,:]
    dop_loc = dop[1,:]
    don_loc = don[1,:]
    rdoc_loc = rdoc[1,:]
    rdop_loc = rdop[1,:]
    rdon_loc = rdon[1,:]
    phy_loc = phy[1,:]
    dia_loc = dia[1,:]
    aoa_loc = aoa[1,:]
    nob_loc = nob[1,:]
    aox_loc = aox[1,:]
    nar_loc = nar[1,:]
    nai_loc = nai[1,:]
    nir_loc = nir[1,:]
    nos_loc = nos[1,:]
    fnar_loc = fnar[1,:]
    fnai_loc = fnai[1,:]
    fnir_loc = fnir[1,:]
    fnos_loc = fnos[1,:]
    zoo_loc = zoo[1,:]
    mes_loc = mes[1,:]
    
    # Make tracer values zero if negative
    o2 = np.fmax(o2_loc, 0.0)
    n2 = np.fmax(n2_loc, 0.0)
    no3 = np.fmax(no3_loc, 0.0)
    no2 = np.fmax(no2_loc, 0.0)
    nh4 = np.fmax(nh4_loc, 0.0)
    n2o = np.fmax(n2o_loc, 0.0)
    po4 = np.fmax(po4_loc, 0.0)
    dic = np.fmax(dic_loc, 0.0)
    doc = np.fmax(doc_loc, 0.0)
    dop = np.fmax(dop_loc, 0.0)
    don = np.fmax(don_loc, 0.0)
    rdoc = np.fmax(rdoc_loc, 0.0)
    rdop = np.fmax(rdop_loc, 0.0)
    rdon = np.fmax(rdon_loc, 0.0)
    phy = np.fmax(phy_loc, 0.0)
    dia = np.fmax(dia_loc, 0.0)
    aoa = np.fmax(aoa_loc, 0.0)
    nob = np.fmax(nob_loc, 0.0)
    aox = np.fmax(aox_loc, 0.0)
    nar = np.fmax(nar_loc, 0.0)
    nai = np.fmax(nai_loc, 0.0)
    nir = np.fmax(nir_loc, 0.0)
    nos = np.fmax(nos_loc, 0.0)
    fnar = np.fmax(fnar_loc, 0.0)
    fnai = np.fmax(fnai_loc, 0.0)
    fnir = np.fmax(fnir_loc, 0.0)
    fnos = np.fmax(fnos_loc, 0.0)
    zoo = np.fmax(zoo_loc, 0.0)
    mes = np.fmax(mes_loc, 0.0)
    
    
    #----------------------------------------------------------------------
    # (0) Define important parameters
    #----------------------------------------------------------------------
    d2s = 86400.0
    eps = 1e-16
    # temperature-dependency
    tc = 12.0
    Tfunc = 1.7**( ((tc + 273.15) - (30 + 273.15))/10.0)
    Tfunc_diat = 1.55**( ((tc + 273.15) - (30 + 273.15))/10.0)
    # Stoichiometry of phytoplankon / zooplankton
    p_pom_CP = 106.0
    p_pom_CN = 106.0/16.0 
    p_pom_O2C = 122.0/106.0
    # Light terms for phytoplankton growth 
    p_PAR_k = 0.05      # attenuation coefficient in 1/m
    p_PAR_in = 50       # incoming photosynthetically active radiation (W/m2)
    p_alpha = 0.4 / d2s # Initial slope of the PI curve (mmol C m-2 per mg Chl W sec)
    ####### Nanophytoplannkton  ####
    ################################
    # PO4 + DIC + NH4 + NO2 + NO3 --> POC + O2
    p_muphy = 3.54 / d2s * Tfunc
    p_kphy_po4 = 0.0075
    p_kphy_nh4 = 0.0025
    p_kphy_no2 = 0.03
    p_kphy_no3 = 0.2
    p_phy_ChlC = 0.1
    p_phy_bmin = 0.0
    p_phy_lmort = 0.1 / d2s * Tfunc
    p_phy_qmort = 0.1 / d2s * Tfunc
    p_phy2dom = 0.15
    if conserving:
        p_phy2pom = 0.0
    else:
        p_phy2pom = 0.2
    ####### Diatoms  ####
    #####################
    # PO4 + DIC + NH4 + NO2 + NO3 --> POC + O2
    p_mudia = 3.54 / d2s * Tfunc_diat
    p_kdia_po4 = 0.06
    p_kdia_nh4 = 0.02
    p_kdia_no2 = 0.24
    p_kdia_no3 = 0.6
    p_dia_ChlC = 0.2
    p_dia_bmin = 0.0
    p_dia_lmort = 0.1 / d2s * Tfunc
    p_dia_qmort = 0.1 / d2s * Tfunc
    p_dia2dom = 0.15
    if conserving:
        p_dia2pom = 0.0
    else:
        p_dia2pom = 0.4

    # Sinking terms
    p_martinb = -0.86
    p_poc_diss = 88
    p_odzfac = 3.3
    
    # DOM terms
    f_lability = 1.0
    r_rdom2dom = 0.0 / d2s
    
    ####### Ammonium oxidation ####
    ###############################
    # Ammox: NH4 --> NO2
    p_muaoa = 1.0 / d2s * Tfunc
    p_kaoa_nh4 = 0.1
    p_paoa_o2 = 293.0 / d2s
    p_yaoa_nh4 = 11.0
    p_yaoa_oxy = 15.5
    p_aoa_CN = 4.0
    p_aoa_CP = 55.0
    ####### Nitrite oxidation #####
    ###############################
    # Nitrox: NO2 --> NO3
    p_munob = 2.0 / d2s * Tfunc
    p_knob_no2 = 0.1 
    p_pnob_o2 = 71.0 / d2s 
    p_ynob_no2 = 27.8 
    p_ynob_oxy = 12.9 
    p_nob_CN = 4.0 
    p_nob_CP = 55.0 
    ########## Anammox ############
    ###############################
    p_muaox = 0.5 / d2s * Tfunc
    p_kaox_nh4 = 0.45 
    p_kaox_no2 = 0.45 
    p_yaox_nh4 = 14.0 
    p_yaox_no2 = 16.2 
    p_paox_no3 = 2.0 
    p_aox_CN = 5.0 
    p_aox_CP = 55.0 
    ###### Facultative NO3 reducers #####
    #####################################
    # NAR: NO3 --> NO2
    p_munar = 1.0 / d2s * Tfunc 
    p_mufnar = 4.0 / d2s * Tfunc 
    p_knar_doc = 10.0 
    p_knar_rdoc = 100.0 
    p_knar_no3 = 4.0 
    p_pnar_o2 = 450.0 / d2s 
    p_ynar_aer = 5.9 
    p_ynar_oxy = 5.4 
    p_ynar_ana = 8.6 
    p_ynar_no3 = 16.8 
    p_nar_CN = 4.5 
    p_nar_CP = 35.0 
    ###### Facultative NO3 reducers #####
    #####################################
    # NAI: NO3 --> N2O
    p_munai = 1.0 / d2s * Tfunc
    p_mufnai = 4.0 / d2s * Tfunc 
    p_knai_doc = 10.0 
    p_knai_rdoc = 100.0 
    p_knai_no3 = 4.0 
    p_pnai_o2 = 450.0 / d2s 
    p_ynai_aer = 5.9 
    p_ynai_oxy = 5.4 
    p_ynai_ana = 8.6 
    p_ynai_no3 = 16.8 
    p_nai_CN = 4.5 
    p_nai_CP = 35.0 
    ###### Facultative NO2 reducers #####
    #####################################
    # NIR: NO2 --> N2O
    p_munir = 1.0 / d2s * Tfunc
    p_mufnir = 4.0 / d2s * Tfunc 
    p_knir_doc = 10.0 
    p_knir_rdoc = 100.0 
    p_knir_no2 = 4.0 
    p_pnir_o2 = 450.0 / d2s 
    p_ynir_aer = 5.9 
    p_ynir_oxy = 5.4 
    p_ynir_ana = 6.5 
    p_ynir_no2 = 12.1 
    p_nir_CN = 4.5 
    p_nir_CP = 35.0 
    ###### Facultative N2O reducers #####
    #####################################
    # NOS: N2O --> N2
    p_munos = 1.0 / d2s * Tfunc
    p_mufnos = 4.0 / d2s * Tfunc 
    p_knos_doc = 10.0 
    p_knos_rdoc = 100.0 
    p_knos_n2o = 0.3 
    p_pnos_o2 = 450.0 / d2s 
    p_ynos_aer = 5.9 
    p_ynos_oxy = 5.4 
    p_ynos_ana = 4.3 
    p_ynos_n2o = 14.8 
    p_nos_CN = 4.5 
    p_nos_CP = 35.0 
    ##### Small zooplankton #####
    p_mumax_zoo = 3.3 / d2s * Tfunc
    p_kzoo_phy = 6.2
    p_kzoo_dia = 6.2
    p_kzoo_aoa = 6.2
    p_kzoo_nob = 6.2
    p_kzoo_aox = 6.2
    p_kzoo_nar = 6.2
    p_kzoo_nai = 6.2
    p_kzoo_nir = 6.2
    p_kzoo_nos = 6.2
    p_zoogrz2zoo = 0.3
    p_zoogrz2dom = 0.06
    p_zoo_bmin = 0.00
    p_zoo_lmort = 0.05 / d2s * Tfunc
    p_zoo_qmort = 0.5 / d2s * Tfunc
    if conserving:
        p_zoo2dom = 0.15
        p_zoo2pom = 0.0
    else:
        p_zoo2dom = 0.1275
        p_zoo2pom = 0.25
    p_zoo2dic = 1.0 - p_zoo2dom - p_zoo2pom
    
    ##### Large zooplankton #####
    p_mumax_mes = 3.3 / d2s * Tfunc
    p_kmes_phy = 1.2
    p_kmes_dia = 1.2
    p_kmes_aoa = 2.4
    p_kmes_nob = 2.4
    p_kmes_aox = 2.4
    p_kmes_zoo = 2.4
    p_mesgrz2mes = 0.3
    p_mesgrz2dom = 0.06
    p_mes_bmin = 0.00
    p_mes_lmort = 0.1 / d2s * Tfunc
    p_mes_qmort = 0.5 / d2s * Tfunc
    if conserving:
        p_mes2dom = 0.1275
        p_mes2pom = 0.0
        p_mes2dic = 1.0 - p_mes2dom - p_mes2pom
    else:
        p_mes2dom = 0.1275
        p_mes2pom = 0.25
        p_mes2dic = 1.0 - p_mes2dom - p_mes2pom
    
    # Mortality of microbes
    p_mic_lmort = 0.025
    p_mic_qmort = 0.1 
    p_aoa_lmort = p_mic_lmort * p_muaoa # already has Tfunc applied
    p_aoa_qmort = p_mic_qmort / d2s * Tfunc
    p_nob_lmort = p_mic_lmort * p_munob # already has Tfunc applied
    p_nob_qmort = p_mic_qmort / d2s * Tfunc
    p_aox_lmort = p_mic_lmort * p_muaox # already has Tfunc applied
    p_aox_qmort = p_mic_qmort / d2s * Tfunc
    p_nar_lmort = p_mic_lmort * p_munar # already has Tfunc applied
    p_fnar_lmort = p_mic_lmort * p_mufnar # already has Tfunc applied
    p_nar_qmort = p_mic_qmort / d2s * Tfunc
    p_nai_lmort = p_mic_lmort * p_munai # already has Tfunc applied
    p_fnai_lmort = p_mic_lmort * p_mufnai # already has Tfunc applied
    p_nai_qmort = p_mic_qmort / d2s * Tfunc
    p_nir_lmort = p_mic_lmort * p_munir # already has Tfunc applied
    p_fnir_lmort = p_mic_lmort * p_mufnir # already has Tfunc applied
    p_nir_qmort = p_mic_qmort / d2s * Tfunc
    p_nos_lmort = p_mic_lmort * p_munos # already has Tfunc applied 
    p_fnos_lmort = p_mic_lmort * p_mufnos # already has Tfunc applied 
    p_nos_qmort = p_mic_qmort / d2s * Tfunc
    p_che_bmin = 0.0
    p_het_bmin = 0.0
    p_che2dom = 0.5
    p_het2dom = 0.5
    if conserving:
        p_che2poc = 0.0
        p_het2poc = 0.0
    else:
        p_che2poc = 0.2
        p_het2poc = 0.2
    
    p_denfac = 0.90
    


    #----------------------------------------------------------------------
    # (1) Primary production of nanophytoplankton
    #----------------------------------------------------------------------
    # 1.1 Calculate PAR availability
    par = p_PAR_in * np.exp(p_PAR_k * z_zgrid)  # z_zgrid is already negative 
    # 1.2 Calculate growth on light
    phy_lpar = 1.0 - np.exp( (-1.0 * p_alpha * par * p_phy_ChlC) / p_muphy ) 
    # 1.3 Calculate and apply nutrient limitation terms
    phy_lno3 = (no3/p_kphy_no3) / ( 1.0 + (no3/p_kphy_no3) + (no2/p_kphy_no2) + (nh4/p_kphy_nh4))
    phy_lno2 = (no2/p_kphy_no2) / ( 1.0 + (no3/p_kphy_no3) + (no2/p_kphy_no2) + (nh4/p_kphy_nh4))
    phy_lnh4 = (nh4/p_kphy_nh4) / ( 1.0 + (no3/p_kphy_no3) + (no2/p_kphy_no2) + (nh4/p_kphy_nh4))
    phy_lnit = phy_lno3 + phy_lno2 + phy_lnh4
    phy_lpo4 = po4 / (po4 + p_kphy_po4)
    # 1.4 Calculate the realised growth rate
    mu_phy = p_muphy * phy_lpar * np.fmin(phy_lnit, phy_lpo4)
    # 1.5 Collect nutrient assimilation terms
    phy_no3upt = phy_lno3 / phy_lnit * mu_phy * phy / p_pom_CN
    phy_no2upt = phy_lno2 / phy_lnit * mu_phy * phy / p_pom_CN
    phy_nh4upt = phy_lnh4 / phy_lnit * mu_phy * phy / p_pom_CN
    phy_po4upt = mu_phy * phy / p_pom_CP

    #----------------------------------------------------------------------
    # (2) Primary production of diatoms
    #----------------------------------------------------------------------
    # 1.1 Calculate PAR availability
    par = p_PAR_in * np.exp(p_PAR_k * z_zgrid)  # z_zgrid is already negative 
    # 1.2 Calculate growth on light
    dia_lpar = 1.0 - np.exp( (-1.0 * p_alpha * par * p_dia_ChlC) / p_mudia ) 
    # 1.3 Calculate and apply nutrient limitation terms
    dia_lno3 = (no3/p_kdia_no3) / ( 1.0 + (no3/p_kdia_no3) + (no2/p_kdia_no2) + (nh4/p_kdia_nh4))
    dia_lno2 = (no2/p_kdia_no2) / ( 1.0 + (no3/p_kdia_no3) + (no2/p_kdia_no2) + (nh4/p_kdia_nh4))
    dia_lnh4 = (nh4/p_kdia_nh4) / ( 1.0 + (no3/p_kdia_no3) + (no2/p_kdia_no2) + (nh4/p_kdia_nh4))
    dia_lnit = dia_lno3 + dia_lno2 + dia_lnh4
    dia_lpo4 = po4 / (po4 + p_kdia_po4)
    # 1.4 Calculate the realised growth rate
    mu_dia = p_mudia * dia_lpar * np.fmin(dia_lnit, dia_lpo4)
    # 1.5 Collect nutrient assimilation terms
    dia_no3upt = dia_lno3 / dia_lnit * mu_dia * dia / p_pom_CN
    dia_no2upt = dia_lno2 / dia_lnit * mu_dia * dia / p_pom_CN
    dia_nh4upt = dia_lnh4 / dia_lnit * mu_dia * dia / p_pom_CN
    dia_po4upt = mu_dia * dia / p_pom_CP

    #----------------------------------------------------------------------
    # (3) Facultative Heterotrophs
    #----------------------------------------------------------------------
    #!!! Respiration rate based on DOC, NO3 and O3
    zlimnar_doc = (doc / p_knar_doc) / (1. + (doc / p_knar_doc) + (rdoc / p_knar_rdoc)) 
    zlimnar_rdoc = (rdoc / p_knar_rdoc) / (1. + (doc / p_knar_doc) + (rdoc / p_knar_rdoc)) 
    
    zuptoxy = p_pnar_o2 * o2 
    zuptdoc = p_munar * p_ynar_aer * (zlimnar_doc + zlimnar_rdoc) 
    mu_nar_aer = np.fmax(0.0, np.fmin( (zuptoxy / p_ynar_oxy), (zuptdoc / p_ynar_aer) ))
    zuptno3 = p_munar * p_denfac * p_ynar_no3 * (no3 / (no3 + p_knar_no3)) 
    zuptdoc = p_munar * p_denfac * p_ynar_ana * (zlimnar_doc + zlimnar_rdoc) 
    mu_nar_ana = np.fmax(0.0, np.fmin( (zuptno3 / p_ynar_no3), (zuptdoc / p_ynar_ana) ))
    
    #!!! Fast growers
    zuptoxy = p_pnar_o2 * o2 
    zuptdoc = p_mufnar * p_ynar_aer * (zlimnar_doc + zlimnar_rdoc) 
    mu_fnar_aer = np.fmax(0.0, np.fmin( (zuptoxy / p_ynar_oxy), (zuptdoc / p_ynar_aer) ))
    zuptno3 = p_mufnar * p_denfac * p_ynar_no3 * (no3 / (no3 + p_knar_no3)) 
    zuptdoc = p_mufnar * p_denfac * p_ynar_ana * (zlimnar_doc + zlimnar_rdoc) 
    mu_fnar_ana = np.fmax(0.0, np.fmin( (zuptno3 / p_ynar_no3), (zuptdoc / p_ynar_ana) ))
    
    
    #!!! Respiration rate based on DOC, NO3 and O3
    zlimnai_doc = (doc / p_knai_doc) / (1. + (doc / p_knai_doc) + (rdoc / p_knai_rdoc)) 
    zlimnai_rdoc = (rdoc / p_knai_rdoc) / (1. + (doc / p_knai_doc) + (rdoc / p_knai_rdoc)) 
    
    zuptoxy = p_pnai_o2 * o2 
    zuptdoc = p_munai * p_ynai_aer * (zlimnai_doc + zlimnai_rdoc) 
    mu_nai_aer = np.fmax(0.0, np.fmin( (zuptoxy / p_ynai_oxy), (zuptdoc / p_ynai_aer) ))
    zuptno3 = p_munai * p_denfac * p_ynai_no3 * (no3 / (no3 + p_knai_no3)) 
    zuptdoc = p_munai * p_denfac * p_ynai_ana * (zlimnai_doc + zlimnai_rdoc) 
    mu_nai_ana = np.fmax(0.0, np.fmin( (zuptno3 / p_ynai_no3), (zuptdoc / p_ynai_ana) ))

    #!!! fast growers
    zuptoxy = p_pnai_o2 * o2 
    zuptdoc = p_mufnai * p_ynai_aer * (zlimnai_doc + zlimnai_rdoc) 
    mu_fnai_aer = np.fmax(0.0, np.fmin( (zuptoxy / p_ynai_oxy), (zuptdoc / p_ynai_aer) ))
    zuptno3 = p_mufnai * p_denfac * p_ynai_no3 * (no3 / (no3 + p_knai_no3)) 
    zuptdoc = p_mufnai * p_denfac * p_ynai_ana * (zlimnai_doc + zlimnai_rdoc) 
    mu_fnai_ana = np.fmax(0.0, np.fmin( (zuptno3 / p_ynai_no3), (zuptdoc / p_ynai_ana) ))


    #!!! Respiration rate based on DOC, NO2 and O3
    zlimnir_doc = (doc / p_knir_doc) / (1. + (doc / p_knir_doc) + (rdoc / p_knir_rdoc)) 
    zlimnir_rdoc = (rdoc / p_knir_rdoc) / (1. + (doc / p_knir_doc) + (rdoc / p_knir_rdoc)) 
    
    zuptoxy = p_pnir_o2 * o2 
    zuptdoc = p_munir * p_ynir_aer * (zlimnir_doc + zlimnir_rdoc) 
    mu_nir_aer = np.fmax(0.0, np.fmin( (zuptoxy / p_ynir_oxy), (zuptdoc / p_ynir_aer) ))
    zuptno2 = p_munir * p_denfac * p_ynir_no2 * (no2 / (no2 + p_knir_no2)) 
    zuptdoc = p_munir * p_denfac * p_ynir_ana * (zlimnir_doc + zlimnir_rdoc) 
    mu_nir_ana = np.fmax(0.0, np.fmin( (zuptno2 / p_ynir_no2), (zuptdoc / p_ynir_ana) ))
    
    #!!! fast growers
    zuptoxy = p_pnir_o2 * o2 
    zuptdoc = p_mufnir * p_ynir_aer * (zlimnir_doc + zlimnir_rdoc) 
    mu_fnir_aer = np.fmax(0.0, np.fmin( (zuptoxy / p_ynir_oxy), (zuptdoc / p_ynir_aer) ))
    zuptno2 = p_mufnir * p_denfac * p_ynir_no2 * (no2 / (no2 + p_knir_no2)) 
    zuptdoc = p_mufnir * p_denfac * p_ynir_ana * (zlimnir_doc + zlimnir_rdoc) 
    mu_fnir_ana = np.fmax(0.0, np.fmin( (zuptno2 / p_ynir_no2), (zuptdoc / p_ynir_ana) ))
    
    #!!! Respiration rate based on DOC, N2O and O3
    zlimnos_doc = (doc / p_knos_doc) / (1. + (doc / p_knos_doc) + (rdoc / p_knos_rdoc)) 
    zlimnos_rdoc = (rdoc / p_knos_rdoc) / (1. + (doc / p_knos_doc) + (rdoc / p_knos_rdoc)) 
    
    zuptoxy = p_pnos_o2 * o2 
    zuptdoc = p_munos * p_ynos_aer * (zlimnos_doc + zlimnos_rdoc) 
    mu_nos_aer = np.fmax(0.0, np.fmin( (zuptoxy / p_ynos_oxy), (zuptdoc / p_ynos_aer) ))
    zuptn2o = p_munos * p_denfac * p_ynos_n2o * (n2o / (n2o + p_knos_n2o)) 
    zuptdoc = p_munos * p_denfac * p_ynos_ana * (zlimnos_doc + zlimnos_rdoc) 
    mu_nos_ana = np.fmax(0.0, np.fmin( (zuptn2o / p_ynos_n2o), (zuptdoc / p_ynos_ana) ))
    
    #!!! fast growers
    zuptoxy = p_pnos_o2 * o2 
    zuptdoc = p_mufnos * p_ynos_aer * (zlimnos_doc + zlimnos_rdoc) 
    mu_fnos_aer = np.fmax(0.0, np.fmin( (zuptoxy / p_ynos_oxy), (zuptdoc / p_ynos_aer) ))
    zuptn2o = p_mufnos * p_denfac * p_ynos_n2o * (n2o / (n2o + p_knos_n2o)) 
    zuptdoc = p_mufnos * p_denfac * p_ynos_ana * (zlimnos_doc + zlimnos_rdoc) 
    mu_fnos_ana = np.fmax(0.0, np.fmin( (zuptn2o / p_ynos_n2o), (zuptdoc / p_ynos_ana) ))
        
    
    mu_nar = np.fmax( mu_nar_aer, mu_nar_ana )
    mu_nai = np.fmax( mu_nai_aer, mu_nai_ana )
    mu_nir = np.fmax( mu_nir_aer, mu_nir_ana )
    mu_nos = np.fmax( mu_nos_aer, mu_nos_ana )
    mu_fnar = np.fmax( mu_fnar_aer, mu_fnar_ana )
    mu_fnai = np.fmax( mu_fnai_aer, mu_fnai_ana )
    mu_fnir = np.fmax( mu_fnir_aer, mu_fnir_ana )
    mu_fnos = np.fmax( mu_fnos_aer, mu_fnos_ana )
  
    mu_nar_fac = mu_nar_ana > mu_nar_aer
    mu_nai_fac = mu_nai_ana > mu_nai_aer
    mu_nir_fac = mu_nir_ana > mu_nir_aer
    mu_nos_fac = mu_nos_ana > mu_nos_aer
    mu_fnar_fac = mu_fnar_ana > mu_fnar_aer
    mu_fnai_fac = mu_fnai_ana > mu_fnai_aer
    mu_fnir_fac = mu_fnir_ana > mu_fnir_aer
    mu_fnos_fac = mu_fnos_ana > mu_fnos_aer
    
    Denitrif1 = mu_nar * nar * p_ynar_no3 * mu_nar_fac \
              + mu_fnar * fnar * p_ynar_no3 * mu_fnar_fac
    Denitrif2 = mu_nai * nai * p_ynai_no3 * mu_nai_fac \
              + mu_fnai * fnai * p_ynai_no3 * mu_fnai_fac 
    Denitrif3 = mu_nir * nir * p_ynir_no2 * mu_nir_fac \
              + mu_fnir * fnir * p_ynir_no2 * mu_fnir_fac
    Denitrif4 = mu_nos * nos * p_ynos_n2o * mu_nos_fac \
              + mu_fnos * fnos * p_ynos_n2o * mu_fnos_fac 


    #----------------------------------------------------------------------
    # (4) Ammonium oxidation (molN-units, mmolN/m3/s):
    #----------------------------------------------------------------------
    #!!! AO based on oxygen and NH4 concentration (loss of NH4)
    zuptoxy = p_paoa_o2 * o2
    zuptnh4 = p_muaoa * p_yaoa_nh4 * (nh4 / (nh4 + p_kaoa_nh4)) 
    mu_aoa = np.fmax(0.0, np.fmin( (zuptoxy / p_yaoa_oxy), (zuptnh4 /p_yaoa_nh4) ) )
    Ammox = mu_aoa * aoa * p_yaoa_nh4   

    #----------------------------------------------------------------------
    # (5) Nitrite oxidation (molN-units, mmolN/m3/s):
    #----------------------------------------------------------------------
    #!!! NO based on oxygen and NO2 concentration (loss of NO2)
    zuptoxy = p_pnob_o2 * o2 
    zuptno2 = p_munob * p_ynob_no2 * (no2 / (no2 + p_knob_no2)) 
    mu_nob = np.fmax(0.0, np.fmin( (zuptoxy / p_ynob_oxy), (zuptno2 /p_ynob_no2) ) )
    Nitrox = mu_nob * nob * p_ynob_no2
    
    #----------------------------------------------------------------------
    # (6) Anaerobic ammonium oxidation (molN2-units, mmolN2/m3/s):
    # Note Anammox is in units of NH4
    #----------------------------------------------------------------------
    zuptnh4 = p_muaox * p_yaox_nh4 * (nh4 / (nh4 + p_kaox_nh4)) 
    zuptno2 = p_muaox * p_yaox_no2 * (no2 / (no2 + p_kaox_no2)) 
    mu_aox = np.fmax(0.0, np.fmin( (zuptnh4 / p_yaox_nh4), (zuptno2 /p_yaox_no2) ) )
    Anammox = mu_aox * aox * p_yaox_nh4
    
    #----------------------------------------------------------------------
    # (7)  Mortality of phytoplankton
    #---------------------------------------------------------------------- 
    PHYprime = np.fmax(0.0, phy - p_phy_bmin)
    mort_phy = p_phy_lmort * PHYprime + p_phy_qmort * (PHYprime * phy)
    mort_phy_doc = mort_phy * (1.0-p_phy2pom) * p_phy2dom
    mort_phy_dop = mort_phy/p_pom_CP * (1.0-p_phy2pom) * p_phy2dom
    mort_phy_don = mort_phy/p_pom_CN * (1.0-p_phy2pom) * p_phy2dom
    mort_phy_dic = mort_phy * (1.0-p_phy2pom) * (1.0-p_phy2dom)
    mort_phy_dip = mort_phy/p_pom_CP * (1.0-p_phy2pom) * (1.0-p_phy2dom)
    mort_phy_din = mort_phy/p_pom_CN * (1.0-p_phy2pom) * (1.0-p_phy2dom)
    mort_phy_poc = mort_phy * (1.0-p_phy2dom) * p_phy2pom

    DIAprime = np.fmax(0.0, dia - p_dia_bmin)
    mort_dia = p_dia_lmort * DIAprime + p_dia_qmort * (DIAprime * dia)
    mort_dia_doc = mort_dia * (1.0-p_dia2pom) * p_dia2dom
    mort_dia_dop = mort_dia/p_pom_CP * (1.0-p_dia2pom) * p_dia2dom
    mort_dia_don = mort_dia/p_pom_CN * (1.0-p_dia2pom) * p_dia2dom
    mort_dia_dic = mort_dia * (1.0-p_dia2pom) * (1.0-p_dia2dom)
    mort_dia_dip = mort_dia/p_pom_CP * (1.0-p_dia2pom) * (1.0-p_dia2dom)
    mort_dia_din = mort_dia/p_pom_CN * (1.0-p_dia2pom) * (1.0-p_dia2dom)
    mort_dia_poc = mort_dia * (1.0-p_dia2dom) * p_dia2pom
    
    #----------------------------------------------------------------------
    # (8)  Mortality of microbes
    #---------------------------------------------------------------------- 
    AOAprime = np.fmax(0.0, aoa - p_che_bmin)
    NOBprime = np.fmax(0.0, nob - p_che_bmin)
    AOXprime = np.fmax(0.0, aox - p_che_bmin)
    mort_aoa = p_aoa_lmort * AOAprime + p_aoa_qmort * (AOAprime * aoa)
    mort_nob = p_nob_lmort * NOBprime + p_nob_qmort * (NOBprime * nob)
    mort_aox = p_aox_lmort * AOXprime + p_aox_qmort * (AOXprime * aox)
    mort_cheC = mort_aoa + mort_nob + mort_aox
    mort_cheP = mort_aoa/p_aoa_CP + mort_nob/p_nob_CP + mort_aox/p_aox_CP
    mort_cheN = mort_aoa/p_aoa_CN + mort_nob/p_nob_CN + mort_aox/p_aox_CN
    mort_che_doc = mort_cheC * p_che2dom * (1.0-p_che2poc)
    mort_che_dic = mort_cheC * (1.0-p_che2dom) * (1.0-p_che2poc)
    mort_che_dip = mort_cheP * (1.0-p_che2dom) * (1.0-p_che2poc)
    mort_che_din = mort_cheN * (1.0-p_che2dom) * (1.0-p_che2poc)
    mort_che_poc = mort_cheC * p_che2poc
    mort_che_dop = mort_cheP - (mort_che_dip + mort_che_poc / p_pom_CP)
    mort_che_don = mort_cheN - (mort_che_din + mort_che_poc / p_pom_CN)
    
    NARprime = np.fmax(0.0, nar - p_het_bmin)
    NAIprime = np.fmax(0.0, nai - p_het_bmin)
    NIRprime = np.fmax(0.0, nir - p_het_bmin)
    NOSprime = np.fmax(0.0, nos - p_het_bmin)
    mort_nar = p_nar_lmort * NARprime + p_nar_qmort * (NARprime * nar)
    mort_nai = p_nai_lmort * NAIprime + p_nai_qmort * (NAIprime * nai)
    mort_nir = p_nir_lmort * NIRprime + p_nir_qmort * (NIRprime * nir)
    mort_nos = p_nos_lmort * NOSprime + p_nos_qmort * (NOSprime * nos)
    mort_hetC = mort_nar + mort_nai + mort_nir + mort_nos
    mort_hetP = mort_nar/p_nar_CP + mort_nai/p_nai_CP + mort_nir/p_nir_CP + mort_nos/p_nos_CP
    mort_hetN = mort_nar/p_nar_CN + mort_nai/p_nai_CN + mort_nir/p_nir_CN + mort_nos/p_nos_CN
    mort_het_doc = mort_hetC * p_het2dom * (1.0-p_het2poc)
    mort_het_dic = mort_hetC * (1.0-p_het2dom) * (1.0-p_het2poc)
    mort_het_dip = mort_hetP * (1.0-p_het2dom) * (1.0-p_het2poc)
    mort_het_din = mort_hetN * (1.0-p_het2dom) * (1.0-p_het2poc)
    mort_het_poc = mort_hetC * p_het2poc
    mort_het_dop = mort_hetP - (mort_het_dip + mort_het_poc / p_pom_CP)
    mort_het_don = mort_hetN - (mort_het_din + mort_het_poc / p_pom_CN)
    
    fNARprime = np.fmax(0.0, fnar - p_het_bmin)
    fNAIprime = np.fmax(0.0, fnai - p_het_bmin)
    fNIRprime = np.fmax(0.0, fnir - p_het_bmin)
    fNOSprime = np.fmax(0.0, fnos - p_het_bmin)
    mort_fnar = p_fnar_lmort * fNARprime + p_nar_qmort * (fNARprime * fnar)
    mort_fnai = p_fnai_lmort * fNAIprime + p_nai_qmort * (fNAIprime * fnai)
    mort_fnir = p_fnir_lmort * fNIRprime + p_nir_qmort * (fNIRprime * fnir)
    mort_fnos = p_fnos_lmort * fNOSprime + p_nos_qmort * (fNOSprime * fnos)
    mort_fhetC = mort_fnar + mort_fnai + mort_fnir + mort_fnos
    mort_fhetP = mort_fnar/p_nar_CP + mort_fnai/p_nai_CP + mort_fnir/p_nir_CP + mort_fnos/p_nos_CP
    mort_fhetN = mort_fnar/p_nar_CN + mort_fnai/p_nai_CN + mort_fnir/p_nir_CN + mort_fnos/p_nos_CN
    mort_fhet_doc = mort_fhetC * p_het2dom * (1.0-p_het2poc)
    mort_fhet_dic = mort_fhetC * (1.0-p_het2dom) * (1.0-p_het2poc)
    mort_fhet_dip = mort_fhetP * (1.0-p_het2dom) * (1.0-p_het2poc)
    mort_fhet_din = mort_fhetN * (1.0-p_het2dom) * (1.0-p_het2poc)
    mort_fhet_poc = mort_fhetC * p_het2poc
    mort_fhet_dop = mort_fhetP - (mort_fhet_dip + mort_fhet_poc / p_pom_CP)
    mort_fhet_don = mort_fhetN - (mort_fhet_din + mort_fhet_poc / p_pom_CN)
    
    #----------------------------------------------------------------------
    # (9)  Small zooplankton grazing and mortality
    #---------------------------------------------------------------------- 
    o2limit = (1.0 - np.exp(-o2/60.0))
    if darwin_arch:
        prey = NARprime + NAIprime + NIRprime + NOSprime + \
               fNARprime + fNAIprime + fNIRprime + fNOSprime 
        zoogrzphy = p_mumax_zoo * zoo * (0.0 / (prey + p_kzoo_phy)) * o2limit
        zoogrzdia = p_mumax_zoo * zoo * (0.0 / (prey + p_kzoo_dia)) * o2limit
        zoogrzaoa = p_mumax_zoo * zoo * (0.0 / (prey + p_kzoo_aoa)) * o2limit
        zoogrznob = p_mumax_zoo * zoo * (0.0 / (prey + p_kzoo_nob)) * o2limit
        zoogrzaox = p_mumax_zoo * zoo * (0.0 / (prey + p_kzoo_aox)) * o2limit
        zoogrznar = p_mumax_zoo * zoo * (NARprime / (prey + p_kzoo_nar)) * o2limit
        zoogrznai = p_mumax_zoo * zoo * (NAIprime / (prey + p_kzoo_nai)) * o2limit
        zoogrznir = p_mumax_zoo * zoo * (NIRprime / (prey + p_kzoo_nir)) * o2limit
        zoogrznos = p_mumax_zoo * zoo * (NOSprime / (prey + p_kzoo_nos)) * o2limit
        zoogrzfnar = p_mumax_zoo * zoo * (fNARprime / (prey + p_kzoo_nar)) * o2limit
        zoogrzfnai = p_mumax_zoo * zoo * (fNAIprime / (prey + p_kzoo_nai)) * o2limit
        zoogrzfnir = p_mumax_zoo * zoo * (fNIRprime / (prey + p_kzoo_nir)) * o2limit
        zoogrzfnos = p_mumax_zoo * zoo * (fNOSprime / (prey + p_kzoo_nos)) * o2limit
    else:
        prey = PHYprime + DIAprime + \
               AOAprime + NOBprime + AOXprime + \
               NARprime + NAIprime + NIRprime + NOSprime + \
               fNARprime + fNAIprime + fNIRprime + fNOSprime 
        zoogrzphy = p_mumax_zoo * zoo * (PHYprime / (prey + p_kzoo_phy)) * o2limit
        zoogrzdia = p_mumax_zoo * zoo * (DIAprime / (prey + p_kzoo_dia)) * o2limit
        zoogrzaoa = p_mumax_zoo * zoo * (AOAprime / (prey + p_kzoo_aoa)) * o2limit
        zoogrznob = p_mumax_zoo * zoo * (NOBprime / (prey + p_kzoo_nob)) * o2limit
        zoogrzaox = p_mumax_zoo * zoo * (AOXprime / (prey + p_kzoo_aox)) * o2limit
        zoogrznar = p_mumax_zoo * zoo * (NARprime / (prey + p_kzoo_nar)) * o2limit
        zoogrznai = p_mumax_zoo * zoo * (NAIprime / (prey + p_kzoo_nai)) * o2limit
        zoogrznir = p_mumax_zoo * zoo * (NIRprime / (prey + p_kzoo_nir)) * o2limit
        zoogrznos = p_mumax_zoo * zoo * (NOSprime / (prey + p_kzoo_nos)) * o2limit
        zoogrzfnar = p_mumax_zoo * zoo * (fNARprime / (prey + p_kzoo_nar)) * o2limit
        zoogrzfnai = p_mumax_zoo * zoo * (fNAIprime / (prey + p_kzoo_nai)) * o2limit
        zoogrzfnir = p_mumax_zoo * zoo * (fNIRprime / (prey + p_kzoo_nir)) * o2limit
        zoogrzfnos = p_mumax_zoo * zoo * (fNOSprime / (prey + p_kzoo_nos)) * o2limit
    
    # collect the total grazing on microbial biomass (C, P and N)
    zoototC = zoogrzphy + zoogrzdia + \
              zoogrzaoa + zoogrznob + zoogrzaox + \
              zoogrznar + zoogrznai + zoogrznir + zoogrznos + \
              zoogrzfnar + zoogrzfnai + zoogrzfnir + zoogrzfnos
    zoototP = zoogrzphy/p_pom_CP + zoogrzdia/p_pom_CP + \
              zoogrzaoa/p_aoa_CP + zoogrznob/p_nob_CP + zoogrzaox/p_aox_CP + \
              zoogrznar/p_nar_CP + zoogrznai/p_nai_CP + zoogrznir/p_nir_CP + zoogrznos/p_nos_CP + \
              zoogrzfnar/p_nar_CP + zoogrzfnai/p_nai_CP + zoogrzfnir/p_nir_CP + zoogrzfnos/p_nos_CP
    zoototN = zoogrzphy/p_pom_CN + zoogrzdia/p_pom_CN + \
              zoogrzaoa/p_aoa_CN + zoogrznob/p_nob_CN + zoogrzaox/p_aox_CN + \
              zoogrznar/p_nar_CN + zoogrznai/p_nai_CN + zoogrznir/p_nir_CN + zoogrznos/p_nos_CN + \
              zoogrzfnar/p_nar_CN + zoogrzfnai/p_nai_CN + zoogrzfnir/p_nir_CN + zoogrzfnos/p_nos_CN 
    
    # partition the ingested microbial biomass to zooplankton biomass
    # ... but for P and N only take up what is needed according to stoichiometry
    zoototC_zoo = zoototC * p_zoogrz2zoo
    
    # excrete the excess P and N that was ingested and is not required via stoichiometric balance
    zoototP_exc = zoototP * p_zoogrz2zoo - zoototC_zoo/p_pom_CP  # phosphorus required by zoo to grow
    zoototN_exc = zoototN * p_zoogrz2zoo - zoototC_zoo/p_pom_CN  # nitrogen required by zoo to grow
    
    # route the remaining grazed microbial biomass to DOC/DIC, DOP/PO4, and DON/NH4
    zoototC_doc = zoototC * (1.0-p_zoogrz2zoo) * p_zoogrz2dom
    zoototP_dop = zoototP * (1.0-p_zoogrz2zoo) * p_zoogrz2dom
    zoototN_don = zoototN * (1.0-p_zoogrz2zoo) * p_zoogrz2dom
    zoototC_dic = zoototC * (1.0-p_zoogrz2zoo) * (1.0-p_zoogrz2dom)
    zoototP_dip = zoototP * (1.0-p_zoogrz2zoo) * (1.0-p_zoogrz2dom)
    zoototN_din = zoototN * (1.0-p_zoogrz2zoo) * (1.0-p_zoogrz2dom)
    
    # Do zooplankton mortality as closure
    ZOOprime = np.fmax(0.0, zoo - p_zoo_bmin)
    mort_zoo = p_zoo_lmort * ZOOprime + p_zoo_qmort * (ZOOprime * zoo)
    mort_zoo_doc = mort_zoo * p_zoo2dom
    mort_zoo_dic = mort_zoo * p_zoo2dic
    mort_zoo_poc = mort_zoo * p_zoo2pom

    #----------------------------------------------------------------------
    # (10)  Meso-zooplankton grazing and mortality
    #---------------------------------------------------------------------- 
    if darwin_arch:
        prey = PHYprime + DIAprime + AOAprime + NOBprime + AOXprime
        mesgrzphy = p_mumax_mes * mes * (PHYprime / (prey + p_kmes_phy)) * o2limit
        mesgrzdia = p_mumax_mes * mes * (DIAprime / (prey + p_kmes_dia)) * o2limit
        mesgrzaoa = p_mumax_mes * mes * (AOAprime / (prey + p_kmes_aoa)) * o2limit
        mesgrznob = p_mumax_mes * mes * (NOBprime / (prey + p_kmes_nob)) * o2limit
        mesgrzaox = p_mumax_mes * mes * (AOXprime / (prey + p_kmes_aox)) * o2limit
        mesgrzzoo = p_mumax_mes * mes * (0.0 / (prey + p_kmes_zoo))
    else:
        prey = PHYprime + DIAprime + AOAprime + NOBprime + AOXprime + ZOOprime
        mesgrzphy = p_mumax_mes * mes * (PHYprime / (prey + p_kmes_phy)) * o2limit
        mesgrzdia = p_mumax_mes * mes * (DIAprime / (prey + p_kmes_dia)) * o2limit
        mesgrzaoa = p_mumax_mes * mes * (AOAprime / (prey + p_kmes_aoa)) * o2limit
        mesgrznob = p_mumax_mes * mes * (NOBprime / (prey + p_kmes_nob)) * o2limit
        mesgrzaox = p_mumax_mes * mes * (AOXprime / (prey + p_kmes_aox)) * o2limit
        mesgrzzoo = p_mumax_mes * mes * (ZOOprime / (prey + p_kmes_zoo)) * o2limit
    
    # collect the total grazing on microbial biomass (C, P and N)
    mestotC = mesgrzphy + mesgrzdia + mesgrzaoa + mesgrznob + mesgrzaox + mesgrzzoo 
    mestotP = mesgrzphy/p_pom_CP + mesgrzdia/p_pom_CP + \
              mesgrzaoa/p_aoa_CP + mesgrznob/p_nob_CP + mesgrzaox/p_aox_CP + mesgrzzoo/p_pom_CP
    mestotN = mesgrzphy/p_pom_CN + mesgrzdia/p_pom_CN + \
              mesgrzaoa/p_aoa_CN + mesgrznob/p_nob_CN + mesgrzaox/p_aox_CN + mesgrzzoo/p_pom_CN
    
    # partition the ingested microbial biomass to mesplankton biomass
    # ... but for P and N only take up what is needed according to stoichiometry
    mestotC_mes = mestotC * p_mesgrz2mes
    
    # excrete the excess P and N that was ingested and is not required via stoichiometric balance
    mestotP_exc = mestotP * p_mesgrz2mes - mestotC_mes/p_pom_CP
    mestotN_exc = mestotN * p_mesgrz2mes - mestotC_mes/p_pom_CN
    
    # route the remaining grazed microbial biomass to DOC/DIC, DOP/PO4, and DON/NH4
    mestotC_doc = mestotC * (1.0-p_mesgrz2mes) * p_mesgrz2dom
    mestotP_dop = mestotP * (1.0-p_mesgrz2mes) * p_mesgrz2dom
    mestotN_don = mestotN * (1.0-p_mesgrz2mes) * p_mesgrz2dom
    mestotC_dic = mestotC * (1.0-p_mesgrz2mes) * (1.0-p_mesgrz2dom)
    mestotP_dip = mestotP * (1.0-p_mesgrz2mes) * (1.0-p_mesgrz2dom)
    mestotN_din = mestotN * (1.0-p_mesgrz2mes) * (1.0-p_mesgrz2dom)
    
    # Do zooplankton mortality as closure
    MESprime = np.fmax(0.0, mes - p_mes_bmin)
    mort_mes = p_mes_lmort * MESprime + p_mes_qmort * (MESprime * mes)
    mort_mes_doc = mort_mes * p_mes2dom
    mort_mes_dic = mort_mes * p_mes2dic
    mort_mes_poc = mort_mes * p_mes2pom


    #----------------------------------------------------------------------
    # (7)  Implicit POC sinking
    #---------------------------------------------------------------------- 
    ### POC production through water column ###
    #poc_prod = -t_poc_flux_top/z_dz * (-z_zgrid / 100.0)**(p_martinb) \
    poc_prod = mort_phy_poc + mort_zoo_poc + mort_mes_poc + mort_che_poc + mort_het_poc + mort_fhet_poc
    poc_remi = poc_prod*0.0
    ### Balance fluxes ###
    fpoc_out = 0.0
    for k in np.arange(0,len(z_zgrid)):
       # 1. set incoming flux from previous outgoing flux (above)
       fpoc_in = fpoc_out
       # 2. Compute the scale length
       lengthscale = 1.0
       # 3. Reduce remin in ODZ by lengthening sinking rate
       poc_diss = p_poc_diss * 1.0
       if o2[k] < 40.0:
           poc_diss = poc_diss * (1.0 + (p_odzfac-1.0) * (40.0 - o2[k])/35.0)
       if o2[k] < 5.0:
           poc_diss = poc_diss * p_odzfac
       # 4. apply lengthscale to sinking rate
       poc_diss = lengthscale * poc_diss
       # 5. calculate decay of POC
       decay = np.exp(-z_dz / poc_diss)
       # 6. calculate outgoing POC
       fpoc_out = fpoc_in * decay + poc_prod[k] * (1.0-decay) * poc_diss
       # 7. calculate remineralised POC by difference 
       poc_remi[k] = (poc_prod[k] + (fpoc_in - fpoc_out) / z_dz)
      
    if conserving:
        poc_remi[:] = poc_remi[:] * 0.0
       
     
    #----------------------------------------------------------------------
    # (8)  Calculate ddt_for each tracer (mmol/m3/s)
    #----------------------------------------------------------------------
    qDOP = np.zeros(np.shape(z_zgrid))
    qDON = np.zeros(np.shape(z_zgrid))
    qrDOP = np.zeros(np.shape(z_zgrid))
    qrDON = np.zeros(np.shape(z_zgrid))
    nar_docrem_ratio = np.zeros(np.shape(z_zgrid))
    nai_docrem_ratio = np.zeros(np.shape(z_zgrid))
    nir_docrem_ratio = np.zeros(np.shape(z_zgrid))
    nos_docrem_ratio = np.zeros(np.shape(z_zgrid))
    for k in np.arange(0,len(z_zgrid)):
        if doc[k] > 0.0:
            qDOP[k] = dop[k] / doc[k]
            qDON[k] = don[k] / doc[k]
            nar_docrem_ratio[k] = zlimnar_doc[k] / (zlimnar_doc[k] + zlimnar_rdoc[k]) 
            nai_docrem_ratio[k] = zlimnai_doc[k] / (zlimnai_doc[k] + zlimnai_rdoc[k]) 
            nir_docrem_ratio[k] = zlimnir_doc[k] / (zlimnir_doc[k] + zlimnir_rdoc[k]) 
            nos_docrem_ratio[k] = zlimnos_doc[k] / (zlimnos_doc[k] + zlimnos_rdoc[k]) 
        if rdoc[k] > 0.0:
            qrDOP[k] = rdop[k] / rdoc[k]
            qrDON[k] = rdon[k] / rdoc[k]
        
    RemOx    = mu_nar * nar * p_ynar_aer * (1.0-mu_nar_fac) \
               + mu_nai * nai * p_ynai_aer * (1.0-mu_nai_fac) \
               + mu_nir * nir * p_ynir_aer * (1.0-mu_nir_fac) \
               + mu_nos * nos * p_ynos_aer * (1.0-mu_nos_fac) \
               + mu_fnar * fnar * p_ynar_aer * (1.0-mu_fnar_fac) \
               + mu_fnai * fnai * p_ynai_aer * (1.0-mu_fnai_fac) \
               + mu_fnir * fnir * p_ynir_aer * (1.0-mu_fnir_fac) \
               + mu_fnos * fnos * p_ynos_aer * (1.0-mu_fnos_fac)
    RemDen1  = mu_nar * nar * p_ynar_ana * mu_nar_fac + \
               mu_fnar * fnar * p_ynar_ana * mu_fnar_fac 
    RemDen2  = mu_nai * nai * p_ynai_ana * mu_nai_fac + \
               mu_fnai * fnai * p_ynai_ana * mu_fnai_fac
    RemDen3  = mu_nir * nir * p_ynir_ana * mu_nir_fac + \
               mu_fnir * fnir * p_ynir_ana * mu_fnir_fac
    RemDen4  = mu_nos * nos * p_ynos_ana * mu_nos_fac + \
               mu_fnos * fnos * p_ynos_ana * mu_fnos_fac
    RemDOC   = (mu_nar * nar * p_ynar_aer * (1.0-mu_nar_fac) \
               + mu_nar * nar * p_ynar_ana * mu_nar_fac) * nar_docrem_ratio \
               + ( mu_nai * nai * p_ynai_aer * (1.0-mu_nai_fac) \
               + mu_nai * nai * p_ynai_ana * mu_nai_fac ) * nai_docrem_ratio \
               + ( mu_nir * nir * p_ynir_aer * (1.0-mu_nir_fac) \
               + mu_nir * nir * p_ynir_ana * mu_nir_fac ) * nir_docrem_ratio \
               + ( mu_nos * nos * p_ynos_aer * (1.0-mu_nos_fac) \
               + mu_nos * nos * p_ynos_ana * mu_nos_fac ) * nos_docrem_ratio  \
               + ( mu_fnar * fnar * p_ynar_aer * (1.0-mu_fnar_fac) \
               + mu_fnar * fnar * p_ynar_ana * mu_fnar_fac ) * nar_docrem_ratio \
               + ( mu_fnai * fnai * p_ynai_aer * (1.0-mu_fnai_fac) \
               + mu_fnai * fnai * p_ynai_ana * mu_fnai_fac ) * nai_docrem_ratio \
               + ( mu_fnir * fnir * p_ynir_aer * (1.0-mu_fnir_fac) \
               + mu_fnir * fnir * p_ynir_ana * mu_fnir_fac ) * nir_docrem_ratio \
               + ( mu_fnos * fnos * p_ynos_aer * (1.0-mu_fnos_fac) \
               + mu_fnos * fnos * p_ynos_ana * mu_fnos_fac ) * nos_docrem_ratio
    RemrDOC  = (mu_nar * nar * p_ynar_aer * (1.0-mu_nar_fac) \
               + mu_nar * nar * p_ynar_ana * mu_nar_fac) * (1.0 - nar_docrem_ratio ) \
               + ( mu_nai * nai * p_ynai_aer * (1.0-mu_nai_fac) \
               + mu_nai * nai * p_ynai_ana * mu_nai_fac ) * (1.0 - nai_docrem_ratio ) \
               + ( mu_nir * nir * p_ynir_aer * (1.0-mu_nir_fac) \
               + mu_nir * nir * p_ynir_ana * mu_nir_fac ) * (1.0 - nir_docrem_ratio ) \
               + ( mu_nos * nos * p_ynos_aer * (1.0-mu_nos_fac) \
               + mu_nos * nos * p_ynos_ana * mu_nos_fac ) * (1.0 - nos_docrem_ratio ) \
               + ( mu_fnar * fnar * p_ynar_aer * (1.0-mu_fnar_fac) \
               + mu_fnar * fnar * p_ynar_ana * mu_fnar_fac ) * (1.0 - nar_docrem_ratio ) \
               + ( mu_fnai * fnai * p_ynai_aer * (1.0-mu_fnai_fac) \
               + mu_fnai * fnai * p_ynai_ana * mu_fnai_fac ) * (1.0 - nai_docrem_ratio ) \
               + ( mu_fnir * fnir * p_ynir_aer * (1.0-mu_fnir_fac) \
               + mu_fnir * fnir * p_ynir_ana * mu_fnir_fac ) * (1.0 - nir_docrem_ratio ) \
               + ( mu_fnos * fnos * p_ynos_aer * (1.0-mu_fnos_fac) \
               + mu_fnos * fnos * p_ynos_ana * mu_fnos_fac ) * (1.0 - nos_docrem_ratio )
    DIPrelease = np.fmax(0.0, ( RemDOC * qDOP + RemrDOC * qrDOP ) - \
                 ( mu_nar * nar / p_nar_CP \
                 + mu_nai * nai / p_nai_CP \
                 + mu_nir * nir / p_nir_CP \
                 + mu_nos * nos / p_nos_CP \
                 + mu_fnar * fnar / p_nar_CP \
                 + mu_fnai * fnai / p_nai_CP \
                 + mu_fnir * fnir / p_nir_CP \
                 + mu_fnos * fnos / p_nos_CP ) )
    DINrelease = np.fmax(0.0, ( RemDOC * qDON + RemrDOC * qrDON ) - \
                 ( mu_nar * nar / p_nar_CN \
                 + mu_nai * nai / p_nai_CN \
                 + mu_nir * nir / p_nir_CN \
                 + mu_nos * nos / p_nos_CN \
                 + mu_fnar * fnar / p_nar_CN \
                 + mu_fnai * fnai / p_nai_CN \
                 + mu_fnir * fnir / p_nir_CN \
                 + mu_fnos * fnos / p_nos_CN ) )
    DIPuptake  = np.fmin(0.0, ( RemDOC * qDOP + RemrDOC * qrDOP ) - \
                 ( mu_nar * nar / p_nar_CP \
                 + mu_nai * nai / p_nai_CP \
                 + mu_nir * nir / p_nir_CP \
                 + mu_nos * nos / p_nos_CP \
                 + mu_fnar * fnar / p_nar_CP \
                 + mu_fnai * fnai / p_nai_CP \
                 + mu_fnir * fnir / p_nir_CP \
                 + mu_fnos * fnos / p_nos_CP ) )
    DINuptake  = np.fmin(0.0, ( RemDOC * qDON + RemrDOC * qrDON ) - \
                 ( mu_nar * nar / p_nar_CN \
                 + mu_nai * nai / p_nai_CN \
                 + mu_nir * nir / p_nir_CN \
                 + mu_nos * nos / p_nos_CN \
                 + mu_fnar * fnar / p_nar_CN \
                 + mu_fnai * fnai / p_nai_CN \
                 + mu_fnir * fnir / p_nir_CN \
                 + mu_fnos * fnos / p_nos_CN ) )
    
    ddt_o2   = mu_phy * phy * p_pom_O2C \
               + mu_dia * dia * p_pom_O2C \
               - mu_nar * nar * p_ynar_oxy * (1.0-mu_nar_fac) \
               - mu_nai * nai * p_ynai_oxy * (1.0-mu_nai_fac) \
               - mu_nir * nir * p_ynir_oxy * (1.0-mu_nir_fac) \
               - mu_nos * nos * p_ynos_oxy * (1.0-mu_nos_fac) \
               - mu_fnar * fnar * p_ynar_oxy * (1.0-mu_fnar_fac) \
               - mu_fnai * fnai * p_ynai_oxy * (1.0-mu_fnai_fac) \
               - mu_fnir * fnir * p_ynir_oxy * (1.0-mu_fnir_fac) \
               - mu_fnos * fnos * p_ynos_oxy * (1.0-mu_fnos_fac) \
               - mu_aoa * aoa * p_yaoa_oxy \
               - mu_nob * nob * p_ynob_oxy \
               - ( zoototC + mestotC ) * p_pom_O2C
               
    ddt_po4  = mort_phy_dip + mort_dia_dip \
               + mort_zoo_dic / p_pom_CP + mort_mes_dic / p_pom_CP \
               + mort_che_dip + mort_het_dip + mort_fhet_dip \
               + zoototP_exc + zoototP_dip \
               + mestotP_exc + mestotP_dip \
               + DIPrelease - DIPuptake  \
               - mu_aoa * aoa * 1.0/p_aoa_CP \
               - mu_nob * nob * 1.0/p_nob_CP \
               - mu_aox * aox * 1.0/p_aox_CP \
               - phy_po4upt - dia_po4upt
    
    ddt_dic = RemDOC + RemrDOC \
              + mort_phy_dic + mort_dia_dic \
              + mort_che_dic + mort_het_dic + mort_fhet_dic \
              + mort_zoo_dic + mort_mes_dic \
              + zoototC_dic + mestotC_dic \
              - mu_nar*nar - mu_nai*nai - mu_nir*nir - mu_nos*nos \
              - mu_fnar*fnar - mu_fnai*fnai - mu_fnir*fnir - mu_fnos*fnos \
              - mu_aoa*aoa - mu_nob*nob - mu_aox*aox \
              - mu_phy * phy - mu_dia * dia
                
    # DOM species
    ddt_doc  =  f_lability * ( poc_remi \
                + mort_phy_doc + mort_dia_doc \
                + mort_che_doc + mort_het_doc + mort_fhet_doc \
                + zoototC_doc + mort_zoo_doc \
                + mestotC_doc + mort_mes_doc ) \
                + rdoc * r_rdom2dom \
                - RemDOC 

    ddt_dop  =  f_lability * ( poc_remi / p_pom_CP \
                + mort_phy_dop + mort_dia_dop \
                + mort_che_dop + mort_het_dop + mort_fhet_dop \
                + zoototP_dop + mort_zoo_doc / p_pom_CP \
                + mestotP_dop + mort_mes_doc / p_pom_CP ) \
                + rdop * r_rdom2dom \
                - (RemDOC * qDOP)
    
    ddt_don  =  f_lability * ( poc_remi / p_pom_CN \
                + mort_phy_don + mort_dia_don \
                + mort_che_don + mort_het_don + mort_fhet_don \
                + zoototN_don + mort_zoo_doc / p_pom_CN \
                + mestotN_don + mort_mes_doc / p_pom_CN ) \
                + rdon * r_rdom2dom \
                - (RemDOC * qDON) 
                
    # rDOM species
    ddt_rdoc  = (1.0-f_lability) * ( poc_remi \
                + mort_phy_doc + mort_dia_doc \
                + mort_che_doc + mort_het_doc + mort_fhet_doc \
                + zoototC_doc + mort_zoo_doc \
                + mestotC_doc + mort_mes_doc ) \
                - rdoc * r_rdom2dom - RemrDOC
    
    ddt_rdop  =  (1.0-f_lability) * ( poc_remi / p_pom_CP \
                + mort_phy_dop + mort_dia_dop \
                + mort_che_dop + mort_het_dop + mort_fhet_dop \
                + zoototP_dop + mort_zoo_doc / p_pom_CP \
                + mestotP_dop + mort_mes_doc / p_pom_CP ) \
                - rdop * r_rdom2dom - (RemrDOC * qrDOP)
                
    ddt_rdon  =  (1.0-f_lability) * ( poc_remi / p_pom_CN \
                + mort_phy_don + mort_dia_don \
                + mort_che_don + mort_het_don + mort_fhet_don \
                + zoototN_don + mort_zoo_doc / p_pom_CN \
                + mestotN_don + mort_mes_doc / p_pom_CN ) \
                - rdon * r_rdom2dom - (RemrDOC * qrDON)
                
    
    # Nitrogen species
    ddt_nh4  =  mort_phy_din + mort_dia_din \
                + mort_che_din + mort_het_din + mort_fhet_din \
                + mort_zoo_dic / p_pom_CN + mort_mes_dic / p_pom_CN \
                + zoototN_exc + zoototN_din \
                + mestotN_exc + mestotN_din \
                + DINrelease - DINuptake \
                - Ammox - Anammox \
                - mu_nob * nob * 1.0/p_nob_CN \
                - phy_nh4upt - dia_nh4upt
    ddt_no2  =  Denitrif1 - Denitrif3 - Nitrox \
                + mu_aoa * aoa * (p_yaoa_nh4 - 1.0/p_aoa_CN) \
                - mu_aox * aox * p_yaox_no2 \
                - phy_no2upt - dia_no2upt
    ddt_no3  =  Nitrox - Denitrif1 - Denitrif2 \
                + mu_aox * aox * p_paox_no3 \
                - phy_no3upt - dia_no3upt
    # N2O (mmol N2O/m3/s, units of N2O, not N)
    ddt_n2o  =  ( Denitrif2 + Denitrif3 ) * 0.5 - Denitrif4 
    # N2 (mmol N2/m3/s, units of N2, not N)
    ddt_n2   =  Denitrif4 + Anammox
   
    # phytoplankton
    ddt_phy = mu_phy * phy - mort_phy - zoogrzphy - mesgrzphy
    ddt_dia = mu_dia * dia - mort_dia - zoogrzdia - mesgrzdia

    # explicit microbes
    ddt_aoa = mu_aoa * aoa - mort_aoa - zoogrzaoa - mesgrzaoa
    ddt_nob = mu_nob * nob - mort_nob - zoogrznob - mesgrznob
    ddt_aox = mu_aox * aox - mort_aox - zoogrzaox - mesgrzaox
    ddt_nar = mu_nar * nar - mort_nar - zoogrznar
    ddt_nai = mu_nai * nai - mort_nai - zoogrznai
    ddt_nir = mu_nir * nir - mort_nir - zoogrznir
    ddt_nos = mu_nos * nos - mort_nos - zoogrznos
    ddt_fnar = mu_fnar * fnar - mort_fnar - zoogrzfnar
    ddt_fnai = mu_fnai * fnai - mort_fnai - zoogrzfnai
    ddt_fnir = mu_fnir * fnir - mort_fnir - zoogrzfnir
    ddt_fnos = mu_fnos * fnos - mort_fnos - zoogrzfnos

    # zooplankton
    ddt_zoo = zoototC_zoo - mort_zoo - mesgrzzoo
    ddt_mes = mestotC_mes - mort_mes

    
    return [ddt_o2, ddt_n2, ddt_no3, ddt_no2, ddt_nh4, ddt_n2o, ddt_po4, ddt_dic, \
            ddt_doc, ddt_dop, ddt_don, ddt_rdoc, ddt_rdop, ddt_rdon, \
            ddt_phy, ddt_dia, \
            ddt_aoa, ddt_nob, ddt_aox, \
            ddt_nar, ddt_nai, ddt_nir, ddt_nos, \
            ddt_fnar, ddt_fnai, ddt_fnir, ddt_fnos, \
            ddt_zoo, ddt_mes, \
            phy_nh4upt, phy_no2upt, phy_no3upt, phy_po4upt, \
            dia_nh4upt, dia_no2upt, dia_no3upt, dia_po4upt, \
            poc_prod, poc_remi, \
            Ammox, Nitrox, Anammox, Denitrif1, Denitrif2, Denitrif3, Denitrif4]
    
