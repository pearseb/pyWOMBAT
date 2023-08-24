#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 10:06:24 2023



@author: pbuchanan
"""

d2s = 86400.0

# Stoichiometry of phytoplankon / zooplankton
p_pom_CP = 106.0
p_pom_CN = 106.0/16.0 
p_pom_O2C = 122.0/106.0

# Sinking terms
p_martinb = -1.00
p_poc_diss = 88
p_odzfac = 3.0

####### Ammonium oxidation ####
###############################
# Ammox: NH4 --> NO2
p_muaoa = 0.5 / d2s 
p_kaoa_nh4 = 0.1    # Martens-Habbena et al. (2009) suggests 134 nM; Liu et al. (2023) shows oligotrophic AOA with K as low as 8-44 nM
p_paoa_o2 = 293.0 / d2s
p_yaoa_nh4 = 11.0
p_yaoa_oxy = 15.5
p_aoa_CN = 4.0
p_aoa_CP = 55.0
p_aoa_lmort = 0.05 / d2s
p_aoa_qmort = 0.5 / d2s

####### Nitrite oxidationn ####
###############################
# Nitrox: NO2 --> NO3
p_munob = 1.0 / d2s; 
p_knob_no2 = 0.1 ;  # Zhang et al. (2020); Xin Sun et al. (2021); Liu et al. (2023) 65-125 nM
p_pnob_o2 = 71.0 / d2s ;
p_ynob_no2 = 27.8 ;
p_ynob_oxy = 12.9 ;
p_nob_CN = 4.0 ;
p_nob_CP = 55.0 ;
p_nob_lmort = 0.1 / d2s ;
p_nob_qmort = 0.5 / d2s ;

########## Anammox ############
###############################
p_muaox = 0.25 / d2s; 
p_kaox_nh4 = 0.45 ;
p_kaox_no2 = 0.45 ;
p_yaox_nh4 = 14.0 ;
p_yaox_no2 = 16.2 ;
p_aox_CN = 5.0 ;
p_aox_CP = 55.0 ;
p_aox_lmort = 0.025 / d2s ;
p_aox_qmort = 0.5 / d2s ;

p_che_bmin = 0.0001;
p_che2doc = 1.0;
p_het_bmin = 0.0001;
p_het2doc = 1.0;


###### Facultative NO3 reducers #####
#####################################
# NAR: NO3 --> NO2
p_munar = 0.5 / d2s; 
p_knar_doc = 0.5 ;
p_knar_no3 = 4.0 ;
p_pnar_o2 = 450.0 / d2s ;
p_ynar_aer = 5.9 ;
p_ynar_oxy = 5.4 ;
p_ynar_ana = 6.5 ;
p_ynar_no3 = 10.8 ;
p_nar_CN = 4.5 ;
p_nar_CP = 35.0 ;
p_nar_lmort = 0.05 / d2s ;
p_nar_qmort = 0.5 / d2s ;

###### Facultative NO3 reducers #####
#####################################
# NAI: NO3 --> N2O
p_munai = 0.5 / d2s; 
p_knai_doc = 0.5 ;
p_knai_no3 = 4.0 ;
p_pnai_o2 = 450.0 / d2s ;
p_ynai_aer = 5.9 ;
p_ynai_oxy = 5.4 ;
p_ynai_ana = 6.5 ;
p_ynai_no3 = 6.0 ;
p_nai_CN = 4.5 ;
p_nai_CP = 35.0 ;
p_nai_lmort = 0.05 / d2s ;
p_nai_qmort = 0.5 / d2s ;

###### Facultative NO2 reducers #####
#####################################
# NIR: NO2 --> N2O
p_munir = 0.5 / d2s; 
p_knir_doc = 0.5 ;
p_knir_no2 = 4.0 ;
p_pnir_o2 = 450.0 / d2s ;
p_ynir_aer = 5.9 ;
p_ynir_oxy = 5.4 ;
p_ynir_ana = 6.5 ;
p_ynir_no2 = 10.8 ;
p_nir_CN = 4.5 ;
p_nir_CP = 35.0 ;
p_nir_lmort = 0.05 / d2s ;
p_nir_qmort = 0.5 / d2s ;

###### Facultative N2O reducers #####
#####################################
# NOS: N2O --> N2
p_munos = 0.5 / d2s; 
p_knos_doc = 0.5 ;
p_knos_n2o = 0.3 ;
p_pnos_o2 = 450.0 / d2s ;
p_ynos_aer = 5.9 ;
p_ynos_oxy = 5.4 ;
p_ynos_ana = 6.5 ;
p_ynos_n2o = 21.6 ;
p_nos_CN = 4.5 ;
p_nos_CP = 35.0 ;
p_nos_lmort = 0.05 / d2s ;
p_nos_qmort = 0.5 / d2s ;


##### Small zooplankton #####
p_mumax_zoo = 0.0 / d2s;
p_grz2zoo = 0.3;
p_grz2doc = 0.8;
p_kaoa_grz = 1.2;
p_knob_grz = 1.2;
p_kaox_grz = 1.2;
p_knar_grz = 1.2;
p_knir_grz = 1.2;
p_knos_grz = 1.2;
p_zoo_bmin = 0.00;
p_zoo_lmort = 0.1 / d2s;
p_zoo_qmort = 0.5 / d2s;
p_zoo2doc = 1.0;
