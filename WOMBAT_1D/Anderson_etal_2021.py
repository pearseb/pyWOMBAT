#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 16:15:32 2024

@author: buc146
"""

#%% imports

import numpy as np


#%% parameters

d2s = 86400.0

phyC = 1.0                   # phytoplankton carbon
zooC = 1.0                   # zooplankton carbon
phy_mu = 1/d2s
p_phy_lmort = 0.025 /d2s       # linear mortality of phytoplankton (basal respiration) (/s)
p_phy_qmort = 0.1 / d2s    # quadratic mortality of phytoplankton (1 / (µM N * s))
p_phy_CN = 122.0/16.0       # mol/mol

    
##########################"
####### Zooplannkton  ####
##########################
p_zoo_mumax = 3.0 / d2s            # scaler for rate of zooplankton grazing
p_zoo_capcoef = 1.0 / d2s   # prey capture efficiency coefficient (m6 / (µM N)2 * s))
p_zoo_qmort = 0.5 / d2s    # quadratic mortality of zooplankton (1 / (µM N * day))
p_zoo_excre = 0.05         # rate of excretion by zooplankton (/day)
p_zoo_assim = 0.7         # zooplankton assimilation efficiency

# Anderson et al. 2021 
p_zoo_betaV = 0.62      # absorption efficiency of protein
p_zoo_betaH = 0.62      # absorption efficiency of carbohydrate
p_protein_CN = 3.7
p_zoo_CN = 5.9
p_zoo_bioturn = 0.06 / d2s  # losses of C and N due to biomass turnover
p_zoo_basalmet = 0.038 / d2s    # losses of C due to some metabolism
p_zoo_sda = 0.072   # specific dynamic action (fixed fraction of C uptake)


#%% model

for ts in np.arange(0,timesteps):

    zoo_capt = p_zoo_capcoef * phyC * phyC
    zoo_mu = p_zoo_mumax * zoo_capt / (p_zoo_mumax + zoo_capt)


    # find what proportion of phytoplankton C is protein and what is carbohydrate
    phy_proteinN = phyC / p_phy_CN               # assume all N is protein
    phy_proteinC = phy_proteinN * p_protein_CN
    phy_carbonhC = phyC - phy_proteinC
    
    # find what proportion of zooplankton is N
    zooN = zooC / p_zoo_CN
    
    ### fixed losses of protein and carbohydrate to fecael matter
    zoo_fecaelC = phy_proteinC * (1-p_zoo_betaV) + phy_carbonhC * (1-p_zoo_betaH)
    zoo_fecaelN = phy_proteinN * (1-p_zoo_betaV)
    
    # losses due to biomass turnover and basal metabolism and specific dynamic action
    zoo_basalmet_lossC = p_zoo_basalmet * zooC 
    zoo_bioturn_lossC = p_zoo_bioturn * zooC 
    zoo_bioturn_lossN = p_zoo_bioturn * zooN
    
    # 


    # sources and sinks
    phyC = phyC * phy_mu - zoo_mu * zooC - phyC*p_phy_lmort - phyC*phyC*p_phy_qmort
    zooC = zooC * zoo_mu - zooC*zooC*qmort - zoo_bioturn_lossC - zoo_basalmet_lossC
