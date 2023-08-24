#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 11:29:33 2023

Purpose
    - Run the 1D model

@author: pbuchanan
"""

#%% imports

import sys
import os
import numpy as np
import pandas as pd

# plotting packages
import seaborn as sb
sb.set(style='ticks')
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.gridspec import GridSpec
import cmocean.cm as cmo
from cmocean.tools import lighten

# numerical packages
from numba import jit


# print versions of packages
print("python version =",sys.version[:5])
print("numpy version =", np.__version__)
print("pandas version =", pd.__version__)
print("seaborn version =", sb.__version__)
print("matplotlib version =", sys.modules[plt.__package__].__version__)
print("cmocean version =", sys.modules[cmo.__package__].__version__)

wrkdir = "/Users/buc146/Dropbox/CSIRO/pyWOMBAT/WOMBAT_0D"
os.chdir(wrkdir)


#%% get parameters 

# set timestep
days = 200                # length of run (days)
dt = 86400.0/6              # timestep (seconds)
timesteps = days*86400/dt   # total number of timesteps
save_freq = 1             # frequency of saving output (days) 
out_freq = save_freq*86400/dt
conserving = True          # check for conservation of mass?
restart = False             # initialise from restart file?

# set the dilution rate of the chemostat (/s)
#   ... at equilibrium, this is what the phy growth rate will be
dil = 0.05 / 86400.0    


#%% initialise tracers

o2  = np.zeros((2))
no3 = np.zeros((2))
dfe = np.zeros((2))
phy = np.zeros((2))
zoo = np.zeros((2))
det = np.zeros((2))
cal = np.zeros((2))
alk = np.zeros((2))
dic = np.zeros((2))

if restart:
    df = pd.read_csv("../output/restart_0D.csv")
    o2_init  = df["O2"] 
    no3_init = df["NO3"]
    dfe_init = df["dFe"]
    phy_init = df["PHY"]
    zoo_init = df["ZOO"]
    det_init = df["DET"]
    cal_init = df["CAL"]
    alk_init = df["ALK"]
    dic_init = df["DIC"]
else:
    o2_init  = 200
    no3_init = 0.1
    dfe_init = 0.1e-3
    phy_init = 0.1
    zoo_init = 0.1
    det_init = 0.1
    cal_init = 0.1
    alk_init = 2400
    dic_init = 2200

o2[:]  = o2_init
no3[:] = no3_init
dfe[:] = dfe_init
phy[:] = phy_init
zoo[:] = zoo_init
det[:] = det_init
cal[:] = cal_init
alk[:] = alk_init
dic[:] = dic_init

### initialise timeseries (ts) arrays for saving to
o2_ts = np.zeros((int(days/save_freq)+1))*np.nan
no3_ts = np.zeros((int(days/save_freq)+1))*np.nan
dfe_ts = np.zeros((int(days/save_freq)+1))*np.nan
phy_ts = np.zeros((int(days/save_freq)+1))*np.nan
zoo_ts = np.zeros((int(days/save_freq)+1))*np.nan
det_ts = np.zeros((int(days/save_freq)+1))*np.nan
cal_ts = np.zeros((int(days/save_freq)+1))*np.nan
alk_ts = np.zeros((int(days/save_freq)+1))*np.nan
dic_ts = np.zeros((int(days/save_freq)+1))*np.nan
o2_ts[0]  = o2_init
no3_ts[0] = no3_init
dfe_ts[0] = dfe_init
phy_ts[0] = phy_init
zoo_ts[0] = zoo_init
det_ts[0] = det_init
cal_ts[0] = cal_init
alk_ts[0] = alk_init
dic_ts[0] = dic_init


#%% run model

from tqdm import tqdm
from bgc_sms_0D import bgc_sms
from mass_balance_0D import massb_n, massb_c, massb_f

## get some parameters required to calculate mass balances
#from bgcparams import p_aoa_CN, p_nob_CN, p_aox_CN
#from bgcparams import p_nar_CN, p_nai_CN, p_nir_CN, p_nos_CN
#from bgcparams import p_aoa_CP, p_nob_CP, p_aox_CP
#from bgcparams import p_nar_CP, p_nai_CP, p_nir_CP, p_nos_CP
#from bgcparams import p_pom_CN, p_pom_CP


# ensure that if we want to check mass balance, close the system
if conserving:
    dil = 0.0

# integrate forward
ii = 0
for t in tqdm(np.arange(timesteps), desc="Running model", unit=' timesteps'): 
    
    # set constant sources to chemostat
    i_o2 = 200.0
    i_no3 = 0.1
    i_dfe = 0.1e-3
    i_phy = 0.0
    i_zoo = 0.0
    i_det = 0.0
    i_cal = 0.0
    i_alk = 2400.0
    i_dic = 2200.0
    
    # get sources minus sinks
    sms = bgc_sms(dil, \
                  o2, no3, dfe, phy, zoo, det, cal, alk, dic, \
                  i_o2, i_no3, i_dfe, i_phy, i_zoo, i_det, i_cal, i_alk, i_dic, \
                  conserving)
    
    # Apply sources and sinks to tracers
    o2[1]  = o2[1]  + sms[0] * dt
    no3[1] = no3[1] + sms[1] * dt
    dfe[1] = dfe[1] + sms[2] * dt
    phy[1] = phy[1] + sms[3] * dt
    zoo[1] = zoo[1] + sms[4] * dt
    det[1] = det[1] + sms[5] * dt
    cal[1] = cal[1] + sms[6] * dt
    alk[1] = alk[1] + sms[7] * dt
    dic[1] = dic[1] + sms[8] * dt
    
    if (t % out_freq) == 0:
        ii += 1
        print("timestep #%i"%(t))
        print("Surface O2  = %.4f µM"%(o2[1]))
        print("Surface NO3 = %.4f µM"%(no3[1]))
        print("Surface dFe = %.4f nM"%(dfe[1]*1e3))
        print("Surface PHY = %.4f µM"%(phy[1]))
        print("Surface ZOO = %.4f µM"%(zoo[1]))
        print("Surface DET = %.4f µM"%(det[1]))
        print("Surface CAL = %.4f µM"%(cal[1]))
        print("Surface ALK = %.4f µM"%(alk[1]))
        print("Surface DIC = %.4f µM"%(dic[1]))
        o2_ts[ii]  = o2[1]
        no3_ts[ii] = no3[1]
        dfe_ts[ii] = dfe[1]
        phy_ts[ii] = phy[1]
        zoo_ts[ii] = zoo[1]
        det_ts[ii] = det[1]
        cal_ts[ii] = cal[1]
        alk_ts[ii] = alk[1]
        dic_ts[ii] = dic[1]
        
        
    
    thresh = 1e-12
    n0,n1,nerr = massb_n(no3, phy, zoo, det, \
                         thresh)
    c0,c1,cerr = massb_c(dic, cal, phy, zoo, det, 106/16.0, \
                         thresh)
    f0,f1,ferr = massb_f(dfe, phy, zoo, det, 106/16.0 * 7.1e-5, \
                         thresh)
        
    if conserving:
        if cerr == 1:
            print("Carbon not conserved at timestep %i"%(t))
            break    
        
        # Fe budget currently not closed because we don't track scavenging losses
        #if ferr == 1:
        #    print("Iron not conserved at timestep %i"%(t))
        #    break
        
        if nerr == 1:
            print("Nitrogen not conserved at timestep %i"%(t))
            break
    else:
        if cerr == 0 and ferr == 0 and nerr == 0:
            print("Arrived at steady state")
            break
    
    
    # old becomes new
    o2[0]  = o2[1] 
    no3[0] = no3[1]
    dfe[0] = dfe[1]
    phy[0] = phy[1]
    zoo[0] = zoo[1]
    det[0] = det[1]
    cal[0] = cal[1]
    alk[0] = alk[1]
    dic[0] = dic[1]
    
    
# get sources minus sinks one last time
sms = bgc_sms(dil, \
              o2, no3, dfe, phy, zoo, det, cal, alk, dic, \
              i_o2, i_no3, i_dfe, i_phy, i_zoo, i_det, i_cal, i_alk, i_dic, \
              conserving)
    
#phy_nh4upt = sms[-17] * 86400

# save the output to file
data = {"O2":o2[1],
        "NO3":no3[1],
        "dFe":dfe[1],
        "PHY":phy[1],
        "ZOO":zoo[1],
        "DET":det[1],
        "CAL":cal[1],
        "ALK":alk[1],
        "DIC":dic[1]}

df = pd.DataFrame.from_dict(data, orient='index', columns=['Value'])
df.to_csv('../output/restart_0D.csv', header=True)



#%% check mass balance between initial state and final state
#   (only works when poc_remi == 0.0, p_wup == 0.0, and p_Kv == 0.0)

tot_n0 = no3_init + phy_init + zoo_init + det_init
tot_n1 = no3[1] + phy[1] + zoo[1] + det[1]
         
print("total N before =",np.sum(tot_n0))
print("total N after =",np.sum(tot_n1))


#%% plot some output 

col1 = 'k'
col2 = 'firebrick'
col3 = 'goldenrod'

lab1 = 'Oxygen (µM)'
lab2 = 'Nitrate (µM)'
lab3 = 'dissolved Fe (nM)'
lab4 = 'Phytoplankton (µM N)'
lab5 = 'Zooplankton (µM N)'
lab6 = 'Detritus (µM N)'
lab7 = 'CaCO$_3$ (µM)'
lab8 = 'Alkalinity (µM Eq)'
lab9 = 'DIC (µM)'

plt.figure(figsize=(14,8))
gs = GridSpec(2,3)

ax1 = plt.subplot(gs[0,0]) # O2
ax2 = plt.subplot(gs[0,1]) # NO3 + dFe
ax3 = plt.subplot(gs[0,2]) # Phy + Zoo + Det
ax4 = plt.subplot(gs[1,0]) # CaCO3
ax5 = plt.subplot(gs[1,1]) # Alk + DIC
ax6 = plt.subplot(gs[1,2]) # Alk + DIC

ax1.plot(o2_ts, color=col1, label=lab1)
ax1.legend()

ax2.plot(no3_ts, color=col1, label=lab2)
ax2.plot(dfe_ts*1e3, color=col2, label=lab3)
ax2.legend()

ax3.plot(phy_ts, color=col1, label=lab4)
ax3.plot(zoo_ts, color=col2, label=lab5)
ax3.plot(det_ts, color=col3, label=lab6)
ax3.legend()

ax4.plot(cal_ts, color=col1, label=lab7)
ax4.legend()

ax5.plot(alk_ts, color=col1, label=lab8)
ax5.legend()

ax6.plot(dic_ts, color=col2, label=lab9)
ax6.legend()


