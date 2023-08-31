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

wrkdir = "/Users/buc146/Dropbox/CSIRO/pyWOMBAT/WOMBAT_1D"
os.chdir(wrkdir)


#%% get parameters 

# set timestep
days = 365
dt = 86400.0/12
timesteps = days*86400/dt
wdet = 24.0    # sinking speed of detritus (m/d)
wcal = 6.0     # sinking speed of CaCO3 (m/d)


# plot results during run?
plot = True
plot_freq = 1*86400/dt  # days 

# conserve mass? Only works when sinking is False
conserving = False
do_mld_avg = True
sinking = True
sourcesinks = True
initialise = True



#%% initialise tracers

from phyparams import z_zgrid, z_npt
from tra_init import *

o2  = np.zeros((2, len(z_zgrid)))
no3 = np.zeros((2, len(z_zgrid)))
dfe = np.zeros((2, len(z_zgrid)))
phy = np.zeros((2, len(z_zgrid)))
zoo = np.zeros((2, len(z_zgrid)))
det = np.zeros((2, len(z_zgrid)))
cal = np.zeros((2, len(z_zgrid)))
alk = np.zeros((2, len(z_zgrid)))
dic = np.zeros((2, len(z_zgrid)))

if initialise:
    o2_init  = np.linspace(t_o2_top, t_o2_bot, z_npt)
    no3_init = np.linspace(t_no3_top, t_no3_bot, z_npt)
    dfe_init = np.linspace(t_dfe_top, t_dfe_bot, z_npt)
    phy_init = np.linspace(t_phy_top, t_phy_bot, z_npt)
    zoo_init = np.linspace(t_zoo_top, t_zoo_bot, z_npt)
    det_init = np.linspace(t_det_top, t_det_bot, z_npt)
    cal_init = np.linspace(t_cal_top, t_cal_bot, z_npt)
    alk_init = np.linspace(t_alk_top, t_alk_bot, z_npt)
    dic_init = np.linspace(t_dic_top, t_dic_bot, z_npt)
else:
    df = pd.read_csv("output/restart_1D.csv")
    o2_init  = df["O2"] 
    no3_init = df["NO3"]
    dfe_init = df["dFe"]
    phy_init = df["PHY"]
    zoo_init = df["ZOO"]
    det_init = df["DET"]
    cal_init = df["CAL"]
    alk_init = df["ALK"]
    dic_init = df["DIC"]

o2[0,:]  = o2_init
no3[0,:] = no3_init
dfe[0,:] = dfe_init
dic[0,:] = dic_init
phy[0,:] = phy_init
zoo[0,:] = zoo_init
det[0,:] = det_init
cal[0,:] = cal_init
alk[0,:] = alk_init
dic[0,:] = dic_init

# initialise the sediment detritus and CaCO3 pool
seddet = np.zeros((2))
sedcal = np.zeros((2))



#%% run model

%matplotlib inline

from tqdm import tqdm
from advec_diff import advec_diff, mix_mld
from bgc_sms_1D import bgc_sms
from sink import sink
from plot_1D import plot_1D
from phyparams import z_dz, z_wup, z_Kv, z_mld, z_tmld
from mass_balance_1D import massb_n, massb_c, massb_f

if conserving:
    z_wup = z_wup * 0
    z_Kv = z_Kv * 0


for t in tqdm(np.arange(timesteps), desc="Running model", unit=' timesteps'): 
    
    # do advection and diffusion on each tracer
    o2  = advec_diff(dt, o2,  t_o2_top,  t_o2_bot,  z_dz, z_wup, z_Kv)
    no3 = advec_diff(dt, no3, t_no3_top, t_no3_bot, z_dz, z_wup, z_Kv)
    dfe = advec_diff(dt, dfe, t_dfe_top, t_dfe_bot, z_dz, z_wup, z_Kv)
    phy = advec_diff(dt, phy, t_phy_top, t_phy_bot, z_dz, z_wup, z_Kv)
    zoo = advec_diff(dt, zoo, t_zoo_top, t_zoo_bot, z_dz, z_wup, z_Kv)
    det = advec_diff(dt, det, t_det_top, t_det_bot, z_dz, z_wup, z_Kv)
    cal = advec_diff(dt, cal, t_cal_top, t_cal_bot, z_dz, z_wup, z_Kv)
    alk = advec_diff(dt, alk, t_alk_top, t_alk_bot, z_dz, z_wup, z_Kv)
    dic = advec_diff(dt, dic, t_dic_top, t_dic_bot, z_dz, z_wup, z_Kv)
    
    
    if do_mld_avg and conserving==False:
        o2 = mix_mld(dt, o2, z_zgrid, z_mld, z_tmld)
        no3 = mix_mld(dt, no3, z_zgrid, z_mld, z_tmld)
        dfe = mix_mld(dt, dfe, z_zgrid, z_mld, z_tmld)
        phy = mix_mld(dt, phy, z_zgrid, z_mld, z_tmld)
        zoo = mix_mld(dt, zoo, z_zgrid, z_mld, z_tmld)
        det = mix_mld(dt, det, z_zgrid, z_mld, z_tmld)
        cal = mix_mld(dt, cal, z_zgrid, z_mld, z_tmld)
        alk = mix_mld(dt, alk, z_zgrid, z_mld, z_tmld)
        dic = mix_mld(dt, dic, z_zgrid, z_mld, z_tmld)
    
    
    if sinking:
        out = sink(det[1,:], cal[1,:], wdet, wcal, z_dz, z_zgrid)
        det[1,:] = det[1,:] + out[0] * dt
        cal[1,:] = cal[1,:] + out[1] * dt
        ## remineralise the detritus and CaCO3 in the bottom box
        #p_phy_CN = 106.0/16.0       # mol/mol
        #p_phy_FeC = 7.1e-5          # mol/mol (based on Fe:P of 0.0075:1 (Moore et al 2015))
        #p_phy_O2C = 172.0/106.0     # mol/mol
        #no3[1,-1] = no3[1,-1] + out[2] * dt
        #dfe[1,-1] = dfe[1,-1] + out[2] * dt * p_phy_CN*p_phy_FeC
        #dic[1,-1] = dic[1,-1] + out[2] * dt * p_phy_CN \
        ##                      - out[3] * dt
        #alk[1,-1] = alk[1,-1] - out[2] * dt * p_phy_CN \
        ##                      - out[3] * dt * 2.0
        #o2[1,-1]  = o2[1,-1]  - out[2] * dt * p_phy_CN*p_phy_O2C
        # record the accumulated remineralised material (µM) in the bottom box
        seddet[1] = seddet[1] + out[2]*dt 
        sedcal[1] = sedcal[1] + out[3]*dt 
    
    
    # get sources minus sinks
    if sourcesinks:
        sms = bgc_sms(o2, no3, dfe, phy, zoo, det, cal, alk, dic,\
                      t_tc_top, t_tc_bot, t_par_top, z_dz, z_zgrid)
        o2[1,:]  = o2[1,:]  + sms[0][:] * dt
        no3[1,:] = no3[1,:] + sms[1][:] * dt
        dfe[1,:] = dfe[1,:] + sms[2][:] * dt
        phy[1,:] = phy[1,:] + sms[3][:] * dt
        zoo[1,:] = zoo[1,:] + sms[4][:] * dt
        det[1,:] = det[1,:] + sms[5][:] * dt
        cal[1,:] = cal[1,:] + sms[6][:] * dt
        alk[1,:] = alk[1,:] + sms[7][:] * dt
        dic[1,:] = dic[1,:] + sms[8][:] * dt
    
    
    if (t % 5000) == 0:
        print("timestep #%i"%(t))
        print("Surface PHY = %.4f"%(phy[1,1]))
        print("Surface ZOO = %.4f"%(zoo[1,1]))
        print("Surface DET = %.4f"%(det[1,1]))
        print("Surface NO3 = %.4f"%(no3[1,1]))
        print("Surface dFe = %.4f"%(dfe[1,1]))
        print("Surface CaCO3 = %.4f"%(cal[1,1]))
        print("Surface Alk = %.4f"%(alk[1,1]))
        print("Surface DIC = %.4f"%(dic[1,1]))
    if plot:
        if (t % plot_freq) == 0:
            fig = plot_1D(o2[1,:], no3[1,:], dfe[1,:], phy[1,:], zoo[1,:], det[1,:], cal[1,:], alk[1,:], dic[1,:], z_zgrid)
            fig.savefig("../figures/plot_1D_day_{0:04d}".format(int(t*dt/86400)))
            plt.clf()
            del fig
        
           
    thresh = 1e-8
    n0,n1,nerr = massb_n(no3, phy, zoo, det, \
                         thresh)
    c0,c1,cerr = massb_c(dic, cal, phy, zoo, det, 106/16.0, \
                         thresh)
    #f0,f1,ferr = massb_f(dfe, phy, zoo, det, 106/16.0 * 7.1e-5, \
    #                     thresh)
    
    if conserving:
        if cerr == 1:
            print("Carbon not conserved at timestep %i"%(t))
            break    
    
        #Fe budget currently not closed because we don't track scavenging losses
        #if ferr == 1:
        #    print("Iron not conserved at timestep %i"%(t))
        #    break
    
        if nerr == 1:
            print("Nitrogen not conserved at timestep %i"%(t))
            break
    #else:
    #    if cerr == 0 and ferr == 0 and nerr == 0:
    #        print("Arrived at steady state")
    #        break
    
    
    # old becomes new
    o2[0,:]  = o2[1,:] 
    no3[0,:] = no3[1,:]
    dfe[0,:] = dfe[1,:]
    phy[0,:] = phy[1,:]
    zoo[0,:] = zoo[1,:]
    det[0,:] = det[1,:]
    cal[0,:] = cal[1,:]
    alk[0,:] = alk[1,:]
    dic[0,:] = dic[1,:]
    seddet[0] = seddet[1]
    sedcal[0] = sedcal[1]


# get sources minus sinks one final time
sms = bgc_sms(o2, no3, dfe, phy, zoo, det, cal, alk, dic,\
              t_tc_top, t_tc_bot, t_par_top, z_dz, z_zgrid,\
              conserving)
    
phy_mu = sms[-4] * 86400
zoo_mu = sms[-3] * 86400
phy_lm = sms[-2] * 86400
phy_qm = sms[-1] * 86400

# save the output to file
data = {"depth":z_zgrid,
        "O2":o2[1,:],
        "NO3":no3[1,:],
        "dFe":dfe[1,:],
        "PHY":phy[1,:],
        "ZOO":zoo[1,:],
        "DET":det[1,:],
        "CAL":cal[1,:],
        "ALK":alk[1,:],
        "DIC":dic[1,:]}

df = pd.DataFrame(data)
df.to_csv('../output/restart_1D.csv', header=True)    


# make figure of steady-state results
fig = plot_1D(o2[1,:], no3[1,:], dfe[1,:], phy[1,:], zoo[1,:], det[1,:], cal[1,:], alk[1,:], dic[1,:], z_zgrid)


#%% make animation

import ffmpeg

os.chdir(wrkdir + "/../figures")
(
    ffmpeg
    .input('plot_1D_day_%04d.png', framerate=12)
    .output('WOMBAT_1D_%idays_anim.mp4'%(days), vcodec='libx264', pix_fmt='yuv420p')
    .run()
)

# remove image files
for ii in np.arange(days):
    os.remove('plot_1D_day_{0:04d}.png'.format(ii))


