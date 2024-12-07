#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 11:29:33 2023

Purpose
    - Run WOMBAT-lite in 1D

@author: pbuchanan
"""

#%% imports

import sys
import os
import numpy as np
import pandas as pd
import xarray as xr

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
print("xarray version =", xr.__version__)
print("seaborn version =", sb.__version__)
print("matplotlib version =", sys.modules[plt.__package__].__version__)
print("cmocean version =", sys.modules[cmo.__package__].__version__)

wrkdir = "/Users/buc146/Dropbox/CSIRO/pyWOMBAT/WOMBAT_1D"
os.chdir(wrkdir)


#%% get parameters 

# set timestep
years = 3
days = 365 * years
dt = 86400.0/12
timesteps = days*86400/dt

# set location
latitude = -10
longitude = 220

# set sinking speeds of detritus
p_phy_biothresh = 0.6   # µM threshold for blooms
wdetbio = 15.0          # sinking speed of detritus (m/d)
wcal    = 6.5           # sinking speed of CaCO3 (m/d)

# logicals
conserving = False
do_mld_avg = True
sourcesinks = True
initialise = False
sinking = True

# plot results during run?
plot = True
plot_freq = 1*86400/dt  # days 


#%% extract incident radiation at surface at our location over one year

# find times that align with our timestep
ts_per_year = 365 * 86400 / dt
min_per_ts = dt / 60.0 
start = '1990-01-01T00:00:00'  # Start datetime
end = '1990-12-31T23:59:59'    # End datetime
times = np.arange(np.datetime64(start), np.datetime64(end), np.timedelta64(int(min_per_ts),'m'))

# get the incident irradiance and surface temperature data
data = xr.open_dataset('inputs/rsds_1990.nc')
rsds = data['rsds']
rsds = rsds.sel(lat=latitude, lon=longitude, method='nearest')
data = xr.open_dataset('inputs/tas_1990.nc')
tas = data['tas']
tas = tas.sel(lat=latitude, lon=longitude, method='nearest')
data.close()

# wrap the timeseries data for proper interpolation
rsds1 = rsds.isel(time=-1).assign_coords(time=np.datetime64("1989-12-31T22:30:00"))
rsds2 = rsds.isel(time=0).assign_coords(time=np.datetime64("1991-01-01T01:30:00"))
rsds_ = xr.concat([rsds1, rsds, rsds2], dim='time')
tas1 = tas.isel(time=-1).assign_coords(time=np.datetime64("1989-12-31T22:30:00"))
tas2 = tas.isel(time=0).assign_coords(time=np.datetime64("1991-01-01T01:30:00"))
tas_ = xr.concat([tas1, tas, tas2], dim='time')

# interpolate values based on number of timesteps
rsds__ = rsds_.interp(time=times)
tas__ = tas_.interp(time=times)

plt.figure()
rsds.plot()
plt.xlim(np.datetime64("1989-12-31"),np.datetime64("1990-01-05"))
rsds_.plot(linewidth=0, marker='o', alpha=0.5)
rsds__.plot(linewidth=0, marker='*', alpha=0.5)
plt.figure()
tas.plot()
plt.xlim(np.datetime64("1989-12-31"),np.datetime64("1990-06-05"))
tas_.plot(linewidth=0, marker='o', alpha=0.5)
tas__.plot(linewidth=0, marker='*', alpha=0.5)

rsds = rsds__.values
tas = np.fmax(-1.8, tas__.values-273.15)
tas = np.fmin(30.0, tas)
del rsds_, rsds__, rsds1, rsds2
del tas_, tas__, tas1, tas2


#%% alter the mixed layer depth over the course of a year

start = '1993-01-01T00:00:00'  # Start datetime
end = '1993-12-31T23:59:59'    # End datetime
times = np.arange(np.datetime64(start), np.datetime64(end), np.timedelta64(int(min_per_ts),'m'))

# get the incident irradiance and surface temperature data
data = xr.open_mfdataset('inputs/ocean_mld_1993_*.nc')
mld = data['mld']
mld = mld.sel(yt_ocean=latitude, xt_ocean=longitude, method='nearest')
data.close()

# wrap the timeseries data for proper interpolation
mld1 = mld.isel(Time=-1).assign_coords(Time=np.datetime64("1992-12-31T22:30:00"))
mld2 = mld.isel(Time=0).assign_coords(Time=np.datetime64("1994-01-01T01:30:00"))
mld_ = xr.concat([mld1, mld, mld2], dim='Time')

# interpolate values based on number of timesteps
mld__ = mld_.interp(Time=times)

plt.figure()
mld.plot()
plt.xlim(np.datetime64("1992-12-31"),np.datetime64("1993-01-05"))
mld_.plot(linewidth=0, marker='o', alpha=0.5)
mld__.plot(linewidth=0, marker='*', alpha=0.5)

mld_timeseries = mld__.values
del mld_, mld__, mld1, mld2


plt.figure()
plt.plot(mld_timeseries)


#%% initialise tracers

os.chdir(wrkdir)

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
pchl = np.zeros((2, len(z_zgrid)))
phyfe = np.zeros((2, len(z_zgrid)))
zoofe = np.zeros((2, len(z_zgrid)))
detfe = np.zeros((2, len(z_zgrid)))

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
    pchl_init = np.linspace(t_pchl_top, t_pchl_bot, z_npt)
    phyfe_init = np.linspace(t_phy_top*7e-6, t_phy_bot*7e-6, z_npt)
    zoofe_init = np.linspace(t_zoo_top*7e-6, t_zoo_bot*7e-6, z_npt)
    detfe_init = np.linspace(t_det_top*7e-6, t_det_bot*7e-6, z_npt)
else:
    df = pd.read_csv("../output/restart_1Dlite.csv")
    o2_init  = df["O2"] 
    no3_init = df["NO3"]
    dfe_init = df["dFe"]
    phy_init = df["PHY"]
    zoo_init = df["ZOO"]
    det_init = df["DET"]
    cal_init = df["CAL"]
    alk_init = df["ALK"]
    dic_init = df["DIC"]
    pchl_init = df["pChl"]
    phyfe_init = df["PHYFe"]
    zoofe_init = df["ZOOFe"]
    detfe_init = df["DETFe"]
    
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
pchl[0,:] = pchl_init
phyfe[0,:] = phyfe_init
zoofe[0,:] = zoofe_init
detfe[0,:] = detfe_init

# initialise the sediment detritus and CaCO3 pool
seddet = np.zeros((2))
sedcal = np.zeros((2))
seddetfe = np.zeros((2))

# get the chlorophyll-dependent attenuation coefficients for RGB PAR
p_Chl_k = np.genfromtxt("inputs/rgb_attenuation_coefs.txt", delimiter="\t", skip_header=1)


#%% run model

%matplotlib inline

from tqdm import tqdm
from advec_diff import advec_diff, mix_mld
from bgc_sms_lite import bgc_sms_1D as bgc_sms
from plot_lite import plot1D
from sink import sink
from phyparams import z_dz, z_wup, z_Kv, z_mld, z_tmld
from mass_balance_lite import massb_n_1D, massb_c_1D, massb_f_1D

if conserving:
    z_wup = z_wup * 0
    z_Kv = z_Kv * 0
    mld_timeseries = mld_timeseries * 0 + 50.0

ts_per_day = ts_per_year / 365

for t in tqdm(np.arange(timesteps), desc="Running model", unit=' timesteps'): 
    
    #if t == 4500:
    #    break
    
    # position of timestep within each year
    ts_within_year = int(t % (365 * 86400 / dt))
    day = int(ts_within_year/ts_per_day + 1)
    z_mld = mld_timeseries[ts_within_year]
    
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
    pchl = advec_diff(dt, pchl, t_pchl_top, t_pchl_bot, z_dz, z_wup, z_Kv)
    phyfe = advec_diff(dt, phyfe, t_phy_top*7e-6, t_phy_bot*7e-6, z_dz, z_wup, z_Kv)
    zoofe = advec_diff(dt, zoofe, t_zoo_top*7e-6, t_zoo_bot*7e-6, z_dz, z_wup, z_Kv)
    detfe = advec_diff(dt, detfe, t_det_top*7e-6, t_det_bot*7e-6, z_dz, z_wup, z_Kv)
    
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
        pchl = mix_mld(dt, pchl, z_zgrid, z_mld, z_tmld)
        phyfe = mix_mld(dt, phyfe, z_zgrid, z_mld, z_tmld)
        zoofe = mix_mld(dt, zoofe, z_zgrid, z_mld, z_tmld)
        detfe = mix_mld(dt, detfe, z_zgrid, z_mld, z_tmld)
    
    if sinking:
        wdet = wdetbio * np.fmax(0.0, phy[0,0] - p_phy_biothresh)**0.21
        out = sink(det[1,:], wdet, z_dz, z_zgrid)
        det[1,:] = det[1,:] + out[0] * dt
        seddet[1] = seddet[1] + out[1] * dt 
        out = sink(cal[1,:], wcal, z_dz, z_zgrid)
        cal[1,:] = cal[1,:] + out[0] * dt
        sedcal[1] = sedcal[1] + out[1]*dt 
        out = sink(detfe[1,:], wdet, z_dz, z_zgrid)
        detfe[1,:] = detfe[1,:] + out[0] * dt
        seddetfe[1] = seddetfe[1] + out[1] * dt 

    if sourcesinks:
        par = np.fmax(rsds[ts_within_year],1e-16)
        tos = tas[ts_within_year]
        tob = t_tc_bot*1
        sms = bgc_sms(o2, no3, dfe, phy, zoo, det, cal, alk, dic,\
                      pchl, phyfe, zoofe, detfe,\
                      p_Chl_k, par, tos, tob,\
                      day, latitude, z_dz, z_mld, z_zgrid)
 
        o2[1,:]  = o2[1,:]  + sms[0][:] * dt
        no3[1,:] = no3[1,:] + sms[1][:] * dt
        dfe[1,:] = dfe[1,:] + sms[2][:] * dt
        phy[1,:] = phy[1,:] + sms[3][:] * dt
        zoo[1,:] = zoo[1,:] + sms[4][:] * dt
        det[1,:] = det[1,:] + sms[5][:] * dt
        cal[1,:] = cal[1,:] + sms[6][:] * dt
        alk[1,:] = alk[1,:] + sms[7][:] * dt
        dic[1,:] = dic[1,:] + sms[8][:] * dt
        pchl[1,:] = pchl[1,:] + sms[9][:] * dt
        phyfe[1,:] = phyfe[1,:] + sms[10][:] * dt
        zoofe[1,:] = zoofe[1,:] + sms[11][:] * dt
        detfe[1,:] = detfe[1,:] + sms[12][:] * dt
        
    
    if (t % 1) == 0:
        print("timestep #%i"%(t))
        print("Surface PHY = %.4f"%(phy[1,1]))
        print("Surface ZOO = %.4f"%(zoo[1,1]))
        print("Surface DET = %.4f"%(det[1,1]))
        print("Surface NO3 = %.4f"%(no3[1,1]))
        print("Surface dFe (nM) = %.4f"%(dfe[1,1]*1e3))
        print("Surface CaCO3 = %.4f"%(cal[1,1]))
        print("Surface Alk = %.4f"%(alk[1,1]))
        print("Surface DIC = %.4f"%(dic[1,1]))
        print("Surface Chl = %.4f"%(pchl[1,1]))
        print("Surface Chl:C ratio = %.4f"%(sms[-4][1]))
        print("Surface Phy Fe:C ratio = %.4f"%(sms[-3][1]))
        print("Surface Zoo Fe:C ratio = %.4f"%(sms[-2][1]))
        print("Surface Det Fe:C ratio = %.4f"%(sms[-1][1]))
        
            
    if plot:
        if (t % plot_freq) == 0:
            fig = plot1D(o2[1,:], no3[1,:], dfe[1,:], phy[1,:], zoo[1,:], \
                         det[1,:], cal[1,:], sms[13][:], sms[14][:], sms[-4][:], \
                         sms[-3][:], sms[-2][:], sms[-1][:], z_zgrid)
            fig.savefig("../figures/plot_lite_day_{0:08d}".format(int(t*dt/86400)))
            plt.clf()
            del fig
        
    thresh = 1e-8
    n0,n1,nerr = massb_n_1D(no3, phy, zoo, det, seddet, 122/16.0, thresh)
    c0,c1,cerr = massb_c_1D(dic, cal, phy, zoo, det, seddet, thresh)
    f0,f1,ferr = massb_f_1D(dfe, phyfe, zoofe, detfe, seddetfe, thresh)
    
    if conserving:
        if nerr == 1:
            print("Nitrogen not conserved at timestep %i"%(t))
            break
        if cerr == 1:
            print("Carbon not conserved at timestep %i"%(t))
            break    
        #Fe budget currently not closed because we don't track scavenging losses
        if ferr == 1:
            print("Iron not conserved at timestep %i"%(t))
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
    pchl[0,:] = pchl[1,:]
    phyfe[0,:] = phyfe[1,:]
    zoofe[0,:] = zoofe[1,:]
    detfe[0,:] = detfe[1,:]
    seddet[0] = seddet[1]
    sedcal[0] = sedcal[1]
    seddetfe[0] = seddetfe[1]


# get sources minus sinks one final time
sms = bgc_sms(o2, no3, dfe, phy, zoo, det, cal, alk, dic, \
              pchl, phyfe, zoofe, detfe, \
              p_Chl_k, rsds[ts_within_year], tas[ts_within_year], t_tc_bot,\
              day, latitude, z_dz, z_mld, z_zgrid)
    
phy_mu = sms[-8] * 86400
zoo_mu = sms[-7] * 86400
phy_lm = sms[-6] * 86400
phy_qm = sms[-5] * 86400
phy_chlc = sms[-4]
phy_FeC = sms[-3]
zoo_FeC = sms[-2]
det_FeC = sms[-1]

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
        "DIC":dic[1,:],
        "pChl":pchl[1,:],
        "PHYFe":phyfe[1,:],
        "ZOOFe":zoofe[1,:],
        "DETFe":detfe[1,:]}

df = pd.DataFrame(data)
df.to_csv('../output/restart_1Dlite.csv', header=True)    


# make figure of steady-state results
fig = plot1D(o2[1,:], no3[1,:], dfe[1,:], phy[1,:], zoo[1,:], \
             det[1,:], cal[1,:], alk[1,:], dic[1,:], phy_chlc, phy_FeC, zoo_FeC, det_FeC, \
             z_zgrid)


#%% make animation

if latitude < 0:
    latt = "%iS"%(np.abs(latitude))
else:
    latt = "%iN"%(latitude)

import ffmpeg

os.chdir(wrkdir + "/../figures")
(
    ffmpeg
    .input('plot_lite_day_%08d.png', framerate=24)
    .output("WOMBAT_lite_lowupwelling_lownitrate_"+latt+"_%idays_anim.mp4"%(days), vcodec='libx264', pix_fmt='yuv420p')
    .run()
)

# remove image files
for ii in np.arange(days):
    os.remove('plot_lite_day_{0:08d}.png'.format(ii))


