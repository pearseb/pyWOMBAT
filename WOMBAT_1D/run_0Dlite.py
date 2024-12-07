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
years = 5
days = 365 * years
dt = 86400.0/12
timesteps = days*86400/dt

# set location
latitude = -35
longitude = 220

# logicals
conserving = True
sourcesinks = True
initialise = True

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

from tra_init import *

o2  = np.zeros((2))
no3 = np.zeros((2))
dfe = np.zeros((2))
phy = np.zeros((2))
zoo = np.zeros((2))
det = np.zeros((2))
cal = np.zeros((2))
alk = np.zeros((2))
dic = np.zeros((2))
pchl = np.zeros((2))
phyfe = np.zeros((2))
zoofe = np.zeros((2))
detfe = np.zeros((2))

if initialise:
    o2_init  = t_o2_top
    no3_init = t_no3_top
    dfe_init = t_dfe_top
    phy_init = t_phy_top
    zoo_init = t_zoo_top
    det_init = t_det_top
    cal_init = t_cal_top
    alk_init = t_alk_top
    dic_init = t_dic_top
    pchl_init = t_pchl_top
    phyfe_init = t_phy_top*7e-6
    zoofe_init = t_zoo_top*7e-6
    detfe_init = t_det_top*7e-6
else:
    df = pd.read_csv("output/restart_0Dlite.csv")
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
    
o2[:]  = o2_init
no3[:] = no3_init
dfe[:] = dfe_init
dic[:] = dic_init
phy[:] = phy_init
zoo[:] = zoo_init
det[:] = det_init
cal[:] = cal_init
alk[:] = alk_init
dic[:] = dic_init
pchl[:] = pchl_init
phyfe[:] = phyfe_init
zoofe[:] = zoofe_init
detfe[:] = detfe_init

# initialise the sediment detritus and CaCO3 pool
seddet = np.zeros((2))
sedcal = np.zeros((2))
seddetfe = np.zeros((2))

# get the chlorophyll-dependent attenuation coefficients for RGB PAR
p_Chl_k = np.genfromtxt("inputs/rgb_attenuation_coefs.txt", delimiter="\t", skip_header=1)


#%% run model

#%matplotlib inline

from tqdm import tqdm
from bgc_sms_lite import bgc_sms_0D as bgc_sms
from plot_lite import plot0D
from mass_balance_lite import massb_n_0D, massb_c_0D, massb_f_0D

ts_per_day = ts_per_year / 365
o2_time = np.zeros(int(timesteps/ts_per_day))
no3_time = np.zeros(int(timesteps/ts_per_day))
dfe_time = np.zeros(int(timesteps/ts_per_day))
phy_time = np.zeros(int(timesteps/ts_per_day))
zoo_time = np.zeros(int(timesteps/ts_per_day))
det_time = np.zeros(int(timesteps/ts_per_day))
cal_time = np.zeros(int(timesteps/ts_per_day))
alk_time = np.zeros(int(timesteps/ts_per_day))
dic_time = np.zeros(int(timesteps/ts_per_day))
pchl_time = np.zeros(int(timesteps/ts_per_day))
phyfe_time = np.zeros(int(timesteps/ts_per_day))
zoofe_time = np.zeros(int(timesteps/ts_per_day))
detfe_time = np.zeros(int(timesteps/ts_per_day))



for t in tqdm(np.arange(timesteps), desc="Running model", unit=' timesteps'): 
    
    #if t == 4500:
    #    break
    
    ts_within_year = int(t % (365 * 86400 / dt))
    day = int(ts_within_year/ts_per_day + 1)
    z_mld = mld_timeseries[ts_within_year]
    par = np.fmax(rsds[ts_within_year],1e-16)
    tc = tas[ts_within_year]
    
    sms = bgc_sms(o2, no3, dfe, phy, zoo, det, cal, alk, dic, \
                  pchl, phyfe, zoofe, detfe,\
                  p_Chl_k, par, tc, day, latitude, z_mld)
 
    o2[1]  = o2[1]  + sms[0] * dt
    no3[1] = no3[1] + sms[1] * dt
    dfe[1] = dfe[1] + sms[2] * dt
    phy[1] = phy[1] + sms[3] * dt
    zoo[1] = zoo[1] + sms[4] * dt
    det[1] = det[1] + sms[5] * dt
    cal[1] = cal[1] + sms[6] * dt
    alk[1] = alk[1] + sms[7] * dt
    dic[1] = dic[1] + sms[8] * dt
    pchl[1] = pchl[1] + sms[9] * dt
    phyfe[1] = phyfe[1] + sms[10] * dt
    zoofe[1] = zoofe[1] + sms[11] * dt
    detfe[1] = detfe[1] + sms[12] * dt
        
    
    if (t % ts_per_day) == 0:
        print("timestep #%i"%(t))
        print(" PHY = %.4f"%(phy[1]))
        print(" ZOO = %.4f"%(zoo[1]))
        print(" DET = %.4f"%(det[1]))
        print(" NO3 = %.4f"%(no3[1]))
        print(" dFe = %.4f"%(dfe[1]))
        print(" CaCO3 = %.4f"%(cal[1]))
        print(" Alk = %.4f"%(alk[1]))
        print(" DIC = %.4f"%(dic[1]))
        print(" Chl = %.4f"%(pchl[1]))
        print(" Chl:C ratio = %.4f"%(sms[-4]))
        print(" Phy Fe:C ratio = %.4f"%(sms[-3]))
        print(" Zoo Fe:C ratio = %.4f"%(sms[-2]))
        print(" Det Fe:C ratio = %.4f"%(sms[-1]))
        o2_time[int(t/ts_per_day)] = o2[1]
        no3_time[int(t/ts_per_day)] = no3[1]
        dfe_time[int(t/ts_per_day)] = dfe[1]
        phy_time[int(t/ts_per_day)] = phy[1]
        zoo_time[int(t/ts_per_day)] = zoo[1]
        det_time[int(t/ts_per_day)] = det[1]
        cal_time[int(t/ts_per_day)] = cal[1]
        alk_time[int(t/ts_per_day)] = alk[1]
        dic_time[int(t/ts_per_day)] = dic[1]
        pchl_time[int(t/ts_per_day)] = pchl[1]
        phyfe_time[int(t/ts_per_day)] = phyfe[1]
        zoofe_time[int(t/ts_per_day)] = zoofe[1]
        detfe_time[int(t/ts_per_day)] = detfe[1]
        
            
    thresh = 1e-8
    n0,n1,nerr = massb_n_0D(no3, phy, zoo, det, seddet, 122/16.0, thresh)
    c0,c1,cerr = massb_c_0D(dic, cal, phy, zoo, det, seddet, thresh)
    f0,f1,ferr = massb_f_0D(dfe, phyfe, zoofe, detfe, seddetfe, thresh)
    
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
    o2[0]  = o2[1] 
    no3[0] = no3[1]
    dfe[0] = dfe[1]
    phy[0] = phy[1]
    zoo[0] = zoo[1]
    det[0] = det[1]
    cal[0] = cal[1]
    alk[0] = alk[1]
    dic[0] = dic[1]
    pchl[0] = pchl[1]
    phyfe[0] = phyfe[1]
    zoofe[0] = zoofe[1]
    detfe[0] = detfe[1]
    seddet[0] = seddet[1]
    sedcal[0] = sedcal[1]
    seddetfe[0] = seddetfe[1]


# get sources minus sinks one final time
sms = bgc_sms(o2, no3, dfe, phy, zoo, det, cal, alk, dic, \
              pchl, phyfe, zoofe, detfe,\
              p_Chl_k, par, tc, day, latitude, z_mld)

    
phy_mu = sms[-8] * 86400
zoo_mu = sms[-7] * 86400
phy_lm = sms[-6] * 86400
phy_qm = sms[-5] * 86400
phy_chlc = sms[-4]
phy_FeC = sms[-3]
zoo_FeC = sms[-2]
det_FeC = sms[-1]

# save the output to file
data = {"O2":o2[1],
        "NO3":no3[1],
        "dFe":dfe[1],
        "PHY":phy[1],
        "ZOO":zoo[1],
        "DET":det[1],
        "CAL":cal[1],
        "ALK":alk[1],
        "DIC":dic[1],
        "pChl":pchl[1],
        "PHYFe":phyfe[1],
        "ZOOFe":zoofe[1],
        "DETFe":detfe[1]}

df = pd.DataFrame(data, index=[0])
df.to_csv('../output/restart_0Dlite.csv', header=True)    


# make figure of steady-state results
fig = plot0D(o2_time, no3_time, dfe_time, phy_time, zoo_time, \
             det_time, cal_time, alk_time, dic_time, pchl_time/(phy_time*12), \
             phyfe_time/phy_time*1e6, zoofe_time/zoo_time*1e6, detfe_time/det_time*1e6)
fig.savefig('../figures/WOMBAT-lite_0D.png')


