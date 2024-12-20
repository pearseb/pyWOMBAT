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
years = 2
days = 365 * years
dt = 86400.0/12
timesteps = days*86400/dt

# set location
latitude = -60
longitude = 220

# set sinking speeds of detritus
wdet = 24.0    # sinking speed of detritus (m/d)
wcal = 6.0     # sinking speed of CaCO3 (m/d)

# logicals
conserving = False
do_mld_avg = True
sinking = True
sourcesinks = True
initialise = False
mld_var = True

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
nh4 = np.zeros((2, len(z_zgrid)))
no3 = np.zeros((2, len(z_zgrid)))
dfe = np.zeros((2, len(z_zgrid)))
phy = np.zeros((2, len(z_zgrid)))
dia = np.zeros((2, len(z_zgrid)))
zoo = np.zeros((2, len(z_zgrid)))
mes = np.zeros((2, len(z_zgrid)))
det = np.zeros((2, len(z_zgrid)))
poc = np.zeros((2, len(z_zgrid)))
cal = np.zeros((2, len(z_zgrid)))
alk = np.zeros((2, len(z_zgrid)))
dic = np.zeros((2, len(z_zgrid)))
pchl = np.zeros((2, len(z_zgrid)))
dchl = np.zeros((2, len(z_zgrid)))
phyfe = np.zeros((2, len(z_zgrid)))
diafe = np.zeros((2, len(z_zgrid)))
detfe = np.zeros((2, len(z_zgrid)))
pocfe = np.zeros((2, len(z_zgrid)))
zoofe = np.zeros((2, len(z_zgrid)))
mesfe = np.zeros((2, len(z_zgrid)))

if initialise:
    o2_init  = np.linspace(t_o2_top, t_o2_bot, z_npt)
    nh4_init = np.linspace(t_nh4_top, t_nh4_bot, z_npt)
    no3_init = np.linspace(t_no3_top, t_no3_bot, z_npt)
    dfe_init = np.linspace(t_dfe_top, t_dfe_bot, z_npt)
    phy_init = np.linspace(t_phy_top, t_phy_bot, z_npt)
    dia_init = np.linspace(t_dia_top, t_dia_bot, z_npt)
    zoo_init = np.linspace(t_zoo_top, t_zoo_bot, z_npt)
    mes_init = np.linspace(t_mes_top, t_mes_bot, z_npt)
    det_init = np.linspace(t_det_top, t_det_bot, z_npt)
    poc_init = np.linspace(t_poc_top, t_poc_bot, z_npt)
    cal_init = np.linspace(t_cal_top, t_cal_bot, z_npt)
    alk_init = np.linspace(t_alk_top, t_alk_bot, z_npt)
    dic_init = np.linspace(t_dic_top, t_dic_bot, z_npt)
    pchl_init = np.linspace(t_chl_top, t_chl_bot, z_npt)
    dchl_init = np.linspace(t_chl_top, t_chl_bot, z_npt)
    phyfe_init = phy_init*7e-6
    diafe_init = dia_init*7e-6
    detfe_init = det_init*7e-6
    pocfe_init = poc_init*7e-6
    zoofe_init = zoo_init*60e-6
    mesfe_init = mes_init*20e-6
else:
    df = pd.read_csv("../output/restart_2P2Z_1D_60S.csv")
    o2_init  = df["O2"] 
    nh4_init = df["NH4"]
    no3_init = df["NO3"]
    dfe_init = df["dFe"]
    phy_init = df["PHY"]
    dia_init = df["DIA"]
    zoo_init = df["ZOO"]
    mes_init = df["MES"]
    det_init = df["DET"]
    poc_init = df["POC"]
    cal_init = df["CAL"]
    alk_init = df["ALK"]
    dic_init = df["DIC"]
    pchl_init = df["pChl"]
    dchl_init = df["dChl"]
    phyfe_init = df["PHYFe"]
    diafe_init = df["DIAFe"]
    detfe_init = df["DETFe"]
    pocfe_init = df["POCFe"]
    zoofe_init = df["ZOOFe"]
    mesfe_init = df["MESFe"]
    

o2[0,:]  = o2_init
nh4[0,:] = nh4_init
no3[0,:] = no3_init
dfe[0,:] = dfe_init
dic[0,:] = dic_init
phy[0,:] = phy_init
dia[0,:] = dia_init
zoo[0,:] = zoo_init
mes[0,:] = mes_init
det[0,:] = det_init
poc[0,:] = poc_init
cal[0,:] = cal_init
alk[0,:] = alk_init
dic[0,:] = dic_init
pchl[0,:] = pchl_init
dchl[0,:] = dchl_init
phyfe[0,:] = phyfe_init
diafe[0,:] = diafe_init
detfe[0,:] = detfe_init
pocfe[0,:] = pocfe_init
zoofe[0,:] = zoofe_init
mesfe[0,:] = mesfe_init

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
from bgc_sms_1D import bgc_sms_2P2Z_nh4 as bgc_sms
from plot_1D import plot_1D_2P2Z_nh4 as plot_1D
from sink import sink
from phyparams import z_dz, z_wup, z_Kv, z_mld, z_tmld
from mass_balance_2P2Z import massb_n, massb_c, massb_f


if conserving:
    z_wup = z_wup * 0
    z_Kv = z_Kv * 0


for t in tqdm(np.arange(timesteps), desc="Running model", unit=' timesteps'): 
    
    #if t == 1000:
    #    break
    
    # position of timestep within each year
    ts_within_year = int(t % (365 * 86400 / dt))
    z_mld = mld_timeseries[ts_within_year]
    
    # do advection and diffusion on each tracer
    o2  = advec_diff(dt, o2,  t_o2_top,  t_o2_bot,  z_dz, z_wup, z_Kv)
    nh4 = advec_diff(dt, nh4, t_nh4_top, t_nh4_bot, z_dz, z_wup, z_Kv)
    no3 = advec_diff(dt, no3, t_no3_top, t_no3_bot, z_dz, z_wup, z_Kv)
    dfe = advec_diff(dt, dfe, t_dfe_top, t_dfe_bot, z_dz, z_wup, z_Kv)
    phy = advec_diff(dt, phy, t_phy_top, t_phy_bot, z_dz, z_wup, z_Kv)
    dia = advec_diff(dt, dia, t_dia_top, t_dia_bot, z_dz, z_wup, z_Kv)
    zoo = advec_diff(dt, zoo, t_zoo_top, t_zoo_bot, z_dz, z_wup, z_Kv)
    mes = advec_diff(dt, mes, t_mes_top, t_mes_bot, z_dz, z_wup, z_Kv)
    det = advec_diff(dt, det, t_det_top, t_det_bot, z_dz, z_wup, z_Kv)
    poc = advec_diff(dt, poc, t_poc_top, t_poc_bot, z_dz, z_wup, z_Kv)
    cal = advec_diff(dt, cal, t_cal_top, t_cal_bot, z_dz, z_wup, z_Kv)
    alk = advec_diff(dt, alk, t_alk_top, t_alk_bot, z_dz, z_wup, z_Kv)
    dic = advec_diff(dt, dic, t_dic_top, t_dic_bot, z_dz, z_wup, z_Kv)
    pchl = advec_diff(dt, pchl, t_chl_top, t_chl_bot, z_dz, z_wup, z_Kv)
    dchl = advec_diff(dt, dchl, t_chl_top, t_chl_bot, z_dz, z_wup, z_Kv)
    phyfe = advec_diff(dt, phyfe, t_phy_top*7e-6, t_phy_bot*7e-6, z_dz, z_wup, z_Kv)
    diafe = advec_diff(dt, diafe, t_dia_top*7e-6, t_dia_bot*7e-6, z_dz, z_wup, z_Kv)
    detfe = advec_diff(dt, detfe, t_det_top*7e-6, t_det_bot*7e-6, z_dz, z_wup, z_Kv)
    pocfe = advec_diff(dt, pocfe, t_poc_top*7e-6, t_poc_bot*7e-6, z_dz, z_wup, z_Kv)
    zoofe = advec_diff(dt, zoofe, t_zoo_top*60e-6, t_zoo_bot*60e-6, z_dz, z_wup, z_Kv)
    mesfe = advec_diff(dt, mesfe, t_mes_top*20e-6, t_mes_bot*20e-6, z_dz, z_wup, z_Kv)
    
    if do_mld_avg and conserving==False:
        o2 = mix_mld(dt, o2, z_zgrid, z_mld, z_tmld)
        nh4 = mix_mld(dt, nh4, z_zgrid, z_mld, z_tmld)
        no3 = mix_mld(dt, no3, z_zgrid, z_mld, z_tmld)
        dfe = mix_mld(dt, dfe, z_zgrid, z_mld, z_tmld)
        phy = mix_mld(dt, phy, z_zgrid, z_mld, z_tmld)
        dia = mix_mld(dt, dia, z_zgrid, z_mld, z_tmld)
        zoo = mix_mld(dt, zoo, z_zgrid, z_mld, z_tmld)
        mes = mix_mld(dt, mes, z_zgrid, z_mld, z_tmld)
        det = mix_mld(dt, det, z_zgrid, z_mld, z_tmld)
        poc = mix_mld(dt, poc, z_zgrid, z_mld, z_tmld)
        cal = mix_mld(dt, cal, z_zgrid, z_mld, z_tmld)
        alk = mix_mld(dt, alk, z_zgrid, z_mld, z_tmld)
        dic = mix_mld(dt, dic, z_zgrid, z_mld, z_tmld)
        pchl = mix_mld(dt, pchl, z_zgrid, z_mld, z_tmld)
        dchl = mix_mld(dt, dchl, z_zgrid, z_mld, z_tmld)
        phyfe = mix_mld(dt, phyfe, z_zgrid, z_mld, z_tmld)
        diafe = mix_mld(dt, diafe, z_zgrid, z_mld, z_tmld)
        detfe = mix_mld(dt, detfe, z_zgrid, z_mld, z_tmld)
        pocfe = mix_mld(dt, pocfe, z_zgrid, z_mld, z_tmld)
        zoofe = mix_mld(dt, zoofe, z_zgrid, z_mld, z_tmld)
        mesfe = mix_mld(dt, mesfe, z_zgrid, z_mld, z_tmld)
    
    if sinking:
        out = sink(det[1,:], wdet*0.1, z_dz, z_zgrid)
        det[1,:] = det[1,:] + out[0] * dt
        seddet[1] = seddet[1] + out[1]*dt 
        out = sink(detfe[1,:], wdet*0.1, z_dz, z_zgrid)
        detfe[1,:] = detfe[1,:] + out[0] * dt
        seddetfe[1] = seddetfe[1] + out[1]*dt 
        out = sink(poc[1,:], wdet, z_dz, z_zgrid)
        poc[1,:] = poc[1,:] + out[0] * dt
        seddet[1] = seddet[1] + out[1]*dt 
        out = sink(pocfe[1,:], wdet, z_dz, z_zgrid)
        pocfe[1,:] = pocfe[1,:] + out[0] * dt
        seddetfe[1] = seddetfe[1] + out[1]*dt 
        out = sink(cal[1,:], wcal, z_dz, z_zgrid)
        cal[1,:] = cal[1,:] + out[0] * dt
        sedcal[1] = sedcal[1] + out[1]*dt 
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

    # get sources minus sinks
    if sourcesinks:
        day = ts_within_year*dt/86400
        par = np.fmax(rsds[ts_within_year],1e-16)
        tos = tas[ts_within_year]
        sms = bgc_sms(o2, nh4, no3, dfe, \
                      phy, dia, zoo, mes, \
                      det, poc, cal, alk, dic, \
                      pchl, dchl, \
                      phyfe, diafe, detfe, pocfe, zoofe, mesfe, \
                      p_Chl_k, par, tos, t_tc_bot,\
                      z_dz, z_mld, z_zgrid, day, latitude)
        
        o2[1,:]  = o2[1,:]  + sms[0][:] * dt
        nh4[1,:] = nh4[1,:] + sms[1][:] * dt
        no3[1,:] = no3[1,:] + sms[2][:] * dt
        dfe[1,:] = dfe[1,:] + sms[3][:] * dt
        phy[1,:] = phy[1,:] + sms[4][:] * dt
        dia[1,:] = dia[1,:] + sms[5][:] * dt
        zoo[1,:] = zoo[1,:] + sms[6][:] * dt
        mes[1,:] = mes[1,:] + sms[7][:] * dt
        det[1,:] = det[1,:] + sms[8][:] * dt
        poc[1,:] = poc[1,:] + sms[9][:] * dt
        cal[1,:] = cal[1,:] + sms[10][:] * dt
        alk[1,:] = alk[1,:] + sms[11][:] * dt
        dic[1,:] = dic[1,:] + sms[12][:] * dt
        pchl[1,:] = pchl[1,:] + sms[13][:] * dt
        dchl[1,:] = dchl[1,:] + sms[14][:] * dt
        phyfe[1,:] = phyfe[1,:] + sms[15][:] * dt
        diafe[1,:] = diafe[1,:] + sms[16][:] * dt
        detfe[1,:] = detfe[1,:] + sms[17][:] * dt
        pocfe[1,:] = pocfe[1,:] + sms[18][:] * dt
        zoofe[1,:] = zoofe[1,:] + sms[19][:] * dt
        mesfe[1,:] = mesfe[1,:] + sms[20][:] * dt
        
    
    if (t % 100) == 0:
        print("timestep #%i"%(t))
        print("Surface PHY = %.4f"%(phy[1,1]))
        print("Surface DIA = %.4f"%(dia[1,1]))
        print("Surface ZOO = %.4f"%(zoo[1,1]))
        print("Surface MES = %.4f"%(mes[1,1]))
        print("Surface DET = %.4f"%(det[1,1]))
        print("Surface POC = %.4f"%(poc[1,1]))
        print("Surface NH4 = %.4f"%(nh4[1,1]))
        print("Surface NO3 = %.4f"%(no3[1,1]))
        print("Surface dFe = %.4f"%(dfe[1,1]))
        print("Surface CaCO3 = %.4f"%(cal[1,1]))
        print("Surface Alk = %.4f"%(alk[1,1]))
        print("Surface DIC = %.4f"%(dic[1,1]))
        print("Surface pChl:C = %.4f"%(sms[-8][1]))
        print("Surface dChl:C = %.4f"%(sms[-7][1]))
        print("Surface PHY Fe:C (µmol/mol) = %.4f"%(sms[-6][1]))
        print("Surface DIA Fe:C (µmol/mol) = %.4f"%(sms[-5][1]))
        print("Surface DET Fe:C (µmol/mol) = %.4f"%(sms[-4][1]))
        print("Surface POC Fe:C (µmol/mol) = %.4f"%(sms[-3][1]))
        print("Surface ZOO Fe:C (µmol/mol) = %.4f"%(sms[-2][1]))
        print("Surface MES Fe:C (µmol/mol) = %.4f"%(sms[-1][1]))
            
    if plot:
        if (t % plot_freq) == 0:
            fig = plot_1D(o2[1,:], nh4[1,:], no3[1,:], dfe[1,:], phy[1,:], dia[1,:], zoo[1,:], mes[1,:], det[1,:], poc[1,:], cal[1,:], \
                          sms[-8][:], sms[-7][:], sms[-6][:], sms[-5][:], sms[-4][:], sms[-3][:], sms[-2][:], sms[-1][:], \
                          sms[-18][:]*86400, sms[-17][:]*86400, z_zgrid)
            fig.savefig("../figures/plot_1D_day_{0:08d}".format(int(t*dt/86400)))
            plt.clf()
            del fig
        
           
    thresh = 1e-8
    n0,n1,nerr = massb_n(no3, phy, dia, zoo, mes, det, \
                         thresh)
    c0,c1,cerr = massb_c(dic, cal, phy, dia, zoo, mes, det, 106/16.0, \
                         thresh)
    #f0,f1,ferr = massb_f(dfe, phy, dia, zoo, mes, det, 106/16.0 * 7.1e-5, \
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
    nh4[0,:] = nh4[1,:]
    no3[0,:] = no3[1,:]
    dfe[0,:] = dfe[1,:]
    phy[0,:] = phy[1,:]
    dia[0,:] = dia[1,:]
    zoo[0,:] = zoo[1,:]
    mes[0,:] = mes[1,:]
    det[0,:] = det[1,:]
    poc[0,:] = poc[1,:]
    cal[0,:] = cal[1,:]
    alk[0,:] = alk[1,:]
    dic[0,:] = dic[1,:]
    pchl[0,:] = pchl[1,:]
    dchl[0,:] = dchl[1,:]
    phyfe[0,:] = phyfe[1,:]
    diafe[0,:] = diafe[1,:]
    detfe[0,:] = detfe[1,:]
    pocfe[0,:] = pocfe[1,:]
    zoofe[0,:] = zoofe[1,:]
    mesfe[0,:] = mesfe[1,:]
    seddet[0] = seddet[1]
    sedcal[0] = sedcal[1]
    seddetfe[0] = seddetfe[1]


# get sources minus sinks one final time
par = np.fmax(rsds[ts_within_year],1e-16)
tos = tas[ts_within_year]
sms = bgc_sms(o2, nh4, no3, dfe, phy, dia, zoo, mes, det, poc, cal, alk, dic, \
              pchl, dchl, phyfe, diafe, detfe, pocfe, zoofe, mesfe, \
              p_Chl_k, par, tos, t_tc_bot,\
              z_dz, z_mld, z_zgrid, day, latitude)

    
pgi_zoo = sms[-18] * 86400
pgi_mes = sms[-17] * 86400
phy_mu = sms[-16] * 86400
dia_mu = sms[-15] * 86400
zoo_mu = sms[-14] * 86400
mes_mu = sms[-13] * 86400
phy_lm = sms[-12] * 86400
dia_lm = sms[-11] * 86400
phy_qm = sms[-10] * 86400
dia_qm = sms[-9] * 86400
phy_chlc = sms[-8]
dia_chlc = sms[-7]
phy_FeC = sms[-6]
dia_FeC = sms[-5]
det_FeC = sms[-4]
poc_FeC = sms[-3]
zoo_FeC = sms[-2]
mes_FeC = sms[-1]


#%% save restart file

# save the output to file
data = {"depth":z_zgrid,
        "O2":o2[1,:],
        "NH4":nh4[1,:],
        "NO3":no3[1,:],
        "dFe":dfe[1,:],
        "PHY":phy[1,:],
        "DIA":dia[1,:],
        "ZOO":zoo[1,:],
        "MES":mes[1,:],
        "DET":det[1,:],
        "POC":poc[1,:],
        "CAL":cal[1,:],
        "ALK":alk[1,:],
        "DIC":dic[1,:],
        "pChl":pchl[1,:],
        "dChl":dchl[1,:],
        "PHYFe":phyfe[1,:],
        "DIAFe":diafe[1,:],
        "DETFe":detfe[1,:],
        "POCFe":pocfe[1,:],
        "ZOOFe":zoofe[1,:],
        "MESFe":mesfe[1,:]}

df = pd.DataFrame(data)
df.to_csv('../output/restart_2P2Z_1D_60S.csv', header=True)    


# make figure of steady-state results
fig = plot_1D(o2[1,:], nh4[1,:], no3[1,:], dfe[1,:], phy[1,:], dia[1,:], zoo[1,:], mes[1,:], \
              det[1,:], poc[1,:], cal[1,:], phy_chlc, dia_chlc, phy_FeC, dia_FeC, det_FeC, poc_FeC, zoo_FeC, mes_FeC, \
              pgi_zoo, pgi_mes, z_zgrid)




#%% make animation

if latitude < 0:
    latt = "%iS"%(np.abs(latitude))
else:
    latt = "%iN"%(latitude)

import ffmpeg

os.chdir(wrkdir + "/../figures")
(
    ffmpeg
    .input('plot_1D_day_%08d.png', framerate=24)
    .output("WOMBAT_1D_"+latt+"_%idays_anim.mp4"%(days), vcodec='libx264', pix_fmt='yuv420p')
    .run()
)

# remove image files
for ii in np.arange(days):
    os.remove('plot_1D_day_{0:08d}.png'.format(ii))


