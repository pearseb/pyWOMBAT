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

wrkdir = "/Users/pbuchanan/Dropbox/PostDoc/ROMS/ROMS_1D_python"
os.chdir(wrkdir)


#%% get parameters 

# set timestep
days = 10000
dt = 86400.0/6
timesteps = days*86400/dt
conserving = False
do_mld_avg = True
darwin_arch = True
initialise = True



#%% initialise tracers

from phyparams import z_zgrid, z_npt
from traparams import *

o2  = np.zeros((2, len(z_zgrid)))
n2  = np.zeros((2, len(z_zgrid)))
no3 = np.zeros((2, len(z_zgrid)))
no2 = np.zeros((2, len(z_zgrid)))
nh4 = np.zeros((2, len(z_zgrid)))
po4 = np.zeros((2, len(z_zgrid)))
n2o = np.zeros((2, len(z_zgrid)))
dic = np.zeros((2, len(z_zgrid)))
doc = np.zeros((2, len(z_zgrid)))
dop = np.zeros((2, len(z_zgrid)))
don = np.zeros((2, len(z_zgrid)))
rdoc = np.zeros((2, len(z_zgrid)))
rdop = np.zeros((2, len(z_zgrid)))
rdon = np.zeros((2, len(z_zgrid)))
phy = np.zeros((2, len(z_zgrid)))
dia = np.zeros((2, len(z_zgrid)))
aoa = np.zeros((2, len(z_zgrid)))
nob = np.zeros((2, len(z_zgrid)))
aox = np.zeros((2, len(z_zgrid)))
nar = np.zeros((2, len(z_zgrid)))
nai = np.zeros((2, len(z_zgrid)))
nir = np.zeros((2, len(z_zgrid)))
nos = np.zeros((2, len(z_zgrid)))
fnar = np.zeros((2, len(z_zgrid)))
fnai = np.zeros((2, len(z_zgrid)))
fnir = np.zeros((2, len(z_zgrid)))
fnos = np.zeros((2, len(z_zgrid)))
zoo = np.zeros((2, len(z_zgrid)))
mes = np.zeros((2, len(z_zgrid)))

if initialise:
    o2_init  = np.linspace(t_o2_top, t_o2_bot, z_npt)
    n2_init  = np.linspace(t_n2_top, t_n2_bot, z_npt)
    no3_init = np.linspace(t_no3_top, t_no3_bot, z_npt)
    no2_init = np.linspace(t_no2_top, t_no2_bot, z_npt)
    nh4_init = np.linspace(t_nh4_top, t_nh4_bot, z_npt)
    n2o_init = np.linspace(t_n2o_top, t_n2o_bot, z_npt)
    po4_init = np.linspace(t_po4_top, t_po4_bot, z_npt)
    dic_init = np.linspace(t_dic_top, t_dic_bot, z_npt)
    doc_init = np.linspace(t_doc_top, t_doc_bot, z_npt)
    dop_init = np.linspace(t_dop_top, t_dop_bot, z_npt)
    don_init = np.linspace(t_don_top, t_don_bot, z_npt)
    rdoc_init = np.linspace(t_rdoc_top, t_rdoc_bot, z_npt)
    rdop_init = np.linspace(t_rdop_top, t_rdop_bot, z_npt)
    rdon_init = np.linspace(t_rdon_top, t_rdon_bot, z_npt)
    phy_init = np.linspace(t_phy_top, t_phy_bot, z_npt)
    dia_init = np.linspace(t_dia_top, t_dia_bot, z_npt)
    aoa_init = np.linspace(t_aoa_top, t_aoa_bot, z_npt)
    nob_init = np.linspace(t_nob_top, t_nob_bot, z_npt)
    aox_init = np.linspace(t_aox_top, t_aox_bot, z_npt)
    nar_init = np.linspace(t_nar_top, t_nar_bot, z_npt)
    nai_init = np.linspace(t_nai_top, t_nai_bot, z_npt)
    nir_init = np.linspace(t_nir_top, t_nir_bot, z_npt)
    nos_init = np.linspace(t_nos_top, t_nos_bot, z_npt)
    fnar_init = np.linspace(t_fnar_top, t_fnar_bot, z_npt)
    fnai_init = np.linspace(t_fnai_top, t_fnai_bot, z_npt)
    fnir_init = np.linspace(t_fnir_top, t_fnir_bot, z_npt)
    fnos_init = np.linspace(t_fnos_top, t_fnos_bot, z_npt)
    zoo_init = np.linspace(t_zoo_top, t_zoo_bot, z_npt)
    mes_init = np.linspace(t_mes_top, t_mes_bot, z_npt)
else:
    if darwin_arch:
        df = pd.read_csv("output/darwin_arch.csv")
    else:
        df = pd.read_csv("output/default.csv")
    o2_init  = df["O2"] 
    n2_init  = df["N2"] 
    no3_init = df["NO3"]
    no2_init = df["NO2"]
    nh4_init = df["NH4"]
    n2o_init = df["N2O"]
    po4_init = df["PO4"]
    dic_init = df["DIC"]
    doc_init = df["DOC"]
    dop_init = df["DOP"]
    don_init = df["DON"]
    rdoc_init = df["rDOC"]
    rdop_init = df["rDOP"]
    rdon_init = df["rDON"]
    phy_init = df["PHY"]
    dia_init = df["DIA"]
    aoa_init = df["AOA"]
    nob_init = df["NOB"]
    aox_init = df["AOX"]
    nar_init = df["NAR"]
    nai_init = df["NAI"]
    nir_init = df["NIR"]
    nos_init = df["NOS"]
    fnar_init = df["fNAR"]
    fnai_init = df["fNAI"]
    fnir_init = df["fNIR"]
    fnos_init = df["fNOS"]
    zoo_init = df["ZOO"]
    mes_init = df["MES"]


o2[0,:]  = o2_init
n2[0,:]  = n2_init
no3[0,:] = no3_init
no2[0,:] = no2_init
nh4[0,:] = nh4_init
n2o[0,:] = n2o_init
po4[0,:] = po4_init
dic[0,:] = dic_init
doc[0,:] = doc_init
dop[0,:] = dop_init
don[0,:] = don_init
rdoc[0,:] = rdoc_init
rdop[0,:] = rdop_init
rdon[0,:] = rdon_init
phy[0,:] = phy_init
dia[0,:] = dia_init
aoa[0,:] = aoa_init
nob[0,:] = nob_init
aox[0,:] = aox_init
nar[0,:] = nar_init
nai[0,:] = nai_init
nir[0,:] = nir_init
nos[0,:] = nos_init
fnar[0,:] = fnar_init
fnai[0,:] = fnai_init
fnir[0,:] = fnir_init
fnos[0,:] = fnos_init
zoo[0,:] = zoo_init
mes[0,:] = mes_init
    


#%% run model

from tqdm import tqdm
from advec_diff import advec_diff, mix_mld
from bgc_sms import bgc_sms
from phyparams import z_dz, z_wup, z_Kv, z_mld, z_tmld
from mass_balance import massb_c, massb_p, massb_n

# get some parameters required to calculate mass balances
from bgcparams import p_aoa_CN, p_nob_CN, p_aox_CN
from bgcparams import p_nar_CN, p_nai_CN, p_nir_CN, p_nos_CN
from bgcparams import p_aoa_CP, p_nob_CP, p_aox_CP
from bgcparams import p_nar_CP, p_nai_CP, p_nir_CP, p_nos_CP
from bgcparams import p_pom_CN, p_pom_CP


if conserving:
    z_wup = z_wup * 0
    z_Kv = z_Kv * 0


for t in tqdm(np.arange(timesteps), desc="Running model", unit=' timesteps'): 
    
    # do advection and diffusion on each tracer
    o2  = advec_diff(dt, o2,  t_o2_top,  t_o2_bot,  z_dz, z_wup, z_Kv)
    n2  = advec_diff(dt, n2,  t_n2_top,  t_n2_bot,  z_dz, z_wup, z_Kv)
    no3 = advec_diff(dt, no3, t_no3_top, t_no3_bot, z_dz, z_wup, z_Kv)
    no2 = advec_diff(dt, no2, t_no2_top, t_no2_bot, z_dz, z_wup, z_Kv)
    nh4 = advec_diff(dt, nh4, t_nh4_top, t_nh4_bot, z_dz, z_wup, z_Kv)
    n2o = advec_diff(dt, n2o, t_n2o_top, t_n2o_bot, z_dz, z_wup, z_Kv)
    po4 = advec_diff(dt, po4, t_po4_top, t_po4_bot, z_dz, z_wup, z_Kv)
    dic = advec_diff(dt, dic, t_dic_top, t_dic_bot, z_dz, z_wup, z_Kv)
    doc = advec_diff(dt, doc, t_doc_top, t_doc_bot, z_dz, z_wup, z_Kv)
    dop = advec_diff(dt, dop, t_dop_top, t_dop_bot, z_dz, z_wup, z_Kv)
    don = advec_diff(dt, don, t_don_top, t_don_bot, z_dz, z_wup, z_Kv)
    rdoc = advec_diff(dt, rdoc, t_rdoc_top, t_rdoc_bot, z_dz, z_wup, z_Kv)
    rdop = advec_diff(dt, rdop, t_rdop_top, t_rdop_bot, z_dz, z_wup, z_Kv)
    rdon = advec_diff(dt, rdon, t_rdon_top, t_rdon_bot, z_dz, z_wup, z_Kv)
    phy = advec_diff(dt, phy, t_phy_top, t_phy_bot, z_dz, z_wup, z_Kv*2)
    dia = advec_diff(dt, dia, t_dia_top, t_dia_bot, z_dz, z_wup, z_Kv*2)
    aoa = advec_diff(dt, aoa, t_aoa_top, t_aoa_bot, z_dz, z_wup, z_Kv)
    nob = advec_diff(dt, nob, t_nob_top, t_nob_bot, z_dz, z_wup, z_Kv)
    aox = advec_diff(dt, aox, t_aox_top, t_aox_bot, z_dz, z_wup, z_Kv)
    nar = advec_diff(dt, nar, t_nar_top, t_nar_bot, z_dz, z_wup, z_Kv)
    nai = advec_diff(dt, nai, t_nai_top, t_nai_bot, z_dz, z_wup, z_Kv)
    nir = advec_diff(dt, nir, t_nir_top, t_nir_bot, z_dz, z_wup, z_Kv)
    nos = advec_diff(dt, nos, t_nos_top, t_nos_bot, z_dz, z_wup, z_Kv)
    fnar = advec_diff(dt, fnar, t_fnar_top, t_fnar_bot, z_dz, z_wup, z_Kv)
    fnai = advec_diff(dt, fnai, t_fnai_top, t_fnai_bot, z_dz, z_wup, z_Kv)
    fnir = advec_diff(dt, fnir, t_fnir_top, t_fnir_bot, z_dz, z_wup, z_Kv)
    fnos = advec_diff(dt, fnos, t_fnos_top, t_fnos_bot, z_dz, z_wup, z_Kv)
    zoo = advec_diff(dt, zoo, t_zoo_top, t_zoo_bot, z_dz, z_wup, z_Kv*3)
    mes = advec_diff(dt, mes, t_mes_top, t_mes_bot, z_dz, z_wup, z_Kv*4)   
    
    if do_mld_avg and conserving==False:
        o2 = mix_mld(dt, o2, z_zgrid, z_mld, z_tmld)
        n2 = mix_mld(dt, n2, z_zgrid, z_mld, z_tmld)
        no3 = mix_mld(dt, no3, z_zgrid, z_mld, z_tmld)
        no2 = mix_mld(dt, no2, z_zgrid, z_mld, z_tmld)
        nh4 = mix_mld(dt, nh4, z_zgrid, z_mld, z_tmld)
        n2o = mix_mld(dt, n2o, z_zgrid, z_mld, z_tmld)
        po4 = mix_mld(dt, po4, z_zgrid, z_mld, z_tmld)
        dic = mix_mld(dt, dic, z_zgrid, z_mld, z_tmld)
        doc = mix_mld(dt, doc, z_zgrid, z_mld, z_tmld)
        dop = mix_mld(dt, dop, z_zgrid, z_mld, z_tmld)
        don = mix_mld(dt, don, z_zgrid, z_mld, z_tmld)
        rdoc = mix_mld(dt, rdoc, z_zgrid, z_mld, z_tmld)
        rdop = mix_mld(dt, rdop, z_zgrid, z_mld, z_tmld)
        rdon = mix_mld(dt, rdon, z_zgrid, z_mld, z_tmld)
        phy = mix_mld(dt, phy, z_zgrid, z_mld, z_tmld)
        dia = mix_mld(dt, dia, z_zgrid, z_mld, z_tmld)
        aoa = mix_mld(dt, aoa, z_zgrid, z_mld, z_tmld)
        nob = mix_mld(dt, nob, z_zgrid, z_mld, z_tmld)
        aox = mix_mld(dt, aox, z_zgrid, z_mld, z_tmld)
        nar = mix_mld(dt, nar, z_zgrid, z_mld, z_tmld)
        nai = mix_mld(dt, nai, z_zgrid, z_mld, z_tmld)
        nir = mix_mld(dt, nir, z_zgrid, z_mld, z_tmld)
        nos = mix_mld(dt, nos, z_zgrid, z_mld, z_tmld)
        fnar = mix_mld(dt, fnar, z_zgrid, z_mld, z_tmld)
        fnai = mix_mld(dt, fnai, z_zgrid, z_mld, z_tmld)
        fnir = mix_mld(dt, fnir, z_zgrid, z_mld, z_tmld)
        fnos = mix_mld(dt, fnos, z_zgrid, z_mld, z_tmld)
        zoo = mix_mld(dt, zoo, z_zgrid, z_mld, z_tmld)
        mes = mix_mld(dt, mes, z_zgrid, z_mld, z_tmld)
    
    # get sources minus sinks
    sms = bgc_sms(o2, n2, no3, no2, nh4, n2o, po4, dic, \
                  doc, dop, don, rdoc, rdop, rdon, \
                  phy, dia, \
                  aoa, nob, aox, \
                  nar, nai, nir, nos, \
                  fnar, fnai, fnir, fnos, \
                  zoo, mes, \
                  z_dz, z_zgrid, t_poc_flux_top, \
                  conserving, darwin_arch)
    
    # Apply sources and sinks to tracers
    o2[1,1:-1]  = o2[1,1:-1]  + sms[0][1:-1]  * dt
    n2[1,1:-1]  = n2[1,1:-1]  + sms[1][1:-1]  * dt
    no3[1,1:-1] = no3[1,1:-1] + sms[2][1:-1]  * dt
    no2[1,1:-1] = no2[1,1:-1] + sms[3][1:-1]  * dt
    nh4[1,1:-1] = nh4[1,1:-1] + sms[4][1:-1]  * dt
    n2o[1,1:-1] = n2o[1,1:-1] + sms[5][1:-1]  * dt
    po4[1,1:-1] = po4[1,1:-1] + sms[6][1:-1]  * dt
    dic[1,1:-1] = dic[1,1:-1] + sms[7][1:-1]  * dt
    doc[1,1:-1] = doc[1,1:-1] + sms[8][1:-1]  * dt
    dop[1,1:-1] = dop[1,1:-1] + sms[9][1:-1]  * dt
    don[1,1:-1] = don[1,1:-1] + sms[10][1:-1] * dt
    rdoc[1,1:-1] = rdoc[1,1:-1] + sms[11][1:-1]  * dt
    rdop[1,1:-1] = rdop[1,1:-1] + sms[12][1:-1]  * dt
    rdon[1,1:-1] = rdon[1,1:-1] + sms[13][1:-1] * dt
    phy[1,1:-1] = phy[1,1:-1] + sms[14][1:-1] * dt
    dia[1,1:-1] = dia[1,1:-1] + sms[15][1:-1] * dt
    aoa[1,1:-1] = aoa[1,1:-1] + sms[16][1:-1] * dt
    nob[1,1:-1] = nob[1,1:-1] + sms[17][1:-1] * dt
    aox[1,1:-1] = aox[1,1:-1] + sms[18][1:-1] * dt
    nar[1,1:-1] = nar[1,1:-1] + sms[19][1:-1] * dt
    nai[1,1:-1] = nai[1,1:-1] + sms[20][1:-1] * dt
    nir[1,1:-1] = nir[1,1:-1] + sms[21][1:-1] * dt
    nos[1,1:-1] = nos[1,1:-1] + sms[22][1:-1] * dt
    fnar[1,1:-1] = fnar[1,1:-1] + sms[23][1:-1] * dt
    fnai[1,1:-1] = fnai[1,1:-1] + sms[24][1:-1] * dt
    fnir[1,1:-1] = fnir[1,1:-1] + sms[25][1:-1] * dt
    fnos[1,1:-1] = fnos[1,1:-1] + sms[26][1:-1] * dt
    zoo[1,1:-1] = zoo[1,1:-1] + sms[27][1:-1] * dt
    mes[1,1:-1] = mes[1,1:-1] + sms[28][1:-1] * dt
    
    if (t % 5000) == 0:
        print("timestep #%i"%(t))
        print("Surface PHY = %.4f"%(phy[1,1]))
        print("Surface DIA = %.4f"%(dia[1,1]))
        print("Surface ZOO = %.4f"%(zoo[1,1]))
        print("Surface MES = %.4f"%(mes[1,1]))
        print("Surface NH4 = %.4f"%(nh4[1,1]))
        print("Surface NO2 = %.4f"%(no2[1,1]))
        print("Surface NO3 = %.4f"%(no3[1,1]))
        print("Surface PO4 = %.4f"%(po4[1,1]))
        print("Surface DOC = %.4f"%(doc[1,1]))
        print("Surface DOC/DOP = %.4f"%(doc[1,1]/dop[1,1]))
        print("Surface DOC/DON = %.4f"%(doc[1,1]/don[1,1]))
        print("Surface Chemos = %.4f"%(aoa[1,1]+nob[1,1]+aox[1,1]))
        print("Surface slow Hets = %.4f"%(nar[1,1]+nai[1,1]+nir[1,1]+nos[1,1]))
        print("Surface fast Hets = %.4f"%(fnar[1,1]+fnai[1,1]+fnir[1,1]+fnos[1,1]))
        
        
    
    thresh = 1e-8
    c0,c1,cerr = massb_c(dic, doc, rdoc, phy, dia, \
                         aoa, nob, aox, \
                         nar, nai, nir, nos, \
                         fnar, fnai, fnir, fnos, \
                         zoo, mes, thresh)
    p0,p1,perr = massb_p(po4, dop, rdop, phy, dia, \
                         aoa, nob, aox, \
                         nar, nai, nir, nos, \
                         fnar, fnai, fnir, fnos, \
                         zoo, mes, \
                         p_aoa_CP, p_nob_CP, p_aox_CP, p_nar_CP, p_nai_CP, p_nir_CP, p_nos_CP, p_pom_CP, \
                         thresh)
    n0,n1,nerr = massb_n(n2, no3, no2, nh4, n2o, don, rdon, phy, dia, \
                         aoa, nob, aox, \
                         nar, nai, nir, nos, \
                         fnar, fnai, fnir, fnos, \
                         zoo, mes, \
                         p_aoa_CN, p_nob_CN, p_aox_CN, p_nar_CN, p_nai_CN, p_nir_CN, p_nos_CN, p_pom_CN, \
                         thresh)
        
    if conserving:
        if cerr == 1:
            print("Carbon not conserved at timestep %i"%(t))
            break    
        
        if perr == 1:
            print("Phosphorus not conserved at timestep %i"%(t))
            break
        
        if nerr == 1:
            print("Nitrogen not conserved at timestep %i"%(t))
            break
    else:
        if cerr == 0 and perr == 0 and nerr == 0:
            print("Arrived at steady state")
            break
    
    
    # old becomes new
    o2[0,:]  = o2[1,:] 
    n2[0,:]  = n2[1,:]
    no3[0,:] = no3[1,:]
    no2[0,:] = no2[1,:]
    nh4[0,:] = nh4[1,:]
    n2o[0,:] = n2o[1,:]
    po4[0,:] = po4[1,:]
    dic[0,:] = dic[1,:]
    doc[0,:] = doc[1,:]
    dop[0,:] = dop[1,:]
    don[0,:] = don[1,:]
    rdoc[0,:] = rdoc[1,:]
    rdop[0,:] = rdop[1,:]
    rdon[0,:] = rdon[1,:]
    phy[0,:] = phy[1,:]
    dia[0,:] = dia[1,:]
    aoa[0,:] = aoa[1,:]
    nob[0,:] = nob[1,:]
    aox[0,:] = aox[1,:]
    nar[0,:] = nar[1,:]
    nai[0,:] = nai[1,:]
    nir[0,:] = nir[1,:]
    nos[0,:] = nos[1,:]
    fnar[0,:] = fnar[1,:]
    fnai[0,:] = fnai[1,:]
    fnir[0,:] = fnir[1,:]
    fnos[0,:] = fnos[1,:]
    zoo[0,:] = zoo[1,:]
    mes[0,:] = mes[1,:]
    
    
sms = bgc_sms(o2, n2, no3, no2, nh4, n2o, po4, dic, \
              doc, dop, don, rdoc, rdop, rdon, \
              phy, dia, \
              aoa, nob, aox, \
              nar, nai, nir, nos, \
              fnar, fnai, fnir, fnos, \
              zoo, mes, \
              z_dz, z_zgrid, t_poc_flux_top, \
              conserving, darwin_arch)
    
phy_nh4upt = sms[-17] * 86400
phy_no2upt = sms[-16] * 86400
phy_no3upt = sms[-15] * 86400
phy_po4upt = sms[-14] * 86400
dia_nh4upt = sms[-13] * 86400
dia_no2upt = sms[-12] * 86400
dia_no3upt = sms[-11] * 86400
dia_po4upt = sms[-10] * 86400
poc_prod = sms[-9] * 86400
poc_remi = sms[-8] * 86400
ammox = sms[-7] * 86400 * 1e3
nitrox = sms[-6] * 86400 * 1e3
anammox = sms[-5] * 86400 * 1e3
denitrif1 = sms[-4] * 86400 * 1e3
denitrif2 = sms[-3] * 86400 * 1e3
denitrif3 = sms[-2] * 86400 * 1e3
denitrif4 = sms[-1] * 86400 * 1e3

# save the output to file
data = {"depth":z_zgrid,
        "O2":o2[1,:],
        "N2":n2[1,:],
        "NO3":no3[1,:],
        "NO2":no2[1,:],
        "NH4":nh4[1,:],
        "N2O":n2o[1,:],
        "PO4":po4[1,:],
        "DIC":dic[1,:],
        "DOC":doc[1,:],
        "DOP":dop[1,:],
        "DON":don[1,:],
        "rDOC":rdoc[1,:],
        "rDOP":rdop[1,:],
        "rDON":rdon[1,:],
        "PHY":phy[1,:],
        "DIA":dia[1,:],
        "AOA":aoa[1,:],
        "NOB":nob[1,:],
        "AOX":aox[1,:],
        "NAR":nar[1,:],
        "NAI":nai[1,:],
        "NIR":nir[1,:],
        "NOS":nos[1,:],
        "fNAR":fnar[1,:],
        "fNAI":fnai[1,:],
        "fNIR":fnir[1,:],
        "fNOS":fnos[1,:],
        "ZOO":zoo[1,:],
        "MES":mes[1,:],
        "PHY_NH4upt":phy_nh4upt,
        "PHY_NO2upt":phy_no2upt,
        "PHY_NO3upt":phy_no3upt,
        "PHY_PO4upt":phy_po4upt,
        "DIA_NH4upt":dia_nh4upt,
        "DIA_NO2upt":dia_no2upt,
        "DIA_NO3upt":dia_no3upt,
        "DIA_PO4upt":dia_po4upt,
        "POC_production":poc_prod,
        "POC_remineralisation":poc_remi,
        "ammox":ammox,
        "nitrox":nitrox,
        "anammox":anammox,
        "denitrif1":denitrif1,
        "denitrif2":denitrif2,
        "denitrif3":denitrif3,
        "denitrif4":denitrif4}

df = pd.DataFrame(data)

if darwin_arch:
    df.to_csv('output/darwin_arch.csv', header=True)
else:
    df.to_csv('output/default.csv', header=True)



#%% check mass balance between initial state and final state
#   (only works when poc_remi == 0.0, p_wup == 0.0, and p_Kv == 0.0)

tot_n0 = no3_init + no2_init + nh4_init + n2o_init*2 + n2_init*2 + don_init + rdon_init + \
         phy_init / p_pom_CN + dia_init / p_pom_CN + \
         aoa_init / p_aoa_CN + nob_init / p_nob_CN + aox_init / p_aox_CN + \
         nar_init / p_nar_CN + nai_init / p_nai_CN + nir_init / p_nir_CN + \
         nos_init / p_nos_CN + zoo_init / p_pom_CN + mes_init / p_pom_CN
tot_n1 = no3[1,:] + no2[1,:] + nh4[1,:] + n2o[1,:]*2 + n2[1,:]*2 + don[1,:] + rdon[1,:] + \
         phy[1,:] / p_pom_CN + dia[1,:] / p_pom_CN + \
         aoa[1,:] / p_aoa_CN + nob[1,:] / p_nob_CN + aox[1,:] / p_aox_CN + \
         nar[1,:] / p_nar_CN + nai[1,:] / p_nai_CN + nir[1,:] / p_nir_CN + \
         nos[1,:] / p_nos_CN + zoo[1,:] / p_pom_CN + mes[1,:] / p_pom_CN
    
print("total N before =",np.sum(tot_n0))
print("total N after =",np.sum(tot_n1))


#%% BGC tracers

col1 = 'firebrick'
col2 = 'k'

plt.figure(figsize=(14,8))
gs = GridSpec(3,4)

ax1 = plt.subplot(gs[0,0])
ax2 = plt.subplot(gs[0,1])
ax3 = plt.subplot(gs[0,2])
ax4 = plt.subplot(gs[0,3])
ax5 = plt.subplot(gs[1,0])
ax6 = plt.subplot(gs[1,1])
ax7 = plt.subplot(gs[1,2])
ax8 = plt.subplot(gs[1,3])
ax9 = plt.subplot(gs[2,0])
ax10 = plt.subplot(gs[2,1])
ax11 = plt.subplot(gs[2,2])
ax12 = plt.subplot(gs[2,3])

ax1.plot(o2[1,:], z_zgrid, color=col1, label='now')
ax1.plot(o2_init, z_zgrid, color=col2, label='initialised')
ax1.plot(o2_init*0.0, z_zgrid, 'k:')
ax1.legend()

ax2.plot(n2[1,:], z_zgrid, color=col1)
ax2.plot(n2_init, z_zgrid, color=col2)

ax3.plot(no3[1,:], z_zgrid, color=col1)
ax3.plot(no3_init, z_zgrid, color=col2)

ax4.plot(no2[1,:], z_zgrid, color=col1)
ax4.plot(no2_init, z_zgrid, color=col2)

ax5.plot(nh4[1,:], z_zgrid, color=col1)
ax5.plot(nh4_init, z_zgrid, color=col2)

ax6.plot(n2o[1,:]*1e3, z_zgrid, color=col1)
ax6.plot(n2o_init*1e3, z_zgrid, color=col2)

ax7.plot(po4[1,:], z_zgrid, color=col1)
ax7.plot(po4_init, z_zgrid, color=col2)

ax8.plot(dic[1,:], z_zgrid, color=col1)
ax8.plot(dic_init, z_zgrid, color=col2)

ax9.plot(doc[1,:], z_zgrid, color=col1)
ax9.plot(doc_init, z_zgrid, color=col2)
ax9.plot(rdoc[1,:], z_zgrid, color=col1, linestyle='--')
ax9.plot(rdoc_init, z_zgrid, color=col2, linestyle='--')

ax10.plot(doc[1,:]/dop[1,:], z_zgrid, color=col1)
ax10.plot(doc_init/dop_init, z_zgrid, color=col2)

ax11.plot(doc[1,:]/don[1,:], z_zgrid, color=col1)
ax11.plot(doc_init/don_init, z_zgrid, color=col2)

ax12.plot(phy[1,:], z_zgrid, color=col1)
ax12.plot(phy_init, z_zgrid, color=col2)
ax12.plot(dia[1,:], z_zgrid, color=col1, linestyle='--')
ax12.plot(dia_init, z_zgrid, color=col2, linestyle='--')
ax12.set_ylim(-200,0)

ax1.set_title('O2')
ax2.set_title('N2')
ax3.set_title('NO3')
ax4.set_title('NO2')
ax5.set_title('NH4')
ax6.set_title('N2O')
ax7.set_title('PO4')
ax8.set_title('DIC')
ax9.set_title('DOC')
ax10.set_title('DOC:DOP')
ax11.set_title('DOC:DON')
ax12.set_title('Phy & Diatoms')

plt.subplots_adjust(top=0.95, bottom=0.05, hspace=0.4)

#%% Biological tracers

col1 = 'k'
col2 = 'firebrick'
col3 = 'goldenrod'
col4 ='royalblue'

plt.figure(figsize=(14,8))
gs = GridSpec(2,4)

ax1 = plt.subplot(gs[0,0])
ax2 = plt.subplot(gs[0,1])
ax3 = plt.subplot(gs[0,2])
ax4 = plt.subplot(gs[0,3])
ax5 = plt.subplot(gs[1,0])
ax6 = plt.subplot(gs[1,1])
ax7 = plt.subplot(gs[1,2])
ax8 = plt.subplot(gs[1,3])

ax1.plot(o2[1,:], z_zgrid, color=col1, label='oxygen')
ax1.plot(o2_init*0.0, z_zgrid, 'k:')
ax1.legend(loc='upper center', bbox_to_anchor=(0.5,1.5))

ax2.plot(phy[1,:], z_zgrid, color=col1, label='Phy ($\mu$M C)')
ax2.plot(dia[1,:], z_zgrid, color=col2, label='Dia ($\mu$M C)')
ax2.plot(zoo[1,:], z_zgrid, color=col3, label='Zoo ($\mu$M C)')
ax2.plot(mes[1,:], z_zgrid, color=col4, label='Mes ($\mu$M C)')
ax2.legend(loc='upper center', bbox_to_anchor=(0.5,1.5))


chemos = aoa[1,:] + nob[1,:] + aox[1,:]
slowhets = nar[1,:] + nai[1,:] + nir[1,:] + nos[1,:]
fasthets = fnar[1,:] + fnai[1,:] + fnir[1,:] + fnos[1,:]

ax3.plot(chemos, z_zgrid, color=col1, label='chemoautotrophs')
ax3.plot(slowhets, z_zgrid, color=col2, label='slow heterotrophs')
ax3.plot(fasthets, z_zgrid, color=col3, label='fast heterotrophs')
ax3.legend(loc='upper center', bbox_to_anchor=(0.5,1.5))

f_aoa = aoa/chemos
f_nob = nob/chemos
f_aox = aox/chemos

ax4.plot(f_aoa[1,:], z_zgrid, color=col1, label='Fraction AOA')
ax4.plot(f_nob[1,:], z_zgrid, color=col2, label='Fraction NOB')
ax4.plot(f_aox[1,:], z_zgrid, color=col3, label='Fraction AOX')
ax4.legend(loc='upper center', bbox_to_anchor=(0.5,1.5))

f_nar = nar/slowhets
f_nai = nai/slowhets
f_nir = nir/slowhets
f_nos = nos/slowhets

ax5.plot(f_nar[1,:], z_zgrid, color=col1, label='Fraction NAR')
ax5.plot(f_nai[1,:], z_zgrid, color=col2, label='Fraction NAI')
ax5.plot(f_nir[1,:], z_zgrid, color=col3, label='Fraction NIR')
ax5.plot(f_nos[1,:], z_zgrid, color=col4, label='Fraction NOS')
ax5.legend(loc='lower center', bbox_to_anchor=(0.5,-0.65))

ax6.plot(poc_prod, z_zgrid, color=col1, label='POC production ($\mu$MC/day)')
ax6.plot(poc_remi, z_zgrid, color=col2, label='POC remineralisation ($\mu$MC/day)')
ax6.legend(loc='lower center', bbox_to_anchor=(0.5,-0.65))

ax7.plot(ammox, z_zgrid, color=col1, label='ammonia oxidation (nM/day)')
ax7.plot(nitrox, z_zgrid, color=col2, label='nitrite oxidation (nM/day)')
ax7.plot(anammox, z_zgrid, color=col3, label='anammox (nM/day)')
ax7.legend(loc='lower center', bbox_to_anchor=(0.5,-0.65))

ax8.plot(denitrif1, z_zgrid, color=col1, label='NO3 --> NO2 (nM/day)')
ax8.plot(denitrif2, z_zgrid, color=col2, label='NO3 --> N2O (nM/day)')
ax8.plot(denitrif3, z_zgrid, color=col3, label='NO2 --> N2O (nM/day)')
ax8.plot(denitrif4, z_zgrid, color=col4, label='N2O --> N2 (nM/day)')
ax8.legend(loc='lower center', bbox_to_anchor=(0.5,-0.65))


plt.subplots_adjust(bottom=0.2, top=0.8)

y1 = -500; y2 = 20
ax1.set_ylim(y1,y2)
ax2.set_ylim(y1,y2)
ax3.set_ylim(y1,y2)
ax4.set_ylim(y1,y2)
ax5.set_ylim(y1,y2)
ax6.set_ylim(y1,y2)
ax7.set_ylim(y1,y2)
ax8.set_ylim(y1,y2)


#%%


ax5.plot(nh4[1,:], z_zgrid, color=col1)
ax5.plot(nh4_init, z_zgrid, color=col2)

ax6.plot(n2o[1,:]*1e3, z_zgrid, color=col1)
ax6.plot(n2o_init*1e3, z_zgrid, color=col2)

ax7.plot(po4[1,:], z_zgrid, color=col1)
ax7.plot(po4_init, z_zgrid, color=col2)

ax8.plot(dic[1,:], z_zgrid, color=col1)
ax8.plot(dic_init, z_zgrid, color=col2)

ax9.plot(doc[1,:], z_zgrid, color=col1)
ax9.plot(doc_init, z_zgrid, color=col2)
ax9.plot(rdoc[1,:], z_zgrid, color=col1, linestyle='--')
ax9.plot(rdoc_init, z_zgrid, color=col2, linestyle='--')

ax10.plot(doc[1,:]/dop[1,:], z_zgrid, color=col1)
ax10.plot(doc_init/dop_init, z_zgrid, color=col2)

ax11.plot(doc[1,:]/don[1,:], z_zgrid, color=col1)
ax11.plot(doc_init/don_init, z_zgrid, color=col2)

ax12.plot(phy[1,:], z_zgrid, color=col1)
ax12.plot(phy_init, z_zgrid, color=col2)
ax12.plot(dia[1,:], z_zgrid, color=col1, linestyle='--')
ax12.plot(dia_init, z_zgrid, color=col2, linestyle='--')
ax12.set_ylim(-200,0)

ax1.set_title('O2')
ax2.set_title('N2')
ax3.set_title('NO3')
ax4.set_title('NO2')
ax5.set_title('NH4')
ax6.set_title('N2O')
ax7.set_title('PO4')
ax8.set_title('DIC')
ax9.set_title('DOC')
ax10.set_title('DOC:DOP')
ax11.set_title('DOC:DON')
ax12.set_title('Phy & Diatoms')


#%% make figure for paper?

name = "Disconnected_ecosystem"

cols = ['k', 'grey', 'firebrick', 'goldenrod', 'royalblue']
lins = ['-', '--']

fig = plt.figure(figsize=(14,8))
gs = GridSpec(1,4)

ax1 = plt.subplot(gs[0,0])
ax2 = plt.subplot(gs[0,1])
ax3 = plt.subplot(gs[0,2])
ax4 = plt.subplot(gs[0,3])
ax3t = ax3.twiny()

slow_hetero = nar + nai + nir + nos
fast_hetero = fnar + fnai + fnir + fnos
chemos = aoa + nob + aox
phytos = phy + dia
zoopls = mes + zoo

ax1.plot(phytos[1,:], z_zgrid, color=cols[0], linestyle=lins[0], label='Phytos')
ax1.plot(zoopls[1,:], z_zgrid, color=cols[1], linestyle=lins[0], label='Zoos')
ax1.plot(o2_init*0.0, z_zgrid, 'k:')
ax1.legend()

ax2.plot(chemos[1,:], z_zgrid, color=cols[0], linestyle=lins[0], label='Chemos')
ax2.plot(slow_hetero[1,:], z_zgrid, color=cols[1], linestyle=lins[0], label='Slow Heteros')
ax2.plot(fast_hetero[1,:], z_zgrid, color=cols[2], linestyle=lins[0], label='Fast Heteros')
ax2.plot(o2_init*0.0, z_zgrid, 'k:')
ax2.legend()

ax3.plot(nh4[1,:], z_zgrid, color=cols[0], linestyle=lins[0], label='NH$_4$')
ax3.plot(no2[1,:], z_zgrid, color=cols[1], linestyle=lins[0], label='NO$_2$')
ax3t.plot(o2[1,:], z_zgrid, color=cols[2], linestyle=lins[0], label='O$_2$')
ax3.legend(loc='lower right')
ax3t.legend(loc='lower left')

ax4.plot(ammox, z_zgrid, label='NH$_4$ --> NO$_2$')
ax4.plot(nitrox, z_zgrid, label='NO$_2$ --> NO$_3$')
ax4.plot(anammox, z_zgrid, label='NH$_4$ + NO$_2$ --> N$_2$')
ax4.plot(denitrif1, z_zgrid, label='NO$_3$ --> NO$_2$')
ax4.plot(denitrif2, z_zgrid, label='NO$_3$ --> N$_2$O')
ax4.plot(denitrif3, z_zgrid, label='NO$_2$ --> N$_2$O')
ax4.plot(denitrif4, z_zgrid, label='N$_2$O --> N$_2$')
ax4.legend(loc='lower center')


y1 = -800
y2 = 5
ax1.set_ylim(y1,y2)
ax2.set_ylim(y1,y2)
ax3.set_ylim(y1,y2)
ax4.set_ylim(y1,y2)
ax3t.set_ylim(y1,y2)

ax3t.tick_params(color=cols[2], labelcolor=cols[2])

#%%
fig.savefig(name+".png")

