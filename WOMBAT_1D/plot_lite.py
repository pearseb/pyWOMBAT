#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 15:31:52 2023

Purpose
-------
    plot output of WOMBAT-lite 1D every so often

@author: buc146
"""

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def plot0D(o2,no3,dfe,phy,zoo,det,cal,alk,dic,chlc,phyfec,zoofec,detfec):
    
    col1 = 'k'
    col2 = 'firebrick'
    col3 = 'goldenrod'
    
    lab1 = 'Oxygen (µM)'
    lab2 = 'Nitrate (µM)'
    lab3 = 'dissolved Fe (nM)'
    lab4 = 'Phytoplankton (µM C)'
    lab5 = 'Zooplankton (µM C)'
    lab6 = 'Detritus (µM C)'
    lab7 = 'CaCO$_3$ (µM)'
    lab8 = 'Alkalinity (µM Eq)'
    lab9 = 'DIC (µM)'
    lab10 ='Chl:C'
    lab11 ='Phy Fe:C (µmol mol$^{-1}$)'
    lab12 ='Zoo Fe:C (µmol mol$^{-1}$)'
    lab13 ='Det Fe:C (µmol mol$^{-1}$)'
    
    fig = plt.figure(figsize=(14,8))
    gs = GridSpec(2,3)
    
    ax1 = plt.subplot(gs[0,0]) # O2
    ax2 = plt.subplot(gs[0,1]) # NO3 + dFe
    ax3 = plt.subplot(gs[0,2]) # Phy + Zoo + Det
    ax4 = plt.subplot(gs[1,0]) # CaCO3
    ax5 = plt.subplot(gs[1,1]) # Alk + DIC
    ax6 = plt.subplot(gs[1,2]) # chl:C ratio of PHY
    
    ax1.plot(no3, color=col1, label=lab2)
    ax1.plot(dfe*1e3, color=col2, label=lab3)
    ax1.legend()
    
    ax2.plot(phy, color=col1, label=lab4)
    ax2.plot(zoo, color=col2, label=lab5)
    ax2.plot(chl2c*12*phy, color=col3)
    ax2.legend()
    
    ax3.plot(det, color=col1, label=lab6)
    ax3.plot(cal, color=col2, label=lab7)
    ax3.legend(loc='lower left')
    
    ax4.plot(alk, color=col1, label=lab8)
    ax4.plot(dic, color=col2, label=lab9)
    ax4.legend()
    
    ax5.plot(chlc, color=col1, label=lab10)
    ax5.legend()
    
    ax6.plot(phyfec, color=col1, label=lab11)
    ax6.plot(zoofec, color=col2, label=lab12)
    ax6.plot(detfec, color=col3, label=lab13)
    ax6.legend()
    
    return fig


def plot1D(o2,no3,dfe,phy,zoo,det,cal,phymu,zoomu,chlc,phyfec,zoofec,detfec,z_zgrid):
    
    col1 = 'k'
    col2 = 'firebrick'
    col3 = 'goldenrod'
    
    lab1 = 'Oxygen (µM)'
    lab2 = 'Nitrate (µM)'
    lab3 = 'dissolved Fe (nM)'
    lab4 = 'Phytoplankton (µM C)'
    lab5 = 'Zooplankton (µM C)'
    lab6 = 'Detritus (µM C)'
    lab7 = 'CaCO$_3$ (µM)'
    lab8 = '$\mu$(Phy) (day$^{-1}$)'
    lab9 = '$\mu$(Zoo) (day$^{-1}$)'
    lab10 ='Chl:C'
    lab11 ='Phy Fe:C (µmol mol$^{-1}$)'
    lab12 ='Zoo Fe:C (µmol mol$^{-1}$)'
    lab13 ='Det Fe:C (µmol mol$^{-1}$)'
    
    fig = plt.figure(figsize=(14,8))
    gs = GridSpec(2,3)
    
    ax1 = plt.subplot(gs[0,0]) # O2
    ax2 = plt.subplot(gs[0,1]) # NO3 + dFe
    ax3 = plt.subplot(gs[0,2]) # Phy + Zoo + Det
    ax4 = plt.subplot(gs[1,0]) # CaCO3
    ax5 = plt.subplot(gs[1,1]) # growth rates
    ax6 = plt.subplot(gs[1,2]) # chl:C ratio of PHY
    
    ax1.plot(no3, z_zgrid, color=col1, label=lab2)
    ax1.plot(dfe*1e3, z_zgrid, color=col2, label=lab3)
    ax1.legend()
    
    ax2.plot(phy, z_zgrid, color=col1, label=lab4)
    ax2.plot(zoo, z_zgrid, color=col2, label=lab5)
    ax2.plot(chlc*12*phy, z_zgrid, color=col3, label='Chlorophyll (mg m$^{-3}$)')
    ax2.legend()
    
    ax3.plot(det, z_zgrid, color=col1, label=lab6)
    ax3.plot(cal, z_zgrid, color=col2, label=lab7)
    ax3.legend(loc='lower left')
    
    ax4.plot(phymu*86400, z_zgrid, color=col1, label=lab8)
    ax4.plot(zoomu*86400, z_zgrid, color=col2, label=lab9)
    ax4.legend()
    
    ax5.plot(chlc, z_zgrid, color=col1, label=lab10)
    ax5.legend()
    
    ax6.plot(phyfec, z_zgrid, color=col1, label=lab11)
    ax6.plot(zoofec, z_zgrid, color=col2, label=lab12)
    ax6.plot(detfec, z_zgrid, color=col3, label=lab13)
    ax6.legend()
    
    ax1.set_ylim(-500,0)
    ax2.set_ylim(-500,0)
    ax3.set_ylim(-500,0)
    ax4.set_ylim(-500,0)
    ax5.set_ylim(-500,0)
    ax6.set_ylim(-500,0)
    ax6.set_xlim(0,200)
    
    return fig

