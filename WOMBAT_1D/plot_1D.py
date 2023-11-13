#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 15:31:52 2023

Purpose
-------
    plot output of 1D model every so often

@author: buc146
"""

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def plot_1D(o2,no3,dfe,phy,zoo,det,cal,alk,dic,z_zgrid):
    
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
    
    fig = plt.figure(figsize=(14,8))
    gs = GridSpec(2,3)
    
    ax1 = plt.subplot(gs[0,0]) # O2
    ax2 = plt.subplot(gs[0,1]) # NO3 + dFe
    ax3 = plt.subplot(gs[0,2]) # Phy + Zoo + Det
    ax4 = plt.subplot(gs[1,0]) # CaCO3
    ax5 = plt.subplot(gs[1,1]) # Alk + DIC
    ax6 = plt.subplot(gs[1,2]) # Alk + DIC
    
    ax1.plot(o2, z_zgrid, color=col1, label=lab1)
    ax1.legend()
    
    ax2.plot(no3, z_zgrid, color=col1, label=lab2)
    ax2.plot(dfe*1e3, z_zgrid, color=col2, label=lab3)
    ax2.legend()
    
    ax3.plot(phy, z_zgrid, color=col1, label=lab4)
    ax3.plot(zoo, z_zgrid, color=col2, label=lab5)
    ax3.legend()
    
    ax4.plot(det, z_zgrid, color=col1, label=lab6)
    ax4.plot(cal, z_zgrid, color=col2, label=lab7)
    ax4.legend(loc='lower left')
    
    ax5.plot(alk, z_zgrid, color=col1, label=lab8)
    ax5.legend()
    
    ax6.plot(dic, z_zgrid, color=col2, label=lab9)
    ax6.legend()
    
    return fig


def plot_1D_chl(o2,no3,dfe,phy,zoo,det,cal,alk,dic,chl,z_zgrid):
    
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
    lab10 ='Chl:C'
    
    fig = plt.figure(figsize=(14,8))
    gs = GridSpec(2,3)
    
    ax1 = plt.subplot(gs[0,0]) # O2
    ax2 = plt.subplot(gs[0,1]) # NO3 + dFe
    ax3 = plt.subplot(gs[0,2]) # Phy + Zoo + Det
    ax4 = plt.subplot(gs[1,0]) # CaCO3
    ax5 = plt.subplot(gs[1,1]) # Alk + DIC
    ax6 = plt.subplot(gs[1,2]) # chl:C ratio of PHY
    
    ax1.plot(o2, z_zgrid, color=col1, label=lab1)
    ax1.legend()
    
    ax2.plot(no3, z_zgrid, color=col1, label=lab2)
    ax2.plot(dfe*1e3, z_zgrid, color=col2, label=lab3)
    ax2.legend()
    
    ax3.plot(phy, z_zgrid, color=col1, label=lab4)
    ax3.plot(zoo, z_zgrid, color=col2, label=lab5)
    ax3.legend()
    
    ax4.plot(det, z_zgrid, color=col1, label=lab6)
    ax4.plot(cal, z_zgrid, color=col2, label=lab7)
    ax4.legend(loc='lower left')
    
    ax5.plot(alk, z_zgrid, color=col1, label=lab8)
    ax5.plot(dic, z_zgrid, color=col2, label=lab9)
    ax5.legend()
    
    ax6.plot(chl, z_zgrid, color=col1, label=lab10)
    ax6.legend()
    
    return fig


def plot_1D_2P2Z_chl(o2,no3,dfe,phy,dia,zoo,mes,det,cal,pchl,dchl,pgi_zoo,pgi_mes,z_zgrid):
    
    col1 = 'k'
    col2 = 'firebrick'
    col3 = 'goldenrod'
    
    lst1 = '-'
    lst2 = '--'
    
    lab1 = 'Oxygen (µM)'
    lab2 = 'Nitrate (µM)'
    lab3 = 'dissolved Fe (nM)'
    lab4 = 'Nano-Phytoplankton (µM C)'
    lab5 = 'Diatoms (µM C)'
    lab6 = 'Micro-zooplankton (µM C)'
    lab7 = 'Meso-zooplankton (µM C)'
    lab8 = 'Detritus (µM C)'
    lab9 = 'CaCO$_3$ (µM)'
    lab10 ='phy Chl:C'
    lab11 ='dia Chl:C'
    lab12 = 'PGI Zoo (µM Z (µM prey day)$^{-1}$)'
    lab13 = 'PGI Mes (µM Z (µM prey day)$^{-1}$)'
    
    fig = plt.figure(figsize=(14,8))
    gs = GridSpec(2,3)
    
    ax1 = plt.subplot(gs[0,0]) # O2
    ax2 = plt.subplot(gs[0,1]) # NO3 + dFe
    ax3 = plt.subplot(gs[0,2]) # Phy + Zoo + Det
    ax4 = plt.subplot(gs[1,0]) # CaCO3
    ax5 = plt.subplot(gs[1,1]) # Alk + DIC
    ax6 = plt.subplot(gs[1,2]) # chl:C ratio of PHY
    
    ax1.plot(o2, z_zgrid, color=col1, label=lab1)
    ax1.legend()
    
    ax2.plot(no3, z_zgrid, color=col1, label=lab2)
    ax2.plot(dfe*1e3, z_zgrid, color=col2, label=lab3)
    ax2.legend()
    
    ax3.plot(phy, z_zgrid, color=col1, label=lab4)
    ax3.plot(dia, z_zgrid, color=col1, label=lab5, linestyle=lst2)
    ax3.plot(zoo, z_zgrid, color=col2, label=lab6)
    ax3.plot(mes, z_zgrid, color=col2, label=lab7, linestyle=lst2)
    ax3.legend()
    
    ax4.plot(det, z_zgrid, color=col1, label=lab8)
    ax4.plot(cal, z_zgrid, color=col2, label=lab9)
    ax4.legend(loc='lower left')
    
    ax5.plot(pchl, z_zgrid, color=col1, label=lab10)
    ax5.plot(dchl, z_zgrid, color=col1, label=lab11, linestyle=lst2)
    ax5.set_xlim(0,0.05)
    ax5.legend()
    
    ax6.plot(pgi_zoo, z_zgrid, color=col2, label=lab12)
    ax6.plot(pgi_mes, z_zgrid, color=col2, label=lab13, linestyle=lst2)
    ax6.legend()
    
    ax1.set_ylim(-400,5)
    ax2.set_ylim(-400,5)
    ax3.set_ylim(-400,5)
    ax4.set_ylim(-400,5)
    ax5.set_ylim(-400,5)
    ax6.set_ylim(-400,5)
    
    return fig


def plot_1D_2P2Z_varFe(o2,no3,dfe,phy,dia,zoo,mes,det,cal,\
                       pchl,dchl,phyfe,diafe,detfe,zoofe,mesfe,pgi_zoo,pgi_mes,z_zgrid):
    
    col1 = 'k'
    col2 = 'firebrick'
    col3 = 'goldenrod'
    
    lst1 = '-'
    lst2 = '--'
    
    lab1 = 'Oxygen (µM)'
    lab2 = 'Nitrate (µM)'
    lab3 = 'dissolved Fe (nM)'
    lab4 = 'Nano-Phytoplankton (µM C)'
    lab5 = 'Diatoms (µM C)'
    lab6 = 'Micro-zooplankton (µM C)'
    lab7 = 'Meso-zooplankton (µM C)'
    lab8 = 'Detritus (µM C)'
    lab9 = 'phy Chl:C'
    lab10 ='dia Chl:C'
    lab11 ='phy Fe:C'
    lab12 ='dia Fe:C'
    lab13 ='zoo Fe:C'
    lab14 ='mes Fe:C'
    lab15 ='det Fe:C'
    lab16 ='zoo PGI'
    lab17 ='mes PGI'
    
    fig = plt.figure(figsize=(14,8))
    gs = GridSpec(2,3)
    
    ax1 = plt.subplot(gs[0,0]) # O2
    ax2 = plt.subplot(gs[0,1]) # NO3 + dFe
    ax3 = plt.subplot(gs[0,2]) # Phy + Zoo + Det
    ax4 = plt.subplot(gs[1,0]) # chl:C ratio of PHY
    ax5 = plt.subplot(gs[1,1]) # Fe:C ratios
    ax6 = plt.subplot(gs[1,2]) # grazing pressure
    
    ax1.plot(o2, z_zgrid, color=col1, label=lab1)
    ax1.legend()
    
    ax2.plot(no3, z_zgrid, color=col1, label=lab2)
    ax2.plot(dfe*1e3, z_zgrid, color=col2, label=lab3)
    ax2.legend()
    
    ax3.plot(phy, z_zgrid, color=col1, label=lab4)
    ax3.plot(dia, z_zgrid, color=col1, label=lab5, linestyle=lst2)
    ax3.plot(zoo, z_zgrid, color=col2, label=lab6)
    ax3.plot(mes, z_zgrid, color=col2, label=lab7, linestyle=lst2)
    ax3.plot(det, z_zgrid, color=col3, label=lab8)
    ax3.legend()
    
    ax4.plot(pchl, z_zgrid, color=col1, label=lab9)
    ax4.plot(dchl, z_zgrid, color=col1, label=lab10, linestyle=lst2)
    ax4.set_xlim(0,0.05)
    ax4.legend()
    
    ax5.plot(phyfe, z_zgrid, color=col1, label=lab11)
    ax5.plot(diafe, z_zgrid, color=col1, label=lab12, linestyle=lst2)
    ax5.plot(zoofe, z_zgrid, color=col2, label=lab13)
    ax5.plot(mesfe, z_zgrid, color=col2, label=lab14, linestyle=lst2)
    ax5.plot(detfe, z_zgrid, color=col3, label=lab15)
    ax5.set_xlim(0,80)
    ax5.legend()
    
    ax6.plot(pgi_zoo, z_zgrid, color=col2, label=lab16)
    ax6.plot(pgi_mes, z_zgrid, color=col2, label=lab17, linestyle=lst2)
    ax6.legend()
    
    ax1.set_ylim(-400,5)
    ax2.set_ylim(-400,5)
    ax3.set_ylim(-400,5)
    ax4.set_ylim(-400,5)
    ax5.set_ylim(-400,5)
    ax6.set_ylim(-400,5)
    
    return fig