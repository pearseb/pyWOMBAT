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