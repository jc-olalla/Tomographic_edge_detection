#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 13:33:45 2021

@author: jchavezolalla
"""

import sys
sys.path.append('/home/jfchavezolalla/Desktop/01-All-risk-results-log/00_toolbox')
import laplace_edge as lapedg
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


# Input
RHO_filename = 'RHO_grid_toe.txt'
X_filename = 'X_grid_toe.txt'
Z_filename = 'Z_grid_toe.txt'
RHO_grid = np.loadtxt(RHO_filename)
X_grid = np.loadtxt(X_filename)
Z_grid = np.loadtxt(Z_filename)

# Filter tomography
max_fh_bin = np.inf  # 100  # 3 std away
sigma_h = np.shape(RHO_grid)[1] / (2 * np.pi * (max_fh_bin / 3))
sigma_v = 1.0 * sigma_h
print('ERT sigma_h=', sigma_h)
print('ERT sigma_v=', sigma_v)
blur = gaussian_filter(RHO_grid, [sigma_v, sigma_h])  # [v, h]  gaussian filter


# Laplacian edge detection
plotmin = 5
plotmax = 40
filename_out = 'orientations_toe.csv'
# tomo_lap = lapedg.tomo_laplace(X_grid, Z_grid, blur)
tomo_lap = lapedg.tomo_laplace(X_grid, Z_grid, RHO_grid)
tomo_lap.find_all_interfaces_with_threshold(T=None, nmin=4)
tomo_lap.pick_interfaces([2], form_list=['cover'])  #  , xmin=[10], xmax=[40.0])  # , 120.0])
tomo_lap.plot_selected_interfaces()

# Plot tomography
fig = plt.figure()
ax = fig.add_subplot(111)
ax.contourf(X_grid, Z_grid, RHO_grid, cmap='jet', vmin=plotmin, vmax=plotmax, levels=100)
ax.set_xlabel('Distance (m)')
ax.set_ylabel('Depth (m)')

plt.show()