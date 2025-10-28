# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 11:25:23 2025

@author: 杨艺楠
"""
import numpy as np
import netCDF4 as nc
from cartopy.feature import ShapelyFeature
from matplotlib.collections import LineCollection
from matplotlib.ticker import AutoMinorLocator, FixedLocator, FixedFormatter
import matplotlib.pyplot as plt 
import cartopy.crs as ccrs

import sys
sys.path.append(r'D:/Function/to/your/path')  
from Fun1_threshold import load_wave_ray_data
wn_min = 1; wn_max = 5 # 所需波数的最大，最小值

path = 'D:/data/to/your/path.nc'
nc = nc.Dataset(path)
rlon, rlat, rzwn, rmwn = load_wave_ray_data(path)

fig, ax = plt.subplots(figsize=(14,10), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},dpi=400)
ax.coastlines(linewidth=0.8,color = 'grey',zorder=14)
min_longitude, max_longitude = -180, 180
min_latitude, max_latitude = -90, 90
ax.set_xlim(min_longitude, max_longitude);ax.set_ylim(min_latitude, max_latitude)
longitude = range(min_longitude, max_longitude + 1, 30)
latitude = range(min_latitude, max_latitude + 1, 10)
plt.xticks(longitude,fontsize=19); plt.yticks(latitude,fontsize=19)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.tick_params(which='minor', length=3.5, color='k',direction='out')  
plt.gca().set_xticklabels(['0°' if lon in [-180, 180] else '180°' if lon == 0 else f'{lon+180:.0f}°E' if lon < 0 else f'{abs(180 - lon):.0f}°W' for lon in longitude])
plt.gca().set_yticklabels([f'{lat11:.0f}°N' if lat11 > 0 else (f'{abs(lat11):.0f}°S' if lat11 < 0 else '0°') for lat11 in latitude])

# plot ===================================================================
colors = ['#5DC1B9', '#76BDF2', '#A6E3A1', '#FFD6A5', '#FFB5A7']
for loc in range(rlat.shape[2]):
    color = colors[loc % len(colors)]  
    for wn in range(wn_min - 1, wn_max):
        for j in range(3):
            x = rlon[:, j, loc, wn]
            y = rlat[:, j, loc, wn]
            ax.plot(x, y, linewidth=1.4, c=color,transform=ccrs.PlateCarree()) # ,transform=ccrs.PlateCarree()

ax.set_title('the Great Circle',loc='left', fontsize = 26)
lat_scatter=np.arange(0,21,5);lon_scatter=[1]
# scatter the sources
for j in range(0,len(lat_scatter)):
    for k in range(0,len(lon_scatter)):
        ax.scatter(lon_scatter[k],lat_scatter[j],c='#1E90FF',s=15,transform=ccrs.PlateCarree(),zorder=15)
plt.show()