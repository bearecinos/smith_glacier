"""
Plot inversion output

- Reads the input mesh
- Reads output data (this can be any of the output
 from the inversion if it is stored as .xml)
- Plots things in a multiplot grid

@authors: Fenics_ice contributors
"""
import sys
import os
import salem
from salem import DataLevels
import h5py
import numpy as np
import xarray as xr
import pyproj

from fenics import *
from fenics_ice import config as conf
from pathlib import Path
from configobj import ConfigObj
import argparse

# Matplotlib imports
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec

# Load configuration file for more order in paths
parser = argparse.ArgumentParser()
parser.add_argument("-conf", type=str, default="../../../config.ini", help="pass config file")
parser.add_argument("-toml_path_i", type=str,
                    default="../run_experiments/run_workflow/data_products_exp_tomls/smith_cloud_itslive.toml",
                    help="pass .toml file")
parser.add_argument("-toml_path_m", type=str,
                    default="../run_experiments/run_workflow/data_products_exp_tomls/smith_cloud_measures.toml",
                    help="pass .toml file")
parser.add_argument("-sub_plot_dir", type=str, default="temp", help="pass sub plot directory to store the plots")
args = parser.parse_args()
config_file = args.conf
configuration = ConfigObj(os.path.expanduser(config_file))

# Define main repository path
MAIN_PATH = configuration['main_path']
sys.path.append(MAIN_PATH)

from ficetools import backend, utils_funcs, graphics, velocity

rcParams['axes.labelsize'] = 18
rcParams['xtick.labelsize'] = 18
rcParams['ytick.labelsize'] = 18
rcParams['axes.titlesize'] = 20


tick_options = {'axis':'both','which':'both','bottom':False,
     'top':False,'left':False,'right':False,'labelleft':False, 'labelbottom':False}

# Paths to data
sub_plot_dir = args.sub_plot_dir
plot_path = os.path.join(MAIN_PATH, 'plots/'+ sub_plot_dir)
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

tomlf_i = args.toml_path_i
params_il = conf.ConfigParser(tomlf_i)

tomlf_m = args.toml_path_m
params_me = conf.ConfigParser(tomlf_m)

vel_file_itslive = Path(params_il.io.input_dir) / params_il.obs.vel_file
vel_file_measures = Path(params_me.io.input_dir) / params_me.obs.vel_file

f_vel_trn = h5py.File(vel_file_itslive, 'r')
f_vel_test = h5py.File(vel_file_measures, 'r')

x_trn = f_vel_trn['x_cloud'][:]
y_trn = f_vel_trn['y_cloud'][:]
u_obs = f_vel_trn['u_cloud'][:]
v_obs = f_vel_trn['v_cloud'][:]

vv_trn = (u_obs**2 + v_obs**2)**0.5

u_obs_std = f_vel_trn['u_cloud_std'][:]
v_obs_std = f_vel_trn['v_cloud_std'][:]

x_val = f_vel_test['x_cloud'][:]
y_val = f_vel_test['y_cloud'][:]
u_obs_val = f_vel_test['u_cloud'][:]
v_obs_val = f_vel_test['v_cloud'][:]

vv_val = (u_obs_val**2 + v_obs_val**2)**0.5

u_obs_std_val = f_vel_test['u_cloud_std'][:]
v_obs_std_val = f_vel_test['v_cloud_std'][:]


vel_obs =os.path.join(MAIN_PATH, configuration['measures_cloud'])
dv = xr.open_dataset(vel_obs)

smith_bbox = {'xmin': -1609000.0,
              'xmax': -1381000.0,
              'ymin': -718450.0,
              'ymax': -527000.0}

vx = dv.VX
vy = dv.VY
std_vx = dv.STDX
std_vy = dv.STDY

# Crop velocity data to the Smith Glacier extend
vx_s = velocity.crop_velocity_data_to_extend(vx, smith_bbox, return_xarray=True)
vy_s = velocity.crop_velocity_data_to_extend(vy, smith_bbox, return_xarray=True)
std_vx_s = velocity.crop_velocity_data_to_extend(std_vx, smith_bbox, return_xarray=True)
std_vy_s = velocity.crop_velocity_data_to_extend(std_vy, smith_bbox,return_xarray=True)

vv = (vx_s**2 + vy_s**2)**0.5
std = (std_vx_s**2 + std_vy_s**2)**0.5

# Lets define our salem grid. (but we modified things cuz fabi's code only works for the North! TODO: ask fabien)

proj = pyproj.Proj('EPSG:3413')
y_grid = vx_s.y
x_grid = vx_s.x

dy = abs(y_grid[0] - y_grid[1])
dx = abs(x_grid[0] - x_grid[1])

# Pixel corner
origin_y = y_grid[0] + dy * 0.5
origin_x = x_grid[0] - dx * 0.5

gv = salem.Grid(nxny=(len(x_grid), len(y_grid)), dxdy=(dx, -1*dy), # We use -dy as this is the Sout Hemisphere somehow salem
                   x0y0=(origin_x, origin_y), proj=proj) # is not picking that up!

# Now plotting
r = 2.0

fig1 = plt.figure(figsize=(10*r, 6*r))#, constrained_layout=True)
spec = gridspec.GridSpec(2, 3, hspace=0.3, wspace=0.1)

ax0 = plt.subplot(spec[0])
ax0.set_aspect('equal')
smap = salem.Map(gv, countries=False)
x_n, y_n = smap.grid.transform(x_trn, y_trn,
                              crs=gv.proj)

dl = DataLevels(vv_trn, levels=np.arange(0, 1000, 10), extend='both')
c = ax0.scatter(x_n, y_n, c=dl.to_rgb(), s=0.5, cmap='viridis')
smap.set_lonlat_contours(xinterval=1.0, yinterval=0.5, add_tick_labels=False, linewidths=1.5, alpha=0.3)
out = graphics.get_projection_grid_labels(smap)
smap.xtick_pos = out[0]
smap.xtick_val = out[1]
smap.ytick_pos = out[2]
smap.ytick_val = out[3]
smap.visualize(ax=ax0, orientation='horizontal', addcbar=False)
dl.append_colorbar(ax0, position='bottom', size="5%", pad=0.5, label='velocity [m. $yr^{-1}$]')
at = AnchoredText('a', prop=dict(size=18), frameon=True, loc='upper left')
ax0.add_artist(at)

ax1 = plt.subplot(spec[1])
ax1.set_aspect('equal')
smap = salem.Map(gv, countries=False)
x_n, y_n = smap.grid.transform(x_trn, y_trn,
                              crs=gv.proj)

dl = DataLevels(u_obs_std, levels=np.arange(0, 100, 1), extend='both')
c = ax1.scatter(x_n, y_n, c=dl.to_rgb(), s=0.5, cmap='viridis')
smap.set_lonlat_contours(xinterval=1.0, yinterval=0.5, add_tick_labels=False, linewidths=1.5, alpha=0.3)
out = graphics.get_projection_grid_labels(smap)
smap.xtick_pos = out[0]
smap.xtick_val = out[1]
smap.ytick_pos = out[2]
smap.ytick_val = out[3]
smap.visualize(ax=ax1, orientation='horizontal', addcbar=False)
dl.append_colorbar(ax1, position='bottom', size="5%", pad=0.5, label='velocity VX STD [m. $yr^{-1}$]')
at = AnchoredText('b', prop=dict(size=18), frameon=True, loc='upper left')
ax1.add_artist(at)

ax2 = plt.subplot(spec[2])
ax2.set_aspect('equal')
smap = salem.Map(gv, countries=False)
x_n, y_n = smap.grid.transform(x_trn, y_trn,
                              crs=gv.proj)

dl = DataLevels(v_obs_std, levels=np.arange(0, 100, 1), extend='both')
c = ax2.scatter(x_n, y_n, c=dl.to_rgb(), s=0.5, cmap='viridis')
smap.set_lonlat_contours(xinterval=1.0, yinterval=0.5, add_tick_labels=False, linewidths=1.5, alpha=0.3)
out = graphics.get_projection_grid_labels(smap)
smap.xtick_pos = out[0]
smap.xtick_val = out[1]
smap.ytick_pos = out[2]
smap.ytick_val = out[3]
smap.visualize(ax=ax2, orientation='horizontal', addcbar=False)
dl.append_colorbar(ax2, position='bottom', size="5%", pad=0.5, label='velocity VY STD [m. $yr^{-1}$]')
at = AnchoredText('c', prop=dict(size=18), frameon=True, loc='upper left')
ax2.add_artist(at)


ax3 = plt.subplot(spec[3])
ax3.set_aspect('equal')
smap = salem.Map(gv, countries=False)
x_n, y_n = smap.grid.transform(x_val, y_val,
                              crs=gv.proj)

dl = DataLevels(vv_val, levels=np.arange(0, 1000, 10), extend='both')
c = ax3.scatter(x_n, y_n, c=dl.to_rgb(), s=0.5, cmap='viridis')
smap.set_lonlat_contours(xinterval=1.0, yinterval=0.5, add_tick_labels=False, linewidths=1.5, alpha=0.3)
out = graphics.get_projection_grid_labels(smap)
smap.xtick_pos = out[0]
smap.xtick_val = out[1]
smap.ytick_pos = out[2]
smap.ytick_val = out[3]
smap.visualize(ax=ax3, orientation='horizontal', addcbar=False)
dl.append_colorbar(ax3, position='bottom', size="5%", pad=0.5, label='velocity [m. $yr^{-1}$]')
at = AnchoredText('d', prop=dict(size=18), frameon=True, loc='upper left')
ax3.add_artist(at)


ax4 = plt.subplot(spec[4])
ax4.set_aspect('equal')
smap = salem.Map(gv, countries=False)
x_n, y_n = smap.grid.transform(x_val, y_val,
                              crs=gv.proj)

dl = DataLevels(u_obs_std_val, levels=np.arange(0, 100, 1), extend='both')
c = ax4.scatter(x_n, y_n, c=dl.to_rgb(), s=0.5, cmap='viridis')
smap.set_lonlat_contours(xinterval=1.0, yinterval=0.5, add_tick_labels=False, linewidths=1.5, alpha=0.3)
out = graphics.get_projection_grid_labels(smap)
smap.xtick_pos = out[0]
smap.xtick_val = out[1]
smap.ytick_pos = out[2]
smap.ytick_val = out[3]
smap.visualize(ax=ax4, orientation='horizontal', addcbar=False)
dl.append_colorbar(ax4, position='bottom', size="5%", pad=0.5, label='velocity VX STD [m. $yr^{-1}$]')
at = AnchoredText('e', prop=dict(size=18), frameon=True, loc='upper left')
ax4.add_artist(at)

ax5 = plt.subplot(spec[5])
ax5.set_aspect('equal')
smap = salem.Map(gv, countries=False)
x_n, y_n = smap.grid.transform(x_val, y_val,
                              crs=gv.proj)

dl = DataLevels(v_obs_std_val, levels=np.arange(0, 100, 1), extend='both')
c = ax5.scatter(x_n, y_n, c=dl.to_rgb(), s=0.5, cmap='viridis')
smap.set_lonlat_contours(xinterval=1.0, yinterval=0.5, add_tick_labels=False, linewidths=1.5, alpha=0.3)
out = graphics.get_projection_grid_labels(smap)
smap.xtick_pos = out[0]
smap.xtick_val = out[1]
smap.ytick_pos = out[2]
smap.ytick_val = out[3]
smap.visualize(ax=ax5, orientation='horizontal', addcbar=False)
dl.append_colorbar(ax5, position='bottom', size="5%", pad=0.5, label='velocity VY STD [m. $yr^{-1}$]')
at = AnchoredText('f', prop=dict(size=18), frameon=True, loc='upper left')
ax5.add_artist(at)

ax0.set_title('Retaining only 1.6% of ITS_LIVE data', fontdict={'fontsize': 20})
ax1.set_title('VX STD ITS_LIVE adjusted', fontdict={'fontsize': 20})
ax2.set_title('VY STD ITS_LIVE adjusted', fontdict={'fontsize': 20})

ax3.set_title('Validation set MEaSUREs', fontdict={'fontsize': 20})
#ax4.set_title('Validation set MEaSUREs', fontdict={'fontsize': 20})
#ax5.set_title('Validation set MEaSUREs', fontdict={'fontsize': 20})

plt.tight_layout()
plt.savefig(os.path.join(plot_path, 'input_vel_exp.png'),
            bbox_inches='tight', dpi=150)
