"""
Plots mesh and Gridded data used as input
for Fenics_ice

- Reads the input mesh
- Reads input data
- Plots everything (well the most interesting data)
 in a multiple grid

@authors: Fenics_ice contributors
"""
import sys
import os
import salem
import h5py
import numpy as np
import pandas as pd
import xarray as xr
from fenics import *
from fenics_ice import inout
from fenics_ice import config as conf
from pathlib import Path
from fenics_ice import mesh as fice_mesh

#Plotting imports
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import matplotlib.tri as tri
from configobj import ConfigObj
import argparse

# Load configuration file for more order in paths
parser = argparse.ArgumentParser()
parser.add_argument("-conf", type=str, default="../../../config.ini", help="pass config file")
parser.add_argument("-toml_path", type=str, default="../run_experiments/run_workflow/smith_cloud.toml",
                    help="pass .toml file")
parser.add_argument("-sub_plot_dir", type=str, default="temp", help="pass sub plot directory to store the plots")
args = parser.parse_args()
config_file = args.conf
configuration = ConfigObj(os.path.expanduser(config_file))

rcParams['axes.labelsize'] = 16
rcParams['xtick.labelsize'] = 16
rcParams['ytick.labelsize'] = 16

color = sns.color_palette()
cmap_topo = salem.get_cmap('topo')
cmap_thick = plt.cm.get_cmap('YlGnBu')
cmap_glen=plt.get_cmap('RdBu_r')

#Load main repository path
MAIN_PATH = configuration['main_path']
sys.path.append(MAIN_PATH)

from ficetools import graphics, velocity, utils_funcs

# Paths to data
sub_plot_dir = args.sub_plot_dir
plot_path = os.path.join(MAIN_PATH, 'plots/'+ sub_plot_dir)
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

tomlf = args.toml_path
params = conf.ConfigParser(tomlf, top_dir=Path(MAIN_PATH))

#Reading mesh
mesh_in = fice_mesh.get_mesh(params)

x, y, t = graphics.read_fenics_ice_mesh(mesh_in)

# Constructing mesh functions from mesh
Q = FunctionSpace(mesh_in, 'Lagrange',1)
Qh = FunctionSpace(mesh_in, 'Lagrange',3)
M = FunctionSpace(mesh_in, 'DG', 0)

Qp = Q
V = VectorFunctionSpace(mesh_in, 'Lagrange', 1, dim=2)

# Read bed machine data
bedmachine = os.path.join(params.io.input_dir, params.io.bed_data_file)
bedmachine_smith = h5py.File(bedmachine, 'r')

bed = bedmachine_smith['bed'][:]
thick = bedmachine_smith['thick'][:]
surf_ice = bedmachine_smith['surf'][:]
x_bm = bedmachine_smith['x'][:]
y_bm = bedmachine_smith['y'][:]

xgbm, ygbm = np.meshgrid(x_bm, y_bm)

#Read velocity file used for the inversion
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

sg = graphics.define_salem_grid(vx_s)

# Read bglen
output_path = params.io.diagnostics_dir + '/inversion/'+ params.inversion.phase_suffix
file_alpha = params.io.run_name + params.inversion.phase_suffix +'_alpha_init_guess.xml'
file_beta = params.io.run_name + params.inversion.phase_suffix +'_beta_init_guess.xml'
# Read xml files from fenics
alpha_init_file = os.path.join(output_path, file_alpha)
beta_init_file = os.path.join(output_path, file_beta)

v_alphaini = utils_funcs.compute_vertex_for_parameter_field(alpha_init_file, Qp, M, mesh_in)
v_betaini = utils_funcs.compute_vertex_for_parameter_field(beta_init_file, Qp, M, mesh_in)

# Now plotting
g = 1.2
fig1 = plt.figure(figsize=(9*g, 14*g))#, constrained_layout=True)
spec = gridspec.GridSpec(3, 2, wspace=0.3, hspace=0.2)

# tick_options = {'axis':'both','which':'both','bottom':False,
#     'top':False,'left':False,'right':False,'labelleft':False, 'labelbottom':False}

tick_options_mesh = {'axis':'both','which':'both','bottom':False,
    'top':True,'left':False,'right':True,'labelright':True, 'labeltop':True, 'labelbottom':False}


################### MESH ######################################################################

ax0 = plt.subplot(spec[0])
#ax0.tick_params(**tick_options_mesh)
ax0.set_aspect('equal')
smap = salem.Map(sg, countries=False)

x_n, y_n = smap.grid.transform(x, y,
                              crs=sg.proj)
c = ax0.triplot(x_n, y_n, t, color=sns.xkcd_rgb["black"], lw=0.2)
smap.set_lonlat_contours(xinterval=1.0, yinterval=0.5, add_tick_labels=True, linewidths=1.5)
smap.set_scale_bar(location=(0.89, 0.05), add_bbox=True, linewidth=5)
out = graphics.get_projection_grid_labels(smap)
smap.xtick_pos = out[0]
smap.xtick_val = out[1]
smap.ytick_pos = out[2]
smap.ytick_val = out[3]
smap.visualize(ax=ax0, orientation='horizontal', addcbar=False)
at = AnchoredText('a', prop=dict(size=18), frameon=True, loc='upper left')
ax0.add_artist(at)

# lon_lables, lat_lables = graphics.get_projection_grid_labels(smap)
# print(lon_lables)
# print(lat_lables)

################### BED #######################################################################

ax1 = plt.subplot(spec[1])
ax1.set_aspect('equal')
divider = make_axes_locatable(ax1)
cax = divider.append_axes("bottom", size="5%", pad=0.5)

smap = salem.Map(sg, countries=False)
x_n, y_n = smap.grid.transform(x_bm, y_bm,
                              crs=sg.proj)
levels, ticks = graphics.set_levels_ticks_for_colorbar(-3000, 3000)
cs = ax1.contourf(x_n, y_n, bed, levels = levels, cmap=cmap_topo)
smap.set_lonlat_contours(xinterval=1.0, yinterval=0.5, add_tick_labels=True, linewidths=1.5)
smap.set_vmin(-3000)
smap.set_vmax(3000)
smap.set_cmap(cmap_topo)
out = graphics.get_projection_grid_labels(smap)
smap.xtick_pos = out[0]
smap.xtick_val = out[1]
smap.ytick_pos = out[2]
smap.ytick_val = out[3]
smap.visualize(ax=ax1, addcbar=False)
cbar = smap.colorbarbase(cax=cax, orientation="horizontal", label='bed altitude [m above s.l.]')
at = AnchoredText('b', prop=dict(size=18), frameon=True, loc='upper left')
ax1.add_artist(at)

################### THICH ########################### ###############################

ax2 = plt.subplot(spec[2])
ax2.set_aspect('equal')
divider = make_axes_locatable(ax2)
cax = divider.append_axes("bottom", size="5%", pad=0.5)
smap = salem.Map(sg, countries=False)
x_n, y_n = smap.grid.transform(x_bm, y_bm,
                              crs=sg.proj)
levels, ticks = graphics.set_levels_ticks_for_colorbar(1, 3000)
cs = ax2.contourf(x_n, y_n, thick, levels = levels, cmap=cmap_thick)
smap.set_lonlat_contours(xinterval=1.0, yinterval=0.5, add_tick_labels=True, linewidths=1.5)
smap.set_cmap(cmap_thick)
smap.set_vmin(1)
smap.set_vmax(3000)
out = graphics.get_projection_grid_labels(smap)
smap.xtick_pos = out[0]
smap.xtick_val = out[1]
smap.ytick_pos = out[2]
smap.ytick_val = out[3]
smap.visualize(ax=ax2, orientation='horizontal', addcbar=False)
cbar = smap.colorbarbase(cax=cax, orientation="horizontal", label='Ice thickness [m]')
at = AnchoredText('c', prop=dict(size=18), frameon=True, loc='upper left')
ax2.add_artist(at)

######################################### ice surface ############################################

ax3 = plt.subplot(spec[3])
ax3.set_aspect('equal')
divider = make_axes_locatable(ax3)
cax = divider.append_axes("bottom", size="5%", pad=0.5)
smap = salem.Map(sg, countries=False)
x_n, y_n = smap.grid.transform(x_bm, y_bm,
                              crs=sg.proj)
levels, ticks = graphics.set_levels_ticks_for_colorbar(0, 3000)
cs = ax3.contourf(x_n, y_n, surf_ice, levels = levels, cmap=cmap_topo)
smap.set_lonlat_contours(xinterval=1.0, yinterval=0.5, add_tick_labels=True, linewidths=1.5)
smap.set_cmap(cmap_topo)
smap.set_vmin(0)
smap.set_vmax(3000)
out = graphics.get_projection_grid_labels(smap)
smap.xtick_pos = out[0]
smap.xtick_val = out[1]
smap.ytick_pos = out[2]
smap.ytick_val = out[3]
smap.visualize(ax=ax3, orientation='horizontal', addcbar=False)
cbar = smap.colorbarbase(cax=cax, orientation="horizontal",
                         label='Ice surface elevation [m above s.l.]')
at = AnchoredText('d', prop=dict(size=18), frameon=True, loc='upper left')
ax3.add_artist(at)

######################################### velocity ############################################

ax4 = plt.subplot(spec[4])
ax4.set_aspect('equal')
divider = make_axes_locatable(ax4)
cax = divider.append_axes("bottom", size="5%", pad=0.5)
smap = salem.Map(sg, countries=False)
smap.set_data(vv)
smap.set_cmap('viridis')
smap.set_lonlat_contours(xinterval=1.0, yinterval=0.5, add_tick_labels=True, linewidths=1.5)
out = graphics.get_projection_grid_labels(smap)
smap.xtick_pos = out[0]
smap.xtick_val = out[1]
smap.ytick_pos = out[2]
smap.ytick_val = out[3]
smap.visualize(ax=ax4, orientation='horizontal', addcbar=False)
cbar = smap.colorbarbase(cax=cax, orientation="horizontal",
                         label='MEaSUREs velocity (2013-2014)\n [m/yr]')
at = AnchoredText('e', prop=dict(size=18), frameon=True, loc='upper left')
ax4.add_artist(at)

######################################### A creep param ############################################

ax5 = plt.subplot(spec[5])
ax5.set_aspect('equal')
divider = make_axes_locatable(ax5)
cax = divider.append_axes("bottom", size="5%", pad=0.5)
smap = salem.Map(sg, countries=False)
x_n, y_n = smap.grid.transform(x, y,
                              crs=sg.proj)
levels, ticks = graphics.set_levels_ticks_for_colorbar(np.min(v_betaini), np.max(v_betaini))
c = ax5.tricontourf(x_n, y_n, t, v_betaini, levels = levels, cmap='viridis')
smap.set_lonlat_contours(xinterval=1.0, yinterval=0.5, add_tick_labels=True, linewidths=1.5)
out = graphics.get_projection_grid_labels(smap)
smap.xtick_pos = out[0]
smap.xtick_val = out[1]
smap.ytick_pos = out[2]
smap.ytick_val = out[3]
smap.set_vmin(np.min(v_betaini))
smap.set_vmax(np.max(v_betaini))
smap.set_cmap('viridis')
smap.visualize(ax=ax5, orientation='horizontal', addcbar=False)
cbar = smap.colorbarbase(cax=cax, orientation="horizontal", label='A creep parameter [Pa $yr^{1/3}$]')
at = AnchoredText('f', prop=dict(size=18), frameon=True, loc='upper left')
ax5.add_artist(at)

plt.tight_layout()
plt.savefig(os.path.join(plot_path, 'input_data.png'), bbox_inches='tight')