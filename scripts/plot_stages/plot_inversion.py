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
import h5py
import numpy as np
import pandas as pd
import xarray as xr

from fenics import *
from fenics_ice import config as conf
from fenics_ice import mesh as fice_mesh
from pathlib import Path
import seaborn as sns
from configobj import ConfigObj
import argparse

# Matplotlib imports
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.tri as tri

# Load configuration file for more order in paths
parser = argparse.ArgumentParser()
parser.add_argument("-conf", type=str, default="../../../config.ini", help="pass config file")
parser.add_argument("-toml_path", type=str, default="../run_experiments/run_workflow/smith_cloud.toml",
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
color = sns.color_palette()
cmap_vel= plt.get_cmap('viridis')
cmap_params = plt.get_cmap('RdBu_r')
cmap_params_inv = plt.get_cmap('magma_r')
tick_options = {'axis':'both','which':'both','bottom':False,
    'top':False,'left':False,'right':False,'labelleft':False, 'labelbottom':False}

# Paths to data
sub_plot_dir = args.sub_plot_dir
plot_path = os.path.join(MAIN_PATH, 'plots/'+ sub_plot_dir)
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

tomlf = args.toml_path
params = conf.ConfigParser(tomlf, top_dir=Path(MAIN_PATH))

#Read and get mesh information
mesh_in = fice_mesh.get_mesh(params)

print('We are using this velocity data', params.config_dict['obs'])

# Compute the function spaces from the Mesh
Q = FunctionSpace(mesh_in, 'Lagrange',1)
Qh = FunctionSpace(mesh_in, 'Lagrange',3)
M = FunctionSpace(mesh_in, 'DG', 0)

if not params.mesh.periodic_bc:
   Qp = Q
   V = VectorFunctionSpace(mesh_in,'Lagrange', 1, dim=2)
else:
    Qp = fice_mesh.get_periodic_space(params, mesh_in, dim=1)
    V =  fice_mesh.get_periodic_space(params, mesh_in, dim=2)

out_dir = params.io.diagnostics_dir
exp_outdir = Path(out_dir) / params.inversion.phase_name / params.inversion.phase_suffix

name_part = params.io.run_name + params.inversion.phase_suffix

file_u = "_".join((name_part, 'U.xml'))
file_uvobs = "_".join((name_part, 'uv_cloud.xml'))
file_alpha = "_".join((name_part, 'alpha.xml'))
file_bglen = "_".join((name_part, 'beta.xml'))

U = exp_outdir / file_u
uv_obs = exp_outdir / file_uvobs
alpha = exp_outdir / file_alpha
bglen = exp_outdir / file_bglen

assert U.is_file(), "File not found"
assert uv_obs.is_file(), "File not found"
assert alpha.is_file(), "File not found"
assert bglen.is_file(), "File not found"

# Define function spaces for alpha only and uv_comp
alpha_f = Function(Qp, str(alpha))

C2 = project(alpha_f*alpha_f, M)

beta_f = Function(Qp, str(bglen))

def beta_to_bglen(x):
    return x*x

bglen_f = project(beta_to_bglen(beta_f), M)

uv = Function(M, str(uv_obs))
uv_obs_f = project(uv, Q)


# Get mesh triangulation
x, y, t = graphics.read_fenics_ice_mesh(mesh_in)
trim = tri.Triangulation(x, y, t)

U_vertex = utils_funcs.compute_vertex_for_velocity_field(str(U), V, Q, mesh_in)

# The recovered basal traction 
#
C2_v = C2.compute_vertex_values(mesh_in)
bglen_v = bglen_f.compute_vertex_values(mesh_in)

uv_vertex = uv_obs_f.compute_vertex_values(mesh_in)


alpha_proj = project(alpha_f, M)
alpha_v = alpha_proj.compute_vertex_values(mesh_in)

beta_proj = project(beta_f, M)
beta_v = beta_proj.compute_vertex_values(mesh_in)

# Get salem grid for latitude and longitude

import pyproj
import salem
import xarray as xr 

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

from matplotlib.ticker import FormatStrFormatter
from matplotlib import ticker, colors
cmap_vel='viridis'
cmap_params_alpha = sns.diverging_palette(145, 300, s=60, as_cmap=True)


r= 1.2

rcParams['axes.labelsize'] = 19
rcParams['xtick.labelsize'] = 19
rcParams['ytick.labelsize'] = 19

tick_options = {'axis':'both','which':'both','bottom':False,
     'top':False,'left':False,'right':False,'labelleft':False, 'labelbottom':False}

fig1 = plt.figure(figsize=(11*r, 10*r))#, constrained_layout=True)
spec = gridspec.GridSpec(2, 2, wspace=0.01, hspace=0.3)

cmap_params_alpha = sns.diverging_palette(145, 300, s=60, as_cmap=True)
cmap_params_bglen = sns.color_palette("viridis", as_cmap=True)

############################################

ax0 = plt.subplot(spec[0])
ax0.set_aspect('equal')
divider = make_axes_locatable(ax0)
cax = divider.append_axes("bottom", size="5%", pad=0.5)
smap = salem.Map(gv, countries=False)
x_n, y_n = smap.grid.transform(x, y,
                              crs=gv.proj)
minv = 0
maxv = 1000
levels = np.linspace(minv,maxv,200)
ticks = np.linspace(minv,maxv,3)
c = ax0.tricontourf(x_n, y_n, t, U_vertex, levels = levels, cmap=cmap_vel, extend="both")
smap.set_lonlat_contours(xinterval=1.0, yinterval=0.5, add_tick_labels=False, linewidths=1.5, alpha=0.3)
out = graphics.get_projection_grid_labels(smap)
smap.xtick_pos = out[0]
smap.xtick_val = out[1]
smap.ytick_pos = out[2]
smap.ytick_val = out[3]
smap.set_vmin(minv)
smap.set_vmax(maxv)
smap.set_extend('both')
smap.set_cmap(cmap_vel)
smap.visualize(ax=ax0, orientation='horizontal', addcbar=False)
cbar = smap.colorbarbase(cax=cax, orientation="horizontal", ticks=ticks,
                         label='Model velocity \n [m. $yr^{-1}$]')
at = AnchoredText('a', prop=dict(size=18), frameon=True, loc='upper left')
ax0.add_artist(at)

############################################

ax1 = plt.subplot(spec[1])
ax1.set_aspect('equal')
divider = make_axes_locatable(ax1)
cax = divider.append_axes("bottom", size="5%", pad=0.5)
c = ax1.tricontourf(x_n, y_n, t, uv_vertex, levels = levels, cmap=cmap_vel, extend="both")
smap.set_lonlat_contours(xinterval=1.0, yinterval=0.5, add_tick_labels=False, linewidths=1.5, alpha=0.3)
out = graphics.get_projection_grid_labels(smap)
smap.xtick_pos = out[0]
smap.xtick_val = out[1]
smap.ytick_pos = out[2]
smap.ytick_val = out[3]
smap.set_vmin(minv)
smap.set_vmax(maxv)
smap.set_extend('both')
smap.set_cmap(cmap_vel)
smap.visualize(ax=ax1, orientation='horizontal', addcbar=False)
cbar = smap.colorbarbase(cax=cax, orientation="horizontal", ticks=ticks, 
                         label='Observed velocity ITSLIVE \n [m. $yr^{-1}$]')
at = AnchoredText('b', prop=dict(size=18), frameon=True, loc='upper left')
ax1.add_artist(at)

#################################################################################



ax2 = plt.subplot(spec[2])
ax2.set_aspect('equal')
divider = make_axes_locatable(ax2)
cax = divider.append_axes("bottom", size="5%", pad=0.5)

minv = 0
maxv = 30
levels = np.linspace(minv,maxv,200)
ticks = np.linspace(minv,maxv,3)
c = ax2.tricontourf(x_n, y_n, t, alpha_v, levels=levels, cmap=cmap_params_alpha, extend="both")
smap.set_lonlat_contours(xinterval=1.0, yinterval=0.5, add_tick_labels=False, linewidths=1.5)
#ax2.triplot(x_n, y_n, trim.triangles, '-', color='grey', lw=0.2, alpha=0.5)
out = graphics.get_projection_grid_labels(smap)
smap.xtick_pos = out[0]
smap.xtick_val = out[1]
smap.ytick_pos = out[2]
smap.ytick_val = out[3]
smap.set_cmap(cmap_params_alpha)
smap.set_vmin(minv)
smap.set_vmax(maxv)
smap.set_extend('both')
smap.visualize(ax=ax2, orientation='horizontal', addcbar=False)
cbar = smap.colorbarbase(cax=cax, orientation="horizontal", ticks=ticks,
                         label='Sliding parameter' + r'($\alpha$)' + '\n [m$^{-1/6}$ yr$^{1/6}$]')

at = AnchoredText('c', prop=dict(size=18), frameon=True, loc='upper left')
ax2.add_artist(at)

################################################################################

ax3 = plt.subplot(spec[3])
ax3.set_aspect('equal')

divider = make_axes_locatable(ax3)
cax = divider.append_axes("bottom", size="5%", pad=0.5)

minv = 500
maxv = 800
levels = np.linspace(minv,maxv,200)
ticks = np.linspace(minv,maxv,3)
c = ax3.tricontourf(x_n, y_n, t, beta_v, levels=levels, cmap=cmap_params_bglen, extend="both")
smap.set_lonlat_contours(xinterval=1.0, yinterval=0.5, add_tick_labels=False, linewidths=1.5)
#ax3.triplot(x_n, y_n, trim.triangles, '-', color='grey', lw=0.2, alpha=0.5)
out = graphics.get_projection_grid_labels(smap)
smap.xtick_pos = out[0]
smap.xtick_val = out[1]
smap.ytick_pos = out[2]
smap.ytick_val = out[3]
smap.set_cmap(cmap_params_bglen)
smap.set_vmin(minv)
smap.set_vmax(maxv)
smap.set_extend('both')
smap.visualize(ax=ax3, orientation='horizontal', addcbar=False)
cbar = smap.colorbarbase(cax=cax, orientation="horizontal", ticks=ticks,
                         label='Ice stiffness parameter' + r'($\beta$)' + ' \n [Pa$^{1/2}$. yr$^{1/6}$] ')
at = AnchoredText('d', prop=dict(size=18), frameon=True, loc='upper left')
ax3.add_artist(at)

plt.tight_layout()
plt.savefig(os.path.join(plot_path, 'inversion_output.png'), bbox_inches='tight')

