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
import numpy as np

from fenics import *
from fenics_ice import config as conf
from fenics_ice import mesh as fice_mesh
from pathlib import Path
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


#Read and get mesh information
mesh_in = fice_mesh.get_mesh(params_il)

print('We are using this velocity data', params_il.config_dict['obs'])
print('We are using this velocity data', params_me.config_dict['obs'])

# Compute the function spaces from the Mesh
Q = FunctionSpace(mesh_in, 'Lagrange',1)
Qh = FunctionSpace(mesh_in, 'Lagrange',3)
M = FunctionSpace(mesh_in, 'DG', 0)

if not params_il.mesh.periodic_bc:
   Qp = Q
   V = VectorFunctionSpace(mesh_in,'Lagrange', 1, dim=2)
else:
    Qp = fice_mesh.get_periodic_space(params_il, mesh_in, dim=1)
    V =  fice_mesh.get_periodic_space(params_il, mesh_in, dim=2)

# Read output data to plot
diag_il = params_il.io.diagnostics_dir
phase_suffix_il = params_il.inversion.phase_suffix

exp_outdir_il = Path(diag_il) / params_il.inversion.phase_name / phase_suffix_il

file_u_il = "_".join((params_il.io.run_name+phase_suffix_il, 'U.xml'))
file_uvobs_il = "_".join((params_il.io.run_name+phase_suffix_il, 'uv_cloud.xml'))

file_u_std_il = "_".join((params_il.io.run_name+phase_suffix_il, 'u_std_cloud.xml'))
file_v_std_il = "_".join((params_il.io.run_name+phase_suffix_il, 'v_std_cloud.xml'))

U_il = exp_outdir_il / file_u_il
uv_obs_il = exp_outdir_il / file_uvobs_il
u_std_il = exp_outdir_il / file_u_std_il
v_std_il = exp_outdir_il / file_v_std_il

diag_me = params_me.io.diagnostics_dir
phase_suffix_me = params_me.inversion.phase_suffix
exp_outdir_me = Path(diag_me) / params_me.inversion.phase_name / phase_suffix_me


file_u_me = "_".join((params_me.io.run_name+phase_suffix_me, 'U.xml'))
file_uvobs_me = "_".join((params_me.io.run_name+phase_suffix_me, 'uv_cloud.xml'))

file_u_std_me = "_".join((params_me.io.run_name+phase_suffix_me, 'u_std_cloud.xml'))
file_v_std_me = "_".join((params_me.io.run_name+phase_suffix_me, 'v_std_cloud.xml'))

U_me = exp_outdir_me / file_u_me
uv_obs_me = exp_outdir_me / file_uvobs_me
u_std_me = exp_outdir_me / file_u_std_me
v_std_me = exp_outdir_me / file_v_std_me

#Check if files exist
assert U_il.is_file(), "File not found"
assert uv_obs_il.is_file(), "File not found"
assert u_std_il.is_file(), "File not found"
assert v_std_il.is_file(), "File not found"

assert U_me.is_file(), "File not found"
assert uv_obs_me.is_file(), "File not found"
assert u_std_me.is_file(), "File not found"
assert v_std_me.is_file(), "File not found"

uv_live = Function(M, str(uv_obs_il))
uv_obs_live = project(uv_live, Q)

uv_me = Function(M, str(uv_obs_me))
uv_obs_measures = project(uv_me, Q)

u_std_live = Function(Qp, str(u_std_il))
u_std_obs_il = project(u_std_live, Q)

v_std_live = Function(Qp, str(v_std_il))
v_std_obs_il = project(v_std_live, Q)

u_std_measures = Function(Qp, str(u_std_me))
u_std_obs_me = project(u_std_measures, Q)

v_std_measures = Function(Qp, str(v_std_me))
v_std_obs_me = project(v_std_measures, Q)

u_std_ilf = u_std_obs_il.compute_vertex_values(mesh_in)
v_std_ilf = v_std_obs_il.compute_vertex_values(mesh_in)
u_std_mef = u_std_obs_me.compute_vertex_values(mesh_in)
v_std_mef = v_std_obs_me.compute_vertex_values(mesh_in)

# Get mesh triangulation
x, y, t = graphics.read_fenics_ice_mesh(mesh_in)
trim = tri.Triangulation(x, y, t)

# Compute vertex values for each parameter function
# in the mesh
# Model velocities
U_measures = utils_funcs.compute_vertex_for_velocity_field(str(U_me), V, Q, mesh_in)
U_itlive =  utils_funcs.compute_vertex_for_velocity_field(str(U_il), V, Q, mesh_in)

# Velocity observations
uv_live = uv_obs_live.compute_vertex_values(mesh_in)
uv_measures = uv_obs_measures.compute_vertex_values(mesh_in)

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


# Now plotting
r = 2.0

fig1 = plt.figure(figsize=(10*r, 6*r))#, constrained_layout=True)
spec = gridspec.GridSpec(2, 3, hspace=0.3, wspace=0.1)

ax0 = plt.subplot(spec[4])
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
c = ax0.tricontourf(x_n, y_n, t, uv_measures, levels = levels, cmap='viridis', extend="both")
smap.set_lonlat_contours(xinterval=1.0, yinterval=0.5, add_tick_labels=False, linewidths=1.5, alpha=0.3)
out = graphics.get_projection_grid_labels(smap)
smap.xtick_pos = out[0]
smap.xtick_val = out[1]
smap.ytick_pos = out[2]
smap.ytick_val = out[3]
smap.set_vmin(minv)
smap.set_vmax(maxv)
smap.set_extend('both')
smap.set_cmap('viridis')
#ax0.triplot(x_n, y_n, trim.triangles, '-', color='grey', lw=0.2, alpha=0.5)
smap.visualize(ax=ax0, orientation='horizontal', addcbar=False)

cbar = smap.colorbarbase(cax=cax, orientation="horizontal",
                         label='MEaSUREs Velocity \n [m. $yr^{-1}$]')
at = AnchoredText('a', prop=dict(size=18), frameon=True, loc='upper left')
ax0.add_artist(at)

ax1 = plt.subplot(spec[5])
ax1.set_aspect('equal')
divider = make_axes_locatable(ax1)
cax = divider.append_axes("bottom", size="5%", pad=0.5)
smap = salem.Map(gv, countries=False)
x_n, y_n = smap.grid.transform(x, y,
                              crs=gv.proj)
minv = 0
maxv = 100
levels = np.linspace(minv,maxv,200)
ticks = np.linspace(minv,maxv,3)
c = ax1.tricontourf(x_n, y_n, t, u_std_mef, levels = levels, cmap='viridis', extend="both")
smap.set_lonlat_contours(xinterval=1.0, yinterval=0.5, add_tick_labels=False, linewidths=1.5, alpha=0.3)
out = graphics.get_projection_grid_labels(smap)
smap.xtick_pos = out[0]
smap.xtick_val = out[1]
smap.ytick_pos = out[2]
smap.ytick_val = out[3]
smap.set_vmin(minv)
smap.set_vmax(maxv)
smap.set_extend('both')
smap.set_cmap('viridis')
#ax0.triplot(x_n, y_n, trim.triangles, '-', color='grey', lw=0.2, alpha=0.5)
smap.visualize(ax=ax1, orientation='horizontal', addcbar=False)
cbar = smap.colorbarbase(cax=cax, orientation="horizontal",
                         label='VX STD MEaSUREs \n [m. $yr^{-1}$]')
at = AnchoredText('b', prop=dict(size=18), frameon=True, loc='upper left')
ax1.add_artist(at)

ax2 = plt.subplot(spec[6])
ax2.set_aspect('equal')
divider = make_axes_locatable(ax2)
cax = divider.append_axes("bottom", size="5%", pad=0.5)
smap = salem.Map(gv, countries=False)
x_n, y_n = smap.grid.transform(x, y,
                              crs=gv.proj)
minv = 0
maxv = 100
levels = np.linspace(minv,maxv,200)
ticks = np.linspace(minv,maxv,3)
c = ax2.tricontourf(x_n, y_n, t, v_std_mef, levels = levels, cmap='viridis', extend="both")
smap.set_lonlat_contours(xinterval=1.0, yinterval=0.5, add_tick_labels=False, linewidths=1.5, alpha=0.3)
out = graphics.get_projection_grid_labels(smap)
smap.xtick_pos = out[0]
smap.xtick_val = out[1]
smap.ytick_pos = out[2]
smap.ytick_val = out[3]
smap.set_vmin(minv)
smap.set_vmax(maxv)
smap.set_extend('both')
smap.set_cmap('viridis')
#ax0.triplot(x_n, y_n, trim.triangles, '-', color='grey', lw=0.2, alpha=0.5)
smap.visualize(ax=ax2, orientation='horizontal', addcbar=False)
cbar = smap.colorbarbase(cax=cax, orientation="horizontal",
                         label='VY STD MEaSUREs \n [m. $yr^{-1}$]')
at = AnchoredText('c', prop=dict(size=18), frameon=True, loc='upper left')
ax2.add_artist(at)


ax3 = plt.subplot(spec[0])
ax3.set_aspect('equal')
divider = make_axes_locatable(ax3)
cax = divider.append_axes("bottom", size="5%", pad=0.5)
smap = salem.Map(gv, countries=False)

x_n, y_n = smap.grid.transform(x, y,
                              crs=gv.proj)
# minv = np.min(alpha_v_il)
# maxv = np.max(alpha_v_il)
minv = 0
maxv = 1000
levels = np.linspace(minv,maxv,200)
ticks = np.linspace(minv,maxv,3)
c = ax3.tricontourf(x_n, y_n, t, uv_live, levels=levels, cmap='viridis', extend="both")
smap.set_lonlat_contours(xinterval=1.0, yinterval=0.5, add_tick_labels=False, linewidths=1.5)
#ax4.triplot(x_n, y_n, trim.triangles, '-', color='grey', lw=0.2, alpha=0.5)
out = graphics.get_projection_grid_labels(smap)
smap.xtick_pos = out[0]
smap.xtick_val = out[1]
smap.ytick_pos = out[2]
smap.ytick_val = out[3]
smap.set_cmap('viridis')
smap.set_vmin(minv)
smap.set_vmax(maxv)
smap.set_extend('both')
smap.visualize(ax=ax3, orientation='horizontal', addcbar=False)
cbar = smap.colorbarbase(cax=cax, orientation="horizontal",
                         label='ITSLive velocity \n [m. $yr^{-1}$]')
at = AnchoredText('d', prop=dict(size=18), frameon=True, loc='upper left')
ax3.add_artist(at)


ax4 = plt.subplot(spec[1])
ax4.set_aspect('equal')
divider = make_axes_locatable(ax4)
cax = divider.append_axes("bottom", size="5%", pad=0.5)
smap = salem.Map(gv, countries=False)
x_n, y_n = smap.grid.transform(x, y,
                              crs=gv.proj)
minv = 0
maxv = 100
levels = np.linspace(minv,maxv,200)
ticks = np.linspace(minv,maxv,3)
c = ax4.tricontourf(x_n, y_n, t, u_std_ilf, levels=levels, cmap='viridis', extend="both")
smap.set_lonlat_contours(xinterval=1.0, yinterval=0.5, add_tick_labels=False, linewidths=1.5)
#ax6.triplot(x_n, y_n, trim.triangles, '-', color='grey', lw=0.2, alpha=0.5)
out = graphics.get_projection_grid_labels(smap)
smap.xtick_pos = out[0]
smap.xtick_val = out[1]
smap.ytick_pos = out[2]
smap.ytick_val = out[3]
smap.set_cmap('viridis')
smap.set_vmin(minv)
smap.set_vmax(maxv)
smap.set_extend('both')
smap.visualize(ax=ax4, orientation='horizontal', addcbar=False)
cbar = smap.colorbarbase(cax=cax, orientation="horizontal",
                         label='VX STD ITSLive \n [m. $yr^{-1}$]')
at = AnchoredText('e', prop=dict(size=18), frameon=True, loc='upper left')
ax4.add_artist(at)

ax5 = plt.subplot(spec[2])
ax5.set_aspect('equal')
divider = make_axes_locatable(ax5)
cax = divider.append_axes("bottom", size="5%", pad=0.5)
smap = salem.Map(gv, countries=False)
x_n, y_n = smap.grid.transform(x, y,
                              crs=gv.proj)
minv = 0
maxv = 100
levels = np.linspace(minv,maxv,200)
ticks = np.linspace(minv,maxv,3)
c = ax5.tricontourf(x_n, y_n, t, v_std_ilf, levels=levels, cmap='viridis', extend="both")
smap.set_lonlat_contours(xinterval=1.0, yinterval=0.5, add_tick_labels=False, linewidths=1.5)
#ax6.triplot(x_n, y_n, trim.triangles, '-', color='grey', lw=0.2, alpha=0.5)
out = graphics.get_projection_grid_labels(smap)
smap.xtick_pos = out[0]
smap.xtick_val = out[1]
smap.ytick_pos = out[2]
smap.ytick_val = out[3]
smap.set_cmap('viridis')
smap.set_vmin(minv)
smap.set_vmax(maxv)
smap.set_extend('both')
smap.visualize(ax=ax5, orientation='horizontal', addcbar=False)
cbar = smap.colorbarbase(cax=cax, orientation="horizontal",
                         label='VY STD ITSLive \n [m. $yr^{-1}$]')
at = AnchoredText('f', prop=dict(size=18), frameon=True, loc='upper left')
ax5.add_artist(at)

plt.tight_layout()
plt.savefig(os.path.join(plot_path, 'input_vel.png'),
            bbox_inches='tight')
