"""
Plot run_invsigma output, Paths of QIS along T's in smith exp.

- Reads output data (stored in .h5)
- Plots things in a multiplot grid

@authors: Fenics_ice contributors
"""
import sys
import numpy as np
import os
import salem
import h5py
import argparse
import pickle
from pathlib import Path
from configobj import ConfigObj

from fenics import *
from fenics_ice import config as conf
from fenics_ice import mesh as fice_mesh
from ufl import finiteelement

import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

import matplotlib.tri as tri
rcParams['axes.labelsize'] = 18
rcParams['xtick.labelsize'] = 18
rcParams['ytick.labelsize'] = 18
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-conf", type=str,
                    default="../../../config.ini", help="pass config file")
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

# Paths to data
sub_plot_dir = args.sub_plot_dir
plot_path = os.path.join(MAIN_PATH, 'plots/'+ sub_plot_dir)
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

# Get the right toml
tomlf = args.toml_path
params = conf.ConfigParser(tomlf, top_dir=Path(MAIN_PATH))

print('We are plotting the following configuration')
print('---------------------------------------------')
dict_config = {'phase_suffix':params.inversion.phase_suffix,
            'gamma_alpha': params.inversion.gamma_alpha,
            'delta_alpha': params.inversion.delta_alpha,
            'gama_beta': params.inversion.gamma_beta,
            'delta_beta': params.inversion.delta_beta}
print(dict_config)
print('---------------------------------------------')
out_dir_main = params.io.output_dir

exp_outdir_fwd = utils_funcs.define_stage_output_paths(params, 'time')
exp_outdir_errp = utils_funcs.define_stage_output_paths(params, 'error_prop')

#Config five
fnames = utils_funcs.get_file_names_for_path_plot(params)

Q_fname = fnames[0]
sigma_fname = fnames[1]
sigma_prior_fname = fnames[2]

Qfile = exp_outdir_fwd / Q_fname
sigmafile = exp_outdir_errp / sigma_fname
sigmapriorfile = exp_outdir_errp / sigma_prior_fname

assert Qfile.is_file()
assert sigmafile.is_file()
assert sigmapriorfile.is_file()


with open(Qfile, 'rb') as f:
    out = pickle.load(f)
dQ_vals = out[0]
dQ_t = out[1]

with open(sigmafile, 'rb') as f:
    out = pickle.load(f)
sigma_vals = out[0]
sigma_t = out[1]

with open(sigmapriorfile, 'rb') as f:
    out = pickle.load(f)
sigma_prior_vals = out[0]

sigma_interp = np.interp(dQ_t, sigma_t, sigma_vals)
sigma_prior_interp = np.interp(dQ_t, sigma_t, sigma_prior_vals)

x_qoi = dQ_t
y_qoi = dQ_vals[0] - dQ_vals
s = 2*sigma_interp
sp = 2*sigma_prior_interp


########### loading things to plot STD sigma alpha / beta
#Reading mesh
mesh_in = fice_mesh.get_mesh(params)

# Get mesh triangulation
x, y, t = graphics.read_fenics_ice_mesh(mesh_in)
trim = tri.Triangulation(x, y, t)

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


exp_outdir_invsigma = utils_funcs.define_stage_output_paths(params, 'inv_sigma')
file_names_invsigma = utils_funcs.get_file_names_for_invsigma_plot(params)

path_alphas = exp_outdir_invsigma / file_names_invsigma[0]
path_betas = exp_outdir_invsigma / file_names_invsigma[1]

assert path_alphas.is_file()
assert path_betas.is_file()

alpha_sigma = Function(M, str(path_alphas))
alpha_sig = project(alpha_sigma, M)
sigma_alpha = alpha_sig.compute_vertex_values(mesh_in)

beta_sigma = Function(M, str(path_betas))
beta_sig = project(beta_sigma, M)
sigma_beta = beta_sig.compute_vertex_values(mesh_in)

######## Load grid for latitude and longitude ##################
import pyproj
import salem
import xarray as xr

#Read velocity file used for the inversion
configuration = ConfigObj(os.path.join(MAIN_PATH, 'config.ini'))
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

# Lets define our salem grid. (but we modified things
# cuz fabi's code only works for the North! TODO: ask fabien)

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


############### Plotting 1 #########################################
from matplotlib import rcParams
rcParams['axes.labelsize'] = 14
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14

g=1.2
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec

fig1 = plt.figure(figsize=(10*g, 5*g))#, constrained_layout=True)
spec = gridspec.GridSpec(1, 2, wspace=0.3, hspace=0.25)

colors = sns.color_palette()

ax0 = plt.subplot(spec[0])

ax0.plot(x_qoi, y_qoi, color=colors[3], label='QoI projection')
ax0.fill_between(x_qoi, y_qoi-s, y_qoi+s, facecolor=colors[3], alpha=0.3)

ax0.set_xlabel('Time (yrs)')
ax0.set_ylabel(r'$Q$ $(m^4)$')
ax0.legend(loc='lower left')
at = AnchoredText('a', prop=dict(size=18), frameon=True, loc='upper right')
ax0.add_artist(at)

ax1 = plt.subplot(spec[1])

ax1.semilogy(x_qoi, sp, color=colors[3], linestyle='dashed', label='prior')
ax1.semilogy(x_qoi, s, color=colors[3], label='posterior')

ax1.legend()
ax1.set_xlabel('Time (yrs)')
ax1.set_ylabel(r'$\sigma$ $(m^4)$')
at = AnchoredText('b', prop=dict(size=18), frameon=True, loc='upper right')
ax1.add_artist(at)

plt.tight_layout()
plt.savefig(os.path.join(plot_path, 'sigma_paths_vaf.png'), bbox_inches='tight')

############### Plotting 2 #########################################

cmap_params_viridis = sns.color_palette("viridis", as_cmap=True)
colors = sns.color_palette()

# Now plotting
r= 1.2

tick_options = {'axis':'both','which':'both','bottom':False,
     'top':False,'left':False,'right':False,'labelleft':False, 'labelbottom':False}


fig2 = plt.figure(figsize=(10*r, 10*r))#, constrained_layout=True)
spec = gridspec.GridSpec(2, 2, wspace=0.3, hspace=0.3)


# Sigma QoI path
ax0 = plt.subplot(spec[0])
colorsea = sns.color_palette()
ax0.plot(x_qoi, y_qoi, color=colors[3], label='QoI projection')
ax0.fill_between(x_qoi, y_qoi-s, y_qoi+s, facecolor=colors[3], alpha=0.3)
ax0.set_xlabel('Time (yrs)')
ax0.set_ylabel(r'$Q$ $(m^4)$')
ax0.legend(loc='lower left')
at = AnchoredText('a', prop=dict(size=18), frameon=True, loc='upper right')
ax0.add_artist(at)


# STD sigma alpha
ax1 = plt.subplot(spec[1])
ax1.set_aspect('equal')
divider = make_axes_locatable(ax1)
cax = divider.append_axes("bottom", size="5%", pad=0.5)
smap = salem.Map(gv, countries=False)
x_n, y_n = smap.grid.transform(x, y,
                              crs=gv.proj)
minv = np.min(sigma_alpha)
maxv = np.max(sigma_alpha)
levels = np.linspace(minv,maxv,200)
ticks = np.linspace(minv,maxv,3)
c = ax1.tricontourf(x_n, y_n, t, sigma_alpha, levels=levels, cmap=cmap_params_viridis, extend="both")
smap.set_lonlat_contours(xinterval=1.0, yinterval=0.5, add_tick_labels=False, linewidths=1.5)
out = graphics.get_projection_grid_labels(smap)
smap.xtick_pos = out[0]
smap.xtick_val = out[1]
smap.ytick_pos = out[2]
smap.ytick_val = out[3]
smap.set_cmap(cmap_params_viridis)
smap.set_vmin(minv)
smap.set_vmax(maxv)
smap.set_extend('both')
smap.visualize(ax=ax1, orientation='horizontal', addcbar=False)
cbar = smap.colorbarbase(cax=cax, orientation="horizontal", ticks=ticks,
                         label='Sliding parameter STD' + r'($\alpha$)'  +  '\n [m$^{-1/6}$ yr$^{1/6}$]')
at = AnchoredText('c', prop=dict(size=18), frameon=True, loc='upper left')
ax1.add_artist(at)


# Prior prosterior projection
ax2 = plt.subplot(spec[2])
ax2.semilogy(x_qoi, sp, color=colors[3], linestyle='dashed', label='prior')
ax2.semilogy(x_qoi, s, color=colors[3], label='posterior')
ax2.legend()
ax2.set_xlabel('Time (yrs)')
ax2.set_ylabel(r'$\sigma$ $(m^4)$')
at = AnchoredText('b', prop=dict(size=18), frameon=True, loc='upper right')
ax2.add_artist(at)

# STD sigma beta
ax3 = plt.subplot(spec[3])
ax3.set_aspect('equal')

divider = make_axes_locatable(ax3)
cax = divider.append_axes("bottom", size="5%", pad=0.5)
smap = salem.Map(gv, countries=False)

x_n, y_n = smap.grid.transform(x, y,
                              crs=gv.proj)
minv = np.min(sigma_beta)
maxv = np.max(sigma_beta)
levels = np.linspace(minv,maxv,200)
ticks = np.linspace(minv,maxv,3)
c = ax3.tricontourf(x_n, y_n, t, sigma_beta, levels=levels, cmap=cmap_params_viridis, extend="both")
smap.set_lonlat_contours(xinterval=1.0, yinterval=0.5, add_tick_labels=False, linewidths=1.5)
out = graphics.get_projection_grid_labels(smap)
smap.xtick_pos = out[0]
smap.xtick_val = out[1]
smap.ytick_pos = out[2]
smap.ytick_val = out[3]
smap.set_cmap(cmap_params_viridis)
smap.set_vmin(minv)
smap.set_vmax(maxv)
smap.set_extend('both')
smap.visualize(ax=ax3, orientation='horizontal', addcbar=False)
cbar = smap.colorbarbase(cax=cax, orientation="horizontal", ticks=ticks,
                         label='Ice stiffness parameter STD' + r'($\beta$)' + '\n [Pa$^{1/2}$. yr$^{1/6}$]')
at = AnchoredText('d', prop=dict(size=18), frameon=True, loc='upper left')
ax3.add_artist(at)


plt.tight_layout()
plt.savefig(os.path.join(plot_path, 'std_sigma_alpha_beta.png'), bbox_inches='tight', dpi=150)
