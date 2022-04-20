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
from fenics import *
from fenics_ice import config as conf
from fenics_ice import mesh as fice_mesh
from pathlib import Path
import seaborn as sns
import numpy as np
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
from ficetools import utils_funcs, graphics

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

# Read output data to plot
diag_dir = params.io.diagnostics_dir
phase_suffix = params.inversion.phase_suffix
phase_name = params.inversion.phase_name
run_name = params.io.run_name

exp_outdir = Path(diag_dir) / phase_name / phase_suffix
file_u_name = "_".join((params.io.run_name+phase_suffix, 'U.xml'))
file_uv_obs = "_".join((params.io.run_name+phase_suffix, 'uv_obs.xml'))
file_alpha_init = "_".join((params.io.run_name+phase_suffix, 'alpha_init_guess.xml'))
file_beta_init = "_".join((params.io.run_name+phase_suffix, 'beta_init_guess.xml'))
file_alpha = "_".join((params.io.run_name+phase_suffix, 'alpha.xml'))

U_file = exp_outdir / file_u_name
uv_obs_file = exp_outdir / file_uv_obs
alpha_init_file = exp_outdir / file_alpha_init
beta_init_file = exp_outdir / file_beta_init
alpha_file = exp_outdir / file_alpha

assert U_file.is_file(), "File not found"
assert uv_obs_file.is_file(), "File not found"
assert  alpha_init_file.is_file(), "File not found"
assert beta_init_file.is_file(), "File not found"
assert alpha_file.is_file(), "File not found"

# Define function spaces for alpha only and uv_comp
alpha = Function(Qp, alpha_file)
# 1. The inverted value of B2; It is explicitly assumed that B2 = alpha**2
B2 = project(alpha*alpha, M)

# Get mesh triangulation
x, y, t = graphics.read_fenics_ice_mesh(mesh_in)

trim = tri.Triangulation(x, y, t)

# Compute vertex values for each parameter function
# in the mesh
v = utils_funcs.compute_vertex_for_velocity_field(U_file, V, Q, mesh_in)
v_alphaini = utils_funcs.compute_vertex_for_parameter_field(alpha_init_file, Qp, M, mesh_in)
v_alpha = B2.compute_vertex_values(mesh_in)

uv_obs = Function(M, uv_obs_file)
uv_obs_proj = project(uv_obs, Q)
v_obs = uv_obs_proj.compute_vertex_values(mesh_in)

# Now plotting
g = 1.5
fig1 = plt.figure(figsize=(10*g, 10*g))#, constrained_layout=True)
spec = gridspec.GridSpec(2, 2, hspace=0.01)#, wspace=0.05, hspace=0.05)


ax0 = plt.subplot(spec[0])
ax0.set_aspect('equal')
ax0.tick_params(**tick_options)
minv = 0
maxv = 2000
levels = np.linspace(minv,maxv,200)
ticks = np.linspace(minv,maxv,3)
divider = make_axes_locatable(ax0)
cax = divider.append_axes("bottom", size="5%", pad=0.1)
c = ax0.tricontourf(x, y, t, v, levels = levels, cmap=cmap_vel)
cbar = plt.colorbar(c, cax=cax, ticks=ticks, orientation="horizontal")
cbar.ax.set_xlabel('[$m^{-1}$ yr]')
ax0.set_title("Model ice velocity (U)", fontsize=18)
at = AnchoredText('a', prop=dict(size=18), frameon=True, loc='upper left')
ax0.add_artist(at)

ax1 = plt.subplot(spec[1])
ax1.set_aspect('equal')
ax1.tick_params(**tick_options)
divider = make_axes_locatable(ax1)
cax = divider.append_axes("bottom", size="5%", pad=0.1)
c = ax1.tricontourf(x, y, t, v_obs, levels = levels, cmap=cmap_vel)
cbar = plt.colorbar(c, cax=cax, ticks=ticks, orientation="horizontal")
cbar.ax.set_xlabel('[$m^{-1}$ yr]')
ax1.set_title("Velocity observations (MEaSUREs v2.0)", fontsize=18)
at = AnchoredText('b', prop=dict(size=18), frameon=True, loc='upper left')
ax1.add_artist(at)

ax2 = plt.subplot(spec[2])
ax2.set_aspect('equal')
ax2.tick_params(**tick_options)
divider = make_axes_locatable(ax2)
cax = divider.append_axes("bottom", size="5%", pad=0.1)
ax2.triplot(x, y, trim.triangles, '-', color='grey', lw=0.2, alpha=0.5)
minv = -70
maxv = 70
levels = np.linspace(minv,maxv,200)
ticks = np.linspace(minv,maxv,3)
c = ax2.tricontourf(x, y, t, v_alphaini, levels = levels, cmap=cmap_params)
cbar = plt.colorbar(c, cax=cax, ticks=ticks, orientation="horizontal")
cbar.ax.set_xlabel('alpha initial guess')
ax2.set_xlim(min(x), max(x))
ax2.set_ylim(min(y), max(y))
at = AnchoredText('c', prop=dict(size=18), frameon=True, loc='upper left')
ax2.set_title("Alpha first guess", fontsize=18)
ax2.add_artist(at)

ax3 = plt.subplot(spec[3])
ax3.set_aspect('equal')
ax3.tick_params(**tick_options)
divider = make_axes_locatable(ax3)
cax = divider.append_axes("bottom", size="5%", pad=0.1)
ax3.triplot(x, y, trim.triangles, '-', color='grey', lw=0.2, alpha=0.5)
minv = 0
maxv = np.max(v_alpha)
levels = np.linspace(minv,maxv,200)
ticks = np.linspace(minv,maxv,3)
c = ax3.tricontourf(x, y, t, v_alpha, levels = levels, cmap=cmap_params_inv)
cbar = plt.colorbar(c, cax=cax, ticks=ticks, orientation="horizontal")
cbar.ax.set_xlabel('$B^{2}$')
ax3.set_xlim(min(x), max(x))
ax3.set_ylim(min(y), max(y))
at = AnchoredText('d', prop=dict(size=18), frameon=True, loc='upper left')
ax3.set_title('inverse of $B^{2}$', fontsize=18)
ax3.add_artist(at)

plt.tight_layout()
plt.savefig(os.path.join(plot_path, 'inversion_output.png'), bbox_inches='tight')

