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
args = parser.parse_args()
config_file = args.conf
configuration = ConfigObj(os.path.expanduser(config_file))

rcParams['axes.labelsize'] = 18
rcParams['xtick.labelsize'] = 18
rcParams['ytick.labelsize'] = 18

color = sns.color_palette()
cmap_topo = salem.get_cmap('topo')
cmap_thick = plt.cm.get_cmap('YlGnBu')
cmap_glen=plt.get_cmap('RdBu_r')

#Load main repository path
MAIN_PATH = configuration['main_path']
sys.path.append(MAIN_PATH)

# Paths to data
plot_path = os.path.join(MAIN_PATH, 'plots/')
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

run_files = os.path.join(MAIN_PATH, 'scripts/run_experiments/run_workflow')
toml = os.path.join(run_files, 'smith.toml')

params = conf.ConfigParser(toml, top_dir=Path(MAIN_PATH))

#Reading mesh
mesh_in = fice_mesh.get_mesh(params)

# Constructing mesh functions from mesh
Q = FunctionSpace(mesh_in, 'Lagrange',1)
Qh = FunctionSpace(mesh_in, 'Lagrange',3)
M = FunctionSpace(mesh_in, 'DG', 0)

Qp = Q
V = VectorFunctionSpace(mesh_in, 'Lagrange', 1, dim=2)

x = mesh_in.coordinates()[:,0]
y = mesh_in.coordinates()[:,1]
t = mesh_in.cells()

trim = tri.Triangulation(x, y, t)

# Read bed machine data
bedmachine = os.path.join(params.io.input_dir, params.io.bed_data_file)
bedmachine_smith = h5py.File(bedmachine, 'r')

bed = bedmachine_smith['bed'][:]
thick = bedmachine_smith['thick'][:]
surf_ice = bedmachine_smith['surf'][:]
x_bm = bedmachine_smith['x'][:]
y_bm = bedmachine_smith['y'][:]

xgbm, ygbm = np.meshgrid(x_bm, y_bm)

#Read velocity for inversion
path_to_vel = Path(os.path.join(params.io.input_dir,params.obs.vel_file))
out = inout.read_vel_obs(path_to_vel, model=None)

uv_obs_pts = out['uv_obs_pts']
u_obs = out['u_obs']
v_obs = out['v_obs']
u_std = out['u_std']
v_std = out['v_std']
x_vel, y_vel = np.split(uv_obs_pts, [-1], axis=1)
vel_obs = np.sqrt(u_obs**2 + v_obs**2)

xgvel, ygvel = np.meshgrid(np.unique(x_vel), np.unique(y_vel))
vel_obs_g = np.flipud(vel_obs.reshape(xgvel.shape))

# Read bglen
b_glen_file = os.path.join(params.io.input_dir, params.io.bglen_data_file)
b_glen = h5py.File(b_glen_file, 'r')

bglen = b_glen['bglen'][:]
x_bg, y_bg = np.meshgrid(b_glen['x'][:], b_glen['y'][:])

# Now plotting
g = 1.5

tick_options = {'axis':'both','which':'both','bottom':False,
    'top':False,'left':False,'right':False,'labelleft':False, 'labelbottom':False}

tick_options_mesh = {'axis':'both','which':'both','bottom':False,
    'top':True,'left':True,'right':False,'labelleft':True, 'labeltop':True, 'labelbottom':False}

fig1 = plt.figure(figsize=(10*g, 14*g))#, constrained_layout=True)
spec = gridspec.GridSpec(3, 2, wspace=0.01, hspace=0.3)

ax0 = plt.subplot(spec[0])
ax0.set_aspect('equal')
ax0.tick_params(**tick_options_mesh)
ax0.set_xlim(min(x), max(x))
ax0.set_ylim(min(y), max(y))
ax0.triplot(x, y, trim.triangles, '-', color='k', lw=0.2)
at = AnchoredText('a', prop=dict(size=18), frameon=True, loc='upper left')
ax0.add_artist(at)

ax1 = plt.subplot(spec[1])
ax1.set_aspect('equal')
ax1.tick_params(**tick_options)
ax1.set_xlim(min(x), max(x))
ax1.set_ylim(min(y), max(y))
minv = -3000
maxv = 3000
levels = np.linspace(minv,maxv,200)
ticks = np.linspace(minv,maxv,3)
divider = make_axes_locatable(ax1)
cax = divider.append_axes("bottom", size="5%", pad=0.2)
c = ax1.contourf(xgbm, ygbm, bed, levels = levels, cmap=cmap_topo)
cbar = plt.colorbar(c, cax=cax, ticks=ticks, orientation="horizontal")
cbar.ax.set_xlabel('bed altitude [m above s.l.]')
at = AnchoredText('b', prop=dict(size=18), frameon=True, loc='upper left')
ax1.add_artist(at)

ax2 = plt.subplot(spec[2])
ax2.set_aspect('equal')
ax2.tick_params(**tick_options)
ax2.set_xlim(min(x), max(x))
ax2.set_ylim(min(y), max(y))
minv = 1
maxv = 3000
levels = np.linspace(minv,maxv,200)
ticks = np.linspace(minv,maxv,3)
divider = make_axes_locatable(ax2)
cax = divider.append_axes("bottom", size="5%", pad=0.2)
c = ax2.contourf(xgbm, ygbm, thick, levels = levels, cmap=cmap_thick)
cbar = plt.colorbar(c, cax=cax, ticks=ticks, orientation="horizontal")
cbar.ax.set_xlabel('Ice thickness [m]')
at = AnchoredText('c', prop=dict(size=18), frameon=True, loc='upper left')
ax2.add_artist(at)

ax3 = plt.subplot(spec[3])
ax3.set_aspect('equal')
ax3.tick_params(**tick_options)
ax3.set_xlim(min(x), max(x))
ax3.set_ylim(min(y), max(y))
minv = 0
maxv = 3000
levels = np.linspace(minv,maxv,200)
ticks = np.linspace(minv,maxv,3)
divider = make_axes_locatable(ax3)
cax = divider.append_axes("bottom", size="5%", pad=0.2)
c = ax3.contourf(xgbm, ygbm, surf_ice, levels = levels, cmap=cmap_topo)
cbar = plt.colorbar(c, cax=cax, ticks=ticks, orientation="horizontal")
cbar.ax.set_xlabel('Ice surface elevation [m above s.l.]')
at = AnchoredText('d', prop=dict(size=18), frameon=True, loc='upper left')
ax3.add_artist(at)

ax4 = plt.subplot(spec[4])
ax4.set_aspect('equal')
ax4.tick_params(**tick_options)
divider = make_axes_locatable(ax4)
cax = divider.append_axes("bottom", size="5%", pad=0.2)
minv = 0
maxv = 2000
levels = np.linspace(minv,maxv,200)
ticks = np.linspace(minv,maxv,3)
c = ax4.contourf(xgvel, ygvel, vel_obs_g, levels = levels, cmap='viridis')
cbar = plt.colorbar(c, cax=cax, ticks=ticks, orientation="horizontal")
cbar.ax.set_xlabel('ice surface velocity [$m^{-1}$ yr]')
at = AnchoredText('e', prop=dict(size=18), frameon=True, loc='upper left')
ax4.add_artist(at)

ax5 = plt.subplot(spec[5])
ax5.set_aspect('equal')
ax5.tick_params(**tick_options)
divider = make_axes_locatable(ax5)
cax = divider.append_axes("bottom", size="5%", pad=0.2)
minv = np.min(bglen)
maxv = np.max(bglen)
print(maxv)
print(minv)
ticks = np.linspace(minv,maxv,3)
levels = np.linspace(minv,maxv,200)
c = ax5.contourf(x_bg, y_bg, bglen, levels = levels, cmap=cmap_glen)
cbar = plt.colorbar(c, cax=cax, ticks=ticks, orientation="horizontal")
cbar.ax.set_xlabel('A creep parameter [Pa $yr^{1/3}$]')
at = AnchoredText('f', prop=dict(size=18), frameon=True, loc='upper left')
ax5.add_artist(at)

plt.tight_layout()
plt.savefig(os.path.join(plot_path, 'input_data.png'), bbox_inches='tight')