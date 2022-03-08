"""
Plot run_forward output, Gradient components

- Reads the input mesh
- Reads output data (stored in .h5)
- Plots things in a multiplot grid

@authors: Fenics_ice contributors
"""
import sys
import numpy as np
import os
import argparse
from pathlib import Path
from configobj import ConfigObj

from fenics_ice import model, config
from fenics_ice import mesh as fice_mesh
from ufl import finiteelement
from dolfin import *

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

import matplotlib.tri as tri
rcParams['axes.labelsize'] = 18
rcParams['xtick.labelsize'] = 18
rcParams['ytick.labelsize'] = 18

# Load configuration file for more order in paths
parser = argparse.ArgumentParser()
parser.add_argument("-conf", type=str, default="../../../config.ini", help="pass config file")
parser.add_argument('-n_sens', nargs="+", type=int, help="pass n_sens to plot (max 2)")
args = parser.parse_args()
config_file = args.conf
n_sens = args.n_sens
configuration = ConfigObj(os.path.expanduser(config_file))

# Define main repository path
MAIN_PATH = configuration['main_path']
sys.path.append(MAIN_PATH)
from meshtools import meshtools

# Paths to data
plot_path = os.path.join(MAIN_PATH, 'plots/')
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

# Get the right toml
run_files = os.path.join(MAIN_PATH, 'scripts/run_experiments/run_workflow')
toml = os.path.join(run_files, 'smith.toml')

# Read in model run parameters
params = config.ConfigParser(toml, top_dir=Path(MAIN_PATH))
outdir = params.io.output_dir

#Read and get mesh information
mesh_in = fice_mesh.get_mesh(params)

# Get mesh stuff
x = mesh_in.coordinates()[:,0]
y = mesh_in.coordinates()[:,1]
t = mesh_in.cells()
trim = tri.Triangulation(x, y, t)

el = finiteelement.FiniteElement("Lagrange", mesh_in.ufl_cell(), 1)
mixedElem = el * el

Q = FunctionSpace(mesh_in, mixedElem)
dQ = Function(Q)

#Get the n_sens
num_sens = np.arange(0, params.time.num_sens)
print('To plot Time')
n_zero = num_sens[n_sens[0]]
print(n_zero)
n_last = num_sens[n_sens[-1]]
print(n_last)

t_sens = np.flip(np.linspace(params.time.run_length, 0, params.time.num_sens))
t_zero = np.round(t_sens[n_sens[0]])
print(t_zero)
t_last = np.round(t_sens[n_sens[-1]])
print(t_last)

# Now lets read the output
hdffile = os.path.join(outdir, params.io.run_name+'_dQ_ts.h5')

valpha_first, vbeta_first = meshtools.compute_vertex_for_dV_components(dQ,
                                                                       mesh_in,
                                                                       hdffile,
                                                                       'dQdalphaXbeta',
                                                                       n_zero,
                                                                       mult_mmatrix=True)
valpha_last, vbeta_last = meshtools.compute_vertex_for_dV_components(dQ,
                                                                     mesh_in,
                                                                     hdffile,
                                                                     'dQdalphaXbeta',
                                                                     n_last,
                                                                     mult_mmatrix=True)

print(type(valpha_first))

valpha_first_std = meshtools.standarise_data_array(valpha_first)
vbeta_first_std = meshtools.standarise_data_array(vbeta_first)
valpha_last_std = meshtools.standarise_data_array(valpha_last)
vbeta_last_std = meshtools.standarise_data_array(vbeta_last)

levelsaf, ticksaf = meshtools.make_colorbar_scale(valpha_first_std, 0.10)
levelsbf, ticksbf = meshtools.make_colorbar_scale(vbeta_first_std, 0.10)
levelsal, ticksal = meshtools.make_colorbar_scale(valpha_last_std, 0.10)
levelsbl, ticksbl = meshtools.make_colorbar_scale(vbeta_last_std, 0.10)

tick_options = {'axis':'both','which':'both','bottom':False,
    'top':False,'left':False,'right':False,'labelleft':False, 'labelbottom':False, 'labeltop':False}

g = 1.0

fig1 = plt.figure(figsize=(12*g, 10*g))#, constrained_layout=True)
spec = gridspec.GridSpec(2, 2, wspace=0.05, hspace=0.25)

ax0 = plt.subplot(spec[0])
ax0.set_aspect('equal')
ax0.tick_params(**tick_options)
divider = make_axes_locatable(ax0)
cax = divider.append_axes("bottom", size="5%", pad=0.1)
c = ax0.tricontourf(x, y, t, valpha_first_std, levels=levelsaf, cmap=plt.get_cmap('RdBu')) # levels = levelsa
ax0.triplot(x, y, trim.triangles, '-', color='grey', lw=0.2, alpha=0.5)
cbar = plt.colorbar(c, cax=cax, ticks=ticksaf, orientation="horizontal") #ticks=ticksa
cbar.ax.set_xlabel(r'$\frac{dQ}{d\alpha}$')
n_text = AnchoredText('after year '+ str(t_zero), prop=dict(size=16), frameon=True, loc='upper right')
ax0.add_artist(n_text)
at = AnchoredText('a', prop=dict(size=16), frameon=True, loc='upper left')
ax0.add_artist(at)


ax1 = plt.subplot(spec[1])
ax1.set_aspect('equal')
ax1.tick_params(**tick_options)
divider = make_axes_locatable(ax1)
cax = divider.append_axes("bottom", size="5%", pad=0.1)
c = ax1.tricontourf(x, y, t, vbeta_first_std, levels=levelsbf, cmap=plt.get_cmap('RdBu')) #levels = levelsb,
ax1.triplot(x, y, trim.triangles, '-', color='grey', lw=0.2, alpha=0.5)
cbar = plt.colorbar(c, cax=cax, ticks=ticksbf, orientation="horizontal") # ticks=ticksb,
cbar.ax.set_xlabel(r'$\frac{dQ}{d\beta}$')
n_text = AnchoredText('after year '+ str(t_zero), prop=dict(size=16), frameon=True, loc='upper right')
ax1.add_artist(n_text)
at = AnchoredText('b', prop=dict(size=16), frameon=True, loc='upper left')
ax1.add_artist(at)


ax2 = plt.subplot(spec[2])
ax2.set_aspect('equal')
ax2.tick_params(**tick_options)
divider = make_axes_locatable(ax2)
cax = divider.append_axes("bottom", size="5%", pad=0.1)
c = ax2.tricontourf(x, y, t, valpha_last_std, levels=levelsaf, cmap=plt.get_cmap('RdBu')) #  levels = levelsa,
ax2.triplot(x, y, trim.triangles, '-', color='grey', lw=0.2, alpha=0.5)
cbar = plt.colorbar(c, cax=cax, ticks=ticksaf, orientation="horizontal") #ticks=ticksa
cbar.ax.set_xlabel(r'$\frac{dQ}{d\alpha}$')
n_text = AnchoredText('after year '+ str(t_last), prop=dict(size=16), frameon=True, loc='upper right')
ax2.add_artist(n_text)
at = AnchoredText('c', prop=dict(size=16), frameon=True, loc='upper left')
ax2.add_artist(at)

ax3 = plt.subplot(spec[3])
ax3.set_aspect('equal')
ax3.tick_params(**tick_options)
divider = make_axes_locatable(ax3)
cax = divider.append_axes("bottom", size="5%", pad=0.1)
c = ax3.tricontourf(x, y, t, vbeta_last_std, levels=levelsbf, cmap=plt.get_cmap('RdBu')) # levels = levelsb,
ax3.triplot(x, y, trim.triangles, '-', color='grey', lw=0.2, alpha=0.5)
cbar = plt.colorbar(c, cax=cax, ticks=ticksbf, orientation="horizontal") #ticks=ticksb,
cbar.ax.set_xlabel(r'$\frac{dQ}{d\beta}$')
n_text = AnchoredText('after year '+ str(t_last), prop=dict(size=16), frameon=True, loc='upper right')
ax3.add_artist(n_text)
at = AnchoredText('d', prop=dict(size=16), frameon=True, loc='upper left')
ax3.add_artist(at)

plt.tight_layout()
plt.savefig(os.path.join(plot_path, 'dq_ts_output_vaf'+
                         str(n_zero)+'_'+str(n_last)+'.png'),
            bbox_inches='tight')