"""
Plot run_forward output, Gradient components

- Reads the input mesh
- Reads output data (stored in .h5)
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
args = parser.parse_args()
config_file = args.conf
configuration = ConfigObj(os.path.expanduser(config_file))

# Define main repository path
MAIN_PATH = configuration['main_path']
sys.path.append(MAIN_PATH)
from meshtools import meshtools

rcParams['axes.labelsize'] = 14
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14
color = sns.color_palette()
cmap_params = plt.get_cmap('RdBu_r')
tick_options = {'axis':'both','which':'both','bottom':False,
                'top':False,'left':False,'right':False,
                'labelleft':False, 'labelbottom':False, 'labeltop':False}

# Paths to data
plot_path = os.path.join(MAIN_PATH, 'plots/')
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

# Get the right toml
run_files = os.path.join(MAIN_PATH, 'scripts/run_experiments/run_workflow')
toml = os.path.join(run_files, 'smith.toml')

# Read in model run parameters
params = conf.ConfigParser(toml, top_dir=Path(MAIN_PATH))
outdir = params.io.output_dir

#Read and get mesh information
mesh_in = fice_mesh.get_mesh(params)

# Get mesh stuff
x = mesh_in.coordinates()[:,0]
y = mesh_in.coordinates()[:,1]
t = mesh_in.cells()
trim = tri.Triangulation(x, y, t)

# Compute the Mixspace function-space from the mesh
# TODO: this is only true for params.mesh.periodic_bc = False
# need to implement a different space definition if true
el = FiniteElement("Lagrange", mesh_in.ufl_cell(), 1)
mixedElem = el * el
Q = FunctionSpace(mesh_in, mixedElem)
dQ = Function(Q)

#Get the n_sens
n_sens = np.arange(0, params.time.num_sens)

# Now lets read the output
hdffile = os.path.join(outdir, params.io.run_name+'_dQ_ts.h5')

valpha_first, vbeta_first = meshtools.compute_vertex_for_dQ_components(dQ,
                                                                       mesh_in,
                                                                       hdffile,
                                                                       n_sens[5])
valpha_last, vbeta_last = meshtools.compute_vertex_for_dQ_components(dQ,
                                                                     mesh_in,
                                                                     hdffile,
                                                                     n_sens[-1])

#This stay fix as they are the best color levels
numlev = 40
minva = -1e11
maxva = 1e11

minvb = -1e10
maxvb = 1e10

levelsa = np.linspace(minva,maxva,200)
ticksa = np.linspace(minva,maxva,3)

levelsb = np.linspace(minvb,maxvb,200)
ticksb = np.linspace(minvb,maxvb,3)

g = 1.2

fig1 = plt.figure(figsize=(12*g, 10*g))#, constrained_layout=True)
spec = gridspec.GridSpec(2, 2, wspace=0.05, hspace=0.25)

ax0 = plt.subplot(spec[0])
ax0.set_aspect('equal')
ax0.tick_params(**tick_options)
divider = make_axes_locatable(ax0)
cax = divider.append_axes("bottom", size="5%", pad=0.1)
c = ax0.tricontourf(x, y, t, valpha_first, levels = levelsa, cmap=plt.get_cmap('RdBu_r'))
ax0.triplot(x, y, trim.triangles, '-', color='grey', lw=0.2, alpha=0.5)
cbar = plt.colorbar(c, cax=cax, ticks=ticksa, orientation="horizontal")
cbar.ax.set_xlabel('dQ_alpha')
n_text = AnchoredText('n_sens = '+ str(n_sens[5]), prop=dict(size=14), frameon=True, loc='lower left')
ax0.add_artist(n_text)
at = AnchoredText('a', prop=dict(size=14), frameon=True, loc='upper left')
ax0.add_artist(at)


ax1 = plt.subplot(spec[1])
ax1.set_aspect('equal')
ax1.tick_params(**tick_options)
divider = make_axes_locatable(ax1)
cax = divider.append_axes("bottom", size="5%", pad=0.1)
c = ax1.tricontourf(x, y, t, vbeta_first, levels = levelsb, cmap=plt.get_cmap('RdBu_r'))
ax1.triplot(x, y, trim.triangles, '-', color='grey', lw=0.2, alpha=0.5)
cbar = plt.colorbar(c, cax=cax, ticks=ticksb, orientation="horizontal")
cbar.ax.set_xlabel('dQ_beta')
n_text = AnchoredText('n_sens = '+ str(n_sens[5]), prop=dict(size=14), frameon=True, loc='lower left')
ax1.add_artist(n_text)
at = AnchoredText('b', prop=dict(size=14), frameon=True, loc='upper left')
ax1.add_artist(at)


ax2 = plt.subplot(spec[2])
ax2.set_aspect('equal')
ax2.tick_params(**tick_options)
divider = make_axes_locatable(ax2)
cax = divider.append_axes("bottom", size="5%", pad=0.1)
c = ax2.tricontourf(x, y, t, valpha_last, levels = levelsa, cmap=plt.get_cmap('RdBu_r'))
ax2.triplot(x, y, trim.triangles, '-', color='grey', lw=0.2, alpha=0.5)
cbar = plt.colorbar(c, cax=cax, ticks=ticksa, orientation="horizontal")
cbar.ax.set_xlabel('dQ_alpha')
n_text = AnchoredText('n_sens = '+ str(n_sens[-1]), prop=dict(size=14), frameon=True, loc='lower left')
ax2.add_artist(n_text)
at = AnchoredText('c', prop=dict(size=14), frameon=True, loc='upper left')
ax2.add_artist(at)

ax3 = plt.subplot(spec[3])
ax3.set_aspect('equal')
ax3.tick_params(**tick_options)
divider = make_axes_locatable(ax3)
cax = divider.append_axes("bottom", size="5%", pad=0.1)
c = ax3.tricontourf(x, y, t, vbeta_last, levels = levelsb, cmap=plt.get_cmap('RdBu_r'))
ax3.triplot(x, y, trim.triangles, '-', color='grey', lw=0.2, alpha=0.5)
cbar = plt.colorbar(c, cax=cax, ticks=ticksb, orientation="horizontal")
cbar.ax.set_xlabel('dQ_beta')
n_text = AnchoredText('n_sens = '+ str(n_sens[-1]), prop=dict(size=14), frameon=True, loc='lower left')
ax3.add_artist(n_text)
at = AnchoredText('d', prop=dict(size=14), frameon=True, loc='upper left')
ax3.add_artist(at)

plt.tight_layout()
plt.savefig(os.path.join(plot_path, 'dq_ts_output.png'),
            bbox_inches='tight')