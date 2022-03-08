"""
Plot run_eigendec output, Eigen value decay
and a set of Eigen vectors

- Reads the input mesh
- Reads output data (stored in .h5 and pickle)
- Plots things in a multiplot grid

@authors: Fenics_ice contributors
"""
import sys
import numpy as np
import os
import argparse
from pathlib import Path
from configobj import ConfigObj
import pickle
from collections import defaultdict

from fenics_ice import model, config
from fenics_ice import mesh as fice_mesh
from ufl import finiteelement
from dolfin import *

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.ticker as ticker
from matplotlib import rcParams
import seaborn as sns
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import Colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable


rcParams['axes.labelsize'] = 14
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14
rcParams['lines.markersize'] = 12
rcParams['legend.fontsize'] = 14

# Load configuration file for more order in paths
parser = argparse.ArgumentParser()
parser.add_argument("-conf", type=str, default="../../../config.ini", help="pass config file")
parser.add_argument('-n_eigen', nargs="+", type=int, help="pass number of eigen vectors to plot (max 4)")
args = parser.parse_args()
config_file = args.conf
num_eigen = args.n_eigen
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

run_name = params.io.run_name
lamfile = os.path.join(outdir, run_name+'_eigvals.p')

# Read eigenvalues decay and make the plot
with open(lamfile, 'rb') as f:
    out = pickle.load(f)

lam = out[0]
lpos = np.argwhere(lam > 0)
lneg = np.argwhere(lam < 0)
lind = np.arange(0,len(lam))

g=1.0
plt.figure(figsize=(8*g, 5*g))
plt.semilogy(lind[lpos], lam[lpos], '.', alpha = 0.2, mew=0)
plt.semilogy(lind[lneg], np.abs(lam[lneg]), '.k', alpha = 0.12, mew=0, label='Negative eigenvalues')
plt.legend()
plt.xlabel('Eigenvalue')
plt.ylabel('Magnitude')
plt.savefig(os.path.join(plot_path, 'eigen_decay.png'),
            bbox_inches='tight')

# Plot eigenvectors from here
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
dE = Function(Q)

#Get the n_sens
print('Number of Eigenvectors to plot')
print(num_eigen)

# Now lets read the output
hdffile = os.path.join(outdir, params.io.run_name+'_vr.h5')
print(hdffile)

# Name of the vectors variable inside the .h5 file
var_name = 'v'

# Lets make our own directory of results
eigen_v_alpha = defaultdict(list)
eigen_v_beta = defaultdict(list)
nametosum = 'eigen_vector'

for e in num_eigen:
    valpha, vbeta = meshtools.compute_vertex_for_dV_components(dE,
                                                               mesh_in,
                                                               hd5_fpath=hdffile,
                                                               var_name=var_name,
                                                               n_item=e,
                                                               mult_mmatrix=False)

    valpha_nom = meshtools.normalise_data_array(valpha)
    print(min(valpha_nom), max(valpha_nom))
    vbeta_nom = meshtools.normalise_data_array(vbeta)
    print(min(vbeta_nom), max(vbeta_nom))

    eigen_v_alpha[f'{nametosum}{e}'].append(valpha_nom)
    eigen_v_beta[f'{nametosum}{e}'].append(vbeta_nom)

# Levels of the color map that works

minva = -1.0
maxva = 1.0

tick_options = {'axis':'both','which':'both','bottom':False,
    'top':False,'left':False,'right':False,'labelleft':False, 'labelbottom':False, 'labeltop':False}

levels = np.linspace(minva, maxva, 100)
ticks = np.linspace(minva, maxva, 3)

g=1.2

fig1 = plt.figure(figsize=(6*g, 13*g))#, constrained_layout=True)
spec = gridspec.GridSpec(4, 2, wspace=0.15, hspace=0.15)

ax0 = plt.subplot(spec[0])
ax0.set_aspect('equal')
ax0.tick_params(**tick_options)
meshtools.plot_field_in_tricontourf(eigen_v_alpha['eigen_vector'+f'{num_eigen[0]}'][0],
                                    mesh_in,
                                    ax=ax0,
                                    varname='alpha',
                                    num_eigen=num_eigen[0],
                                    ticks=ticks,
                                    levels=levels)
at = AnchoredText('a', prop=dict(size=14), frameon=True, loc='upper left')
ax0.add_artist(at)

ax1 = plt.subplot(spec[1])
ax1.set_aspect('equal')
ax1.tick_params(**tick_options)
meshtools.plot_field_in_tricontourf(eigen_v_beta['eigen_vector'+f'{num_eigen[0]}'][0],
                                    mesh_in,
                                    ax=ax1,
                                    varname='beta',
                                    num_eigen=num_eigen[0],
                                    ticks=ticks,
                                    levels=levels, add_text=False)
at = AnchoredText('b', prop=dict(size=14), frameon=True, loc='upper left')
ax1.add_artist(at)

ax2 = plt.subplot(spec[2])
ax2.set_aspect('equal')
ax2.tick_params(**tick_options)
meshtools.plot_field_in_tricontourf(eigen_v_alpha['eigen_vector'+f'{num_eigen[1]}'][0],
                                    mesh_in,
                                    ax=ax2,
                                    varname='alpha',
                                    num_eigen=num_eigen[1],
                                    ticks=ticks,
                                    levels=levels)
at = AnchoredText('c', prop=dict(size=14), frameon=True, loc='upper left')
ax2.add_artist(at)

ax3 = plt.subplot(spec[3])
ax3.set_aspect('equal')
ax3.tick_params(**tick_options)
meshtools.plot_field_in_tricontourf(eigen_v_beta['eigen_vector'+f'{num_eigen[1]}'][0],
                                    mesh_in,
                                    ax=ax3,
                                    varname='beta',
                                    num_eigen=num_eigen[1],
                                    ticks=ticks,
                                    levels=levels, add_text=False)
at = AnchoredText('d', prop=dict(size=14), frameon=True, loc='upper left')
ax3.add_artist(at)


ax4 = plt.subplot(spec[4])
ax4.set_aspect('equal')
ax4.tick_params(**tick_options)
meshtools.plot_field_in_tricontourf(eigen_v_alpha['eigen_vector'+f'{num_eigen[2]}'][0],
                                    mesh_in,
                                    ax=ax4,
                                    varname='alpha',
                                    num_eigen=num_eigen[2],
                                    ticks=ticks,
                                    levels=levels)
at = AnchoredText('e', prop=dict(size=14), frameon=True, loc='upper left')
ax4.add_artist(at)

ax5 = plt.subplot(spec[5])
ax5.set_aspect('equal')
ax5.tick_params(**tick_options)
meshtools.plot_field_in_tricontourf(eigen_v_beta['eigen_vector'+f'{num_eigen[2]}'][0],
                                    mesh_in,
                                    ax=ax5,
                                    varname='beta',
                                    num_eigen=num_eigen[2],
                                    ticks=ticks,
                                    levels=levels, add_text=False)
at = AnchoredText('f', prop=dict(size=14), frameon=True, loc='upper left')
ax5.add_artist(at)


ax6 = plt.subplot(spec[6])
ax6.set_aspect('equal')
ax6.tick_params(**tick_options)
meshtools.plot_field_in_tricontourf(eigen_v_alpha['eigen_vector'+f'{num_eigen[3]}'][0],
                                    mesh_in,
                                    ax=ax6,
                                    varname='alpha',
                                    num_eigen=num_eigen[3],
                                    ticks=ticks,
                                    levels=levels)
at = AnchoredText('g', prop=dict(size=14), frameon=True, loc='upper left')
ax6.add_artist(at)

ax7 = plt.subplot(spec[7])
ax7.set_aspect('equal')
ax7.tick_params(**tick_options)
meshtools.plot_field_in_tricontourf(eigen_v_beta['eigen_vector'+f'{num_eigen[3]}'][0],
                                    mesh_in,
                                    ax=ax7,
                                    varname='beta',
                                    num_eigen=num_eigen[3],
                                    ticks=ticks,
                                    levels=levels, add_text=False)
at = AnchoredText('h', prop=dict(size=14), frameon=True, loc='upper left')
ax7.add_artist(at)

plt.tight_layout()
plt.savefig(os.path.join(plot_path, 'eigen_vectors_default.png'),
            bbox_inches='tight')