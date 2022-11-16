import sys
import salem
import pyproj
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from decimal import Decimal
import os
from pathlib import Path
from configobj import ConfigObj

from fenics import *
from fenics_ice import config as conf
from fenics_ice import mesh as fice_mesh
from ufl import finiteelement

from matplotlib import rcParams
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
parser.add_argument('-n_sens', nargs="+", type=int, help="pass n_sens to plot (max 2)")
parser.add_argument("-sub_plot_dir", type=str, default="temp", help="pass sub plot directory to store the plots")
args = parser.parse_args()
config_file = args.conf
configuration = ConfigObj(os.path.expanduser(config_file))

# Define main repository path
MAIN_PATH = configuration['main_path']
sys.path.append(MAIN_PATH)


# Paths to data
sub_plot_dir = args.sub_plot_dir
plot_path = os.path.join(MAIN_PATH, 'plots/'+ sub_plot_dir)
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

from ficetools import backend, velocity, graphics, utils_funcs

run_golden = args.toml_path_i
run_cornford = args.toml_path_m

n_sens = args.n_sens

# Read each run params
params_budd = conf.ConfigParser(run_golden)
params_corn = conf.ConfigParser(run_cornford)

#Read common data and get mesh information
mesh_in = fice_mesh.get_mesh(params_budd)

# Get mesh triangulation
x, y, t = graphics.read_fenics_ice_mesh(mesh_in)
trim = tri.Triangulation(x, y, t)

el = finiteelement.FiniteElement("Lagrange", mesh_in.ufl_cell(), 1)
mixedElem = el * el

Q = FunctionSpace(mesh_in, mixedElem)
dQ = Function(Q)

#Get the n_sens
num_sens = np.arange(0, params_budd.time.num_sens)
print('To plot Time')
n_zero = num_sens[n_sens[0]]
print(n_zero)

t_sens = np.flip(np.linspace(params_budd.time.run_length, 0, params_budd.time.num_sens))
t_zero = np.round(t_sens[n_sens[0]])
print(t_zero)

# Now lets read the output
# Read output data to plot from BUDD
outdir_B = params_budd.io.output_dir
phase_name_B = params_budd.time.phase_name
run_name_B = params_budd.io.run_name
phase_suffix_B = params_budd.time.phase_suffix

fwd_outdir_B = Path(outdir_B) / phase_name_B / phase_suffix_B
file_qts_B = "_".join((run_name_B + phase_suffix_B, 'dQ_ts.h5'))
hdffile_B = fwd_outdir_B / file_qts_B

assert hdffile_B.is_file(), "File not found"

valpha_first_B, vbeta_first_B = utils_funcs.compute_vertex_for_dV_components(dQ,
                                                                             mesh_in,
                                                                             str(hdffile_B),
                                                                             'dQdalphaXbeta',
                                                                             n_zero,
                                                                             mult_mmatrix=True)
# Read output data to plot from CORN
outdir_C = params_corn.io.output_dir
phase_name_C = params_corn.time.phase_name
run_name_C = params_corn.io.run_name
phase_suffix_C = params_corn.time.phase_suffix

fwd_outdir_C = Path(outdir_C) / phase_name_C / phase_suffix_C
file_qts_C = "_".join((run_name_C + phase_suffix_C, 'dQ_ts.h5'))
hdffile_C = fwd_outdir_C / file_qts_C

assert hdffile_C.is_file(), "File not found"

valpha_first_C, vbeta_first_C = utils_funcs.compute_vertex_for_dV_components(dQ,
                                                                             mesh_in,
                                                                             str(hdffile_C),
                                                                             'dQdalphaXbeta',
                                                                             n_zero,
                                                                             mult_mmatrix=True)

levelsa_B, ticksa_B = graphics.make_colorbar_scale(valpha_first_B, 0.05)
levelsb_B, ticksb_B = graphics.make_colorbar_scale(vbeta_first_B, 0.05)
levelsa_C, ticksa_C = graphics.make_colorbar_scale(valpha_first_C, 0.05)
levelsb_C, ticksb_C = graphics.make_colorbar_scale(vbeta_first_C, 0.05)

tick_options = {'axis':'both','which':'both','bottom':False,
    'top':False,'left':False,'right':False,'labelleft':False, 'labelbottom':False, 'labeltop':False}

rcParams['axes.labelsize'] = 18
rcParams['xtick.labelsize'] = 18
rcParams['ytick.labelsize'] = 18
rcParams['axes.titlesize'] = 18

g = 1.0

fig1 = plt.figure(figsize=(12*g, 10*g))#, constrained_layout=True)
spec = gridspec.GridSpec(2, 2, wspace=0.05, hspace=0.4)

ax0 = plt.subplot(spec[0])
ax0.set_aspect('equal')
ax0.tick_params(**tick_options)
divider = make_axes_locatable(ax0)
cax = divider.append_axes("bottom", size="5%", pad=0.1)
c = ax0.tricontourf(x, y, t, valpha_first_B, levels=levelsa_B, cmap=plt.get_cmap('RdBu'),  extend="both") # levels = levelsa
ax0.triplot(x, y, trim.triangles, '-', color='grey', lw=0.2, alpha=0.5)
cbar = plt.colorbar(c, cax=cax, ticks=ticksa_B, orientation="horizontal", extend="both") #ticks=ticksa
cbar.ax.set_xlabel(r'$\frac{\delta Q}{\delta \alpha}$', fontsize=22)
n_text = AnchoredText('year '+ str(t_zero+1), prop=dict(size=18), frameon=True, loc='upper right')
ax0.add_artist(n_text)
at = AnchoredText('a', prop=dict(size=16), frameon=True, loc='upper left')
ax0.add_artist(at)


ax1 = plt.subplot(spec[1])
ax1.set_aspect('equal')
ax1.tick_params(**tick_options)
divider = make_axes_locatable(ax1)
cax = divider.append_axes("bottom", size="5%", pad=0.1)
c = ax1.tricontourf(x, y, t, vbeta_first_B, levels=levelsb_B, cmap=plt.get_cmap('RdBu'),  extend="both") #levels = levelsb,
ax1.triplot(x, y, trim.triangles, '-', color='grey', lw=0.2, alpha=0.5)
cbar = plt.colorbar(c, cax=cax, ticks=ticksb_B, orientation="horizontal",  extend="both") # ticks=ticksb,
cbar.ax.set_xlabel(r'$\frac{\delta Q}{\delta \beta}$', fontsize=22)
n_text = AnchoredText('year '+ str(t_zero+1), prop=dict(size=18), frameon=True, loc='upper right')
ax1.add_artist(n_text)
at = AnchoredText('b', prop=dict(size=16), frameon=True, loc='upper left')
ax1.add_artist(at)


ax2 = plt.subplot(spec[2])
ax2.set_aspect('equal')
ax2.tick_params(**tick_options)
divider = make_axes_locatable(ax2)
cax = divider.append_axes("bottom", size="5%", pad=0.1)
c = ax2.tricontourf(x, y, t, valpha_first_C, levels=levelsa_C, cmap=plt.get_cmap('RdBu'),  extend="both") #  levels = levelsa,
ax2.triplot(x, y, trim.triangles, '-', color='grey', lw=0.2, alpha=0.5)
cbar = plt.colorbar(c, cax=cax, ticks=ticksa_C, orientation="horizontal",  extend="both") #ticks=ticksa
cbar.ax.set_xlabel(r'$\frac{\delta Q}{\delta \alpha}$', fontsize=22)
n_text = AnchoredText('year '+ str(t_zero+1), prop=dict(size=18), frameon=True, loc='upper right')
ax2.add_artist(n_text)
at = AnchoredText('c', prop=dict(size=16), frameon=True, loc='upper left')
ax2.add_artist(at)

ax3 = plt.subplot(spec[3])
ax3.set_aspect('equal')
ax3.tick_params(**tick_options)
divider = make_axes_locatable(ax3)
cax = divider.append_axes("bottom", size="5%", pad=0.1)
c = ax3.tricontourf(x, y, t, vbeta_first_C, levels=levelsb_C, cmap=plt.get_cmap('RdBu'),  extend="both") # levels = levelsb,
ax3.triplot(x, y, trim.triangles, '-', color='grey', lw=0.2, alpha=0.5)
cbar = plt.colorbar(c, cax=cax, ticks=ticksb_C, orientation="horizontal", extend="both") #ticks=ticksb,
cbar.ax.set_xlabel(r'$\frac{\delta Q}{\delta \beta}$', fontsize=22)
n_text = AnchoredText('year '+ str(t_zero+1), prop=dict(size=18), frameon=True, loc='upper right')
ax3.add_artist(n_text)
at = AnchoredText('d', prop=dict(size=16), frameon=True, loc='upper left')
ax3.add_artist(at)

ax0.title.set_text('Weertman–Budd')
ax2.title.set_text('Cornford')

plt.tight_layout()
plt.savefig(os.path.join(plot_path, 'sliding_sensitivities.png'),
            bbox_inches='tight')