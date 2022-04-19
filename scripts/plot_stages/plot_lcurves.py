"""
Plot inversion l-curves results and fields of extremes parameter values
for over-fitting and under-fitting visualization of J_reg.

- Reads the input mesh
- Reads output data for each field (this can be any field output
from the inversion stored as a .xml e.g. alpha, beta, B_glen)
- Reads the .csv files of J terms for each parameter value and append
all the results from the sensitivity exp in a single pandas dataframe
- Plots things in a multi-plot grid.

@authors: Fenics_ice contributors
"""
import sys
import os
import salem
from pathlib import Path
import numpy as np
from fenics import *
from fenics_ice import config as conf
from fenics_ice import mesh as fice_mesh

#Plotting imports
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from configobj import ConfigObj
import argparse

# Load configuration file for more order in paths
parser = argparse.ArgumentParser()
parser.add_argument("-conf", type=str, default="../../../config.ini", help="pass config file")
parser.add_argument("-toml_path", type=str,
                    default="../run_experiments/run_workflow/smith_cloud.toml",
                    help="pass .toml file")
parser.add_argument("-sub_plot_dir", type=str,
                    default="temp", help="pass sub plot directory to store the plots")
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

from ficetools import graphics, utils_funcs

# Paths to data
sub_plot_dir = args.sub_plot_dir
plot_path = os.path.join(MAIN_PATH, 'plots/'+ sub_plot_dir)
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

tomlf = args.toml_path
params = conf.ConfigParser(tomlf, top_dir=Path(MAIN_PATH))

path_output = os.path.join(params.io.output_dir, 'inversion')

# Set parameter range for gamma-alpha and gamma-beta exp.
param_min = 1e-04
param_max = 1e+04

# Set parameter range for delta-beta-gnd exp
param_min_dbg = 1e-07
param_max_dbg = 1e+01

# Compute round steps for each
steps = int(np.log10(param_max) - np.log10(param_min)) + 1
steps_dbg = int(np.log10(param_max_dbg) - np.log10(param_min_dbg)) + 1

# Make a list of the parameters values
param_range = np.geomspace(param_min, param_max,
                           steps, dtype=np.float64).tolist()
param_range_dbg = np.geomspace(param_min_dbg,
                               param_max_dbg,
                               steps_dbg, dtype=np.float64).tolist()

name_suff = ['gamma_alpha', 'gamma_beta', 'delta_beta_gnd']

exp_names = []

for i in range(steps):
    phase_suffix_name = utils_funcs.composeFileName(name_suff[0], param_range[i])
    exp_names.append(phase_suffix_name)
for i in range(steps):
    phase_suffix_name = utils_funcs.composeFileName(name_suff[1], param_range[i])
    exp_names.append(phase_suffix_name)
for i in range(steps_dbg):
    phase_suffix_name = utils_funcs.composeFileName(name_suff[2], param_range_dbg[i])
    exp_names.append(phase_suffix_name)

j_paths = []
for exp in exp_names:
    j_path = utils_funcs.get_path_for_experiment(path_output, exp)
    j_paths.append(j_path)

gamma_alpha = utils_funcs.get_data_for_experiment(j_paths[0:9])
gamma_beta = utils_funcs.get_data_for_experiment(j_paths[9:9+9])
delta_beta_gnd = utils_funcs.get_data_for_experiment(j_paths[9+9:len(j_paths)])

#Reading mesh
mesh_in = fice_mesh.get_mesh(params)
x, y, t = graphics.read_fenics_ice_mesh(mesh_in)

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

path_diag = os.path.join(params.io.diagnostics_dir, 'inversion')

alpha_min_xml = utils_funcs.get_xml_from_exp(path_diag, exp_names[0], 'alpha', '1E-4')
alpha_max_xml = utils_funcs.get_xml_from_exp(path_diag, exp_names[8], 'alpha', '1E+4')
beta_min_xml = utils_funcs.get_xml_from_exp(path_diag, exp_names[9], 'beta', '1E-4')
beta_max_xml = utils_funcs.get_xml_from_exp(path_diag, exp_names[17], 'beta', '1E+4')
betagnd_min_xml = utils_funcs.get_xml_from_exp(path_diag, exp_names[18], 'beta', '1E-7')
betagnd_max_xml = utils_funcs.get_xml_from_exp(path_diag, exp_names[26], 'beta', '1E+1')

print('Check if these are the right files for some of the extreme values')
assert os.path.exists(betagnd_min_xml)
assert os.path.exists(betagnd_max_xml)
print(betagnd_min_xml)
print(betagnd_max_xml)

U_alpha_min_xml = utils_funcs.get_xml_from_exp(path_diag, exp_names[0], 'U', '1E-4')
U_alpha_max_xml = utils_funcs.get_xml_from_exp(path_diag, exp_names[8], 'U', '1E+4')

U_beta_min_xml = utils_funcs.get_xml_from_exp(path_diag, exp_names[9], 'U', '1E-4')
U_beta_max_xml = utils_funcs.get_xml_from_exp(path_diag, exp_names[17], 'U', '1E+4')

U_betagnd_min_xml = utils_funcs.get_xml_from_exp(path_diag, exp_names[18], 'U', '1E-7')
U_betagnd_max_xml = utils_funcs.get_xml_from_exp(path_diag, exp_names[26], 'U', '1E+1')

print('Check if these are the right files for some of the extreme values')
assert os.path.exists(U_betagnd_min_xml)
assert os.path.exists(U_betagnd_max_xml)
print(U_betagnd_min_xml)
print(U_betagnd_max_xml)

# Compute vertex values for each parameter function
# in the mesh
v_alpha_min = utils_funcs.compute_vertex_for_parameter_field(alpha_min_xml,
                                                           param_space=Qp, dg_space=M, mesh_in=mesh_in)
v_alpha_max = utils_funcs.compute_vertex_for_parameter_field(alpha_max_xml,
                                                           param_space=Qp, dg_space=M, mesh_in=mesh_in)

v_beta_min = utils_funcs.compute_vertex_for_parameter_field(beta_min_xml,
                                                           param_space=Qp, dg_space=M, mesh_in=mesh_in)
v_beta_max = utils_funcs.compute_vertex_for_parameter_field(beta_max_xml,
                                                           param_space=Qp, dg_space=M, mesh_in=mesh_in)

v_bgnd_min = utils_funcs.compute_vertex_for_parameter_field(betagnd_min_xml,
                                                           param_space=Qp, dg_space=M, mesh_in=mesh_in)
v_bgnd_max = utils_funcs.compute_vertex_for_parameter_field(betagnd_max_xml,
                                                           param_space=Qp, dg_space=M, mesh_in=mesh_in)

# Getting velocity vertex for extreme values of parameters
v_U_alpha_min = utils_funcs.compute_vertex_for_velocity_field(U_alpha_min_xml, v_space=V,
                                                           q_space=Q, mesh_in=mesh_in)

v_U_alpha_max = utils_funcs.compute_vertex_for_velocity_field(U_alpha_max_xml, v_space=V,
                                                           q_space=Q, mesh_in=mesh_in)

v_U_beta_min = utils_funcs.compute_vertex_for_velocity_field(U_beta_min_xml, v_space=V,
                                                           q_space=Q, mesh_in=mesh_in)

v_U_beta_max = utils_funcs.compute_vertex_for_velocity_field(U_beta_max_xml, v_space=V,
                                                           q_space=Q, mesh_in=mesh_in)

v_U_bgnd_min = utils_funcs.compute_vertex_for_velocity_field(U_betagnd_min_xml, v_space=V,
                                                           q_space=Q, mesh_in=mesh_in)

v_U_bgnd_max = utils_funcs.compute_vertex_for_velocity_field(U_betagnd_max_xml, v_space=V,
                                                           q_space=Q, mesh_in=mesh_in)


#Now plotting .. this will be long!
cmap_params = sns.color_palette("RdBu_r", as_cmap=True)

# Setting color scale levels
min_alpha = np.min(v_alpha_max)
max_alpha = np.max(v_alpha_max)

min_beta = np.min(v_beta_max)
max_beta = np.max(v_beta_max)

min_vel = 0
max_vel = 2000

g = 1.2
tick_options_lc = {'axis':'both','which':'both','bottom':True,
    'top':False,'left':True,'right':False,'labelleft':True, 'labelbottom':True}

tick_options_mesh = {'axis':'both','which':'both','bottom':False,
    'top':False,'left':False,'right':False,'labelleft':False, 'labelbottom':False}

fig1 = plt.figure(figsize=(14*g, 12*g))#, constrained_layout=True)
spec = gridspec.GridSpec(3, 3, hspace=0.3, wspace=0.35)#, hspace=0.05)

ax0 = plt.subplot(spec[0])
ax0.tick_params(**tick_options_mesh)
graphics.plot_field_in_contour_plot(x, y, t, v_alpha_min, 'alpha',
                               ax=ax0, vmin=min_alpha, vmax=max_alpha, cmap=cmap_params, add_mesh=True)
ax0.set_title("gamma-alpha = "+ str(np.min(gamma_alpha['gamma_alpha'])), fontsize=14)
at = AnchoredText('a', prop=dict(size=14), frameon=True, loc='upper left')
ax0.add_artist(at)

ax1 = plt.subplot(spec[1])
ax1.tick_params(**tick_options_mesh)
graphics.plot_field_in_contour_plot(x, y, t, v_alpha_max, 'alpha',
                               ax=ax1, vmin=min_alpha, vmax=max_alpha, cmap=cmap_params, add_mesh=True)
ax1.set_title("gamma-alpha = "+ str(np.max(gamma_alpha['gamma_alpha'])), fontsize=14)
at = AnchoredText('b', prop=dict(size=14), frameon=True, loc='upper left')
ax1.add_artist(at)

ax2 = plt.subplot(spec[2])
ax2.tick_params(**tick_options_lc)
graphics.plot_lcurve_scatter(gamma_alpha, 'gamma_alpha', ax=ax2,
                              xlim_min=None, xlim_max=10e12, ylim_min=None,
                              ylim_max=3e6, xytext=(40,5), rot=40)
ax2.set_title("gamma-alpha $\it{L-curve}$", fontsize=14)
at = AnchoredText('c', prop=dict(size=14), frameon=True, loc='upper right')
ax2.add_artist(at)

ax3 = plt.subplot(spec[3])
ax3.tick_params(**tick_options_mesh)
graphics.plot_field_in_contour_plot(x, y, t, v_beta_min, 'beta',
                                     ax=ax3, vmin=min_beta, vmax=max_beta, cmap=cmap_params, add_mesh=True)
ax3.set_title("gamma-beta = " + str(np.min(gamma_beta['gamma_beta'])), fontsize=14)
at = AnchoredText('d', prop=dict(size=14), frameon=True, loc='upper left')
ax3.add_artist(at)

ax4 = plt.subplot(spec[4])
ax4.tick_params(**tick_options_mesh)
graphics.plot_field_in_contour_plot(x, y, t, v_beta_max, 'beta',
                                     ax=ax4, vmin=min_beta, vmax=max_beta, cmap=cmap_params, add_mesh=True)
ax4.set_title("gamma-beta = " + str(np.max(gamma_beta['gamma_beta'])), fontsize=14)
at = AnchoredText('e', prop=dict(size=14), frameon=True, loc='upper left')
ax4.add_artist(at)

ax5 = plt.subplot(spec[5])
ax5.tick_params(**tick_options_lc)
graphics.plot_lcurve_scatter(gamma_beta, 'gamma_beta', ax=ax5,
                              xlim_min=-10, xlim_max=10e8, ylim_min=None,
                              ylim_max=9e5, xytext=(35,15), rot=0)
ax5.set_title("gamma-beta $\it{L-curve}$", fontsize=14)
at = AnchoredText('f', prop=dict(size=14), frameon=True, loc='upper right')
ax5.add_artist(at)

ax6 = plt.subplot(spec[6])
ax6.tick_params(**tick_options_mesh)
graphics.plot_field_in_contour_plot(x, y, t, v_bgnd_min, 'beta',
                                     ax=ax6, vmin=min_beta, vmax=max_beta, cmap=cmap_params, add_mesh=True)
ax6.set_title("delta-beta-gnd = " + str(np.min(delta_beta_gnd['delta_beta_gnd'])), fontsize=14)
at = AnchoredText('h', prop=dict(size=14), frameon=True, loc='upper left')
ax6.add_artist(at)

ax7 = plt.subplot(spec[7])
ax7.tick_params(**tick_options_mesh)
graphics.plot_field_in_contour_plot(x, y, t, v_bgnd_max, 'beta',
                                     ax=ax7, vmin=min_beta, vmax=max_beta, cmap=cmap_params, add_mesh=True)
ax7.set_title("delta-beta-gnd = " + str(np.max(delta_beta_gnd['delta_beta_gnd'])), fontsize=14)
at = AnchoredText('i', prop=dict(size=14), frameon=True, loc='upper left')
ax7.add_artist(at)

ax8 = plt.subplot(spec[8])
ax8.tick_params(**tick_options_lc)
graphics.plot_lcurve_scatter(delta_beta_gnd, 'delta_beta_gnd', ax=ax8,
                              xlim_min=None, xlim_max=10e12, ylim_min=None,
                              ylim_max=6e5, xytext=(35,15), rot=35)
ax8.set_title("delta-beta-gnd $\it{L-curve}$", fontsize=14)
at = AnchoredText('j', prop=dict(size=14), frameon=True, loc='upper right')
ax8.add_artist(at)

plt.tight_layout()
plt.savefig(os.path.join(plot_path, 'l-curves_alpha_beta.png'), bbox_inches='tight')
#
#Now plotting velocities.. this will be long!
cmap_params = plt.get_cmap('viridis')

g = 1.2
tick_options_lc = {'axis':'both','which':'both','bottom':True,
    'top':False,'left':True,'right':False,'labelleft':True, 'labelbottom':True}

tick_options_mesh = {'axis':'both','which':'both','bottom':False,
    'top':False,'left':False,'right':False,'labelleft':False, 'labelbottom':False}

fig2 = plt.figure(figsize=(14*g, 12*g))#, constrained_layout=True)
spec = gridspec.GridSpec(3, 3, hspace=0.3, wspace=0.2)#, hspace=0.05)

ax0 = plt.subplot(spec[0])
ax0.tick_params(**tick_options_mesh)
graphics.plot_field_in_contour_plot(x, y, t, v_U_alpha_min, 'U [m/yr]',
                               ax=ax0, vmin=0, vmax=2000, cmap=cmap_params)
ax0.set_title("gamma-alpha = "+str(np.min(gamma_alpha['gamma_alpha'])), fontsize=14)
at = AnchoredText('a', prop=dict(size=14), frameon=True, loc='upper left')
ax0.add_artist(at)

ax1 = plt.subplot(spec[1])
ax1.tick_params(**tick_options_mesh)
graphics.plot_field_in_contour_plot(x, y, t, v_U_alpha_max, 'U [m/yr]',
                               ax=ax1, vmin=0, vmax=2000, cmap=cmap_params)
ax1.set_title("gamma-alpha = "+ str(np.max(gamma_alpha['gamma_alpha'])), fontsize=14)
at = AnchoredText('b', prop=dict(size=14), frameon=True, loc='upper left')
ax1.add_artist(at)


ax2 = plt.subplot(spec[2])
ax2.tick_params(**tick_options_lc)
graphics.plot_lcurve_scatter(gamma_alpha, 'gamma_alpha', ax=ax2,
                              xlim_min=None, xlim_max=10e12, ylim_min=None,
                              ylim_max=3e6, xytext=(40,5), rot=40)
ax2.set_title("gamma-alpha $\it{L-curve}$", fontsize=14)
at = AnchoredText('c', prop=dict(size=14), frameon=True, loc='upper right')
ax2.add_artist(at)

ax3 = plt.subplot(spec[3])
ax3.tick_params(**tick_options_mesh)
graphics.plot_field_in_contour_plot(x, y, t, v_U_beta_min, 'U [m/yr]',
                               ax=ax3, vmin=0, vmax=2000, cmap=cmap_params)
ax3.set_title("gamma-beta = " + str(np.min(gamma_beta['gamma_beta'])), fontsize=14)
at = AnchoredText('d', prop=dict(size=14), frameon=True, loc='upper left')
ax3.add_artist(at)

ax4 = plt.subplot(spec[4])
ax4.tick_params(**tick_options_mesh)
graphics.plot_field_in_contour_plot(x, y, t, v_U_beta_max, 'U [m/yr]',
                               ax=ax4, vmin=0, vmax=2000, cmap=cmap_params)
ax4.set_title("gamma-beta = " + str(np.max(gamma_beta['gamma_beta'])), fontsize=14)
at = AnchoredText('e', prop=dict(size=14), frameon=True, loc='upper left')
ax4.add_artist(at)

ax5 = plt.subplot(spec[5])
ax5.tick_params(**tick_options_lc)
graphics.plot_lcurve_scatter(gamma_beta, 'gamma_beta', ax=ax5,
                             xlim_min=-10, xlim_max=10e8, ylim_min=None,
                             ylim_max=9e5, xytext=(35, 15), rot=0)
ax5.set_title("gamma-beta $\it{L-curve}$", fontsize=14)
at = AnchoredText('f', prop=dict(size=14), frameon=True, loc='upper right')
ax5.add_artist(at)

ax6 = plt.subplot(spec[6])
ax6.tick_params(**tick_options_mesh)
graphics.plot_field_in_contour_plot(x, y, t, v_U_bgnd_min, 'U [m/yr]',
                               ax=ax6, vmin=0, vmax=2000, cmap=cmap_params)
ax6.set_title("delta-beta-gnd = "+ str(np.min(delta_beta_gnd['delta_beta_gnd'])), fontsize=14)
at = AnchoredText('h', prop=dict(size=14), frameon=True, loc='upper left')
ax6.add_artist(at)

ax7 = plt.subplot(spec[7])
ax7.tick_params(**tick_options_mesh)
graphics.plot_field_in_contour_plot(x, y, t, v_U_bgnd_max, 'U [m/yr]',
                               ax=ax7, vmin=0, vmax=2000, cmap=cmap_params)
ax7.set_title("delta-beta-gnd = " + str(np.max(delta_beta_gnd['delta_beta_gnd'])), fontsize=14)
at = AnchoredText('i', prop=dict(size=14), frameon=True, loc='upper left')
ax7.add_artist(at)

ax8 = plt.subplot(spec[8])
ax8.tick_params(**tick_options_lc)
graphics.plot_lcurve_scatter(delta_beta_gnd, 'delta_beta_gnd', ax=ax8,
                             xlim_min=None, xlim_max=10e12, ylim_min=None,
                             ylim_max=6e5, xytext=(35, 15), rot=35)
ax8.set_title("delta-beta-gnd $\it{L-curve}$", fontsize=14)
at = AnchoredText('j', prop=dict(size=14), frameon=True, loc='upper right')
ax8.add_artist(at)

plt.tight_layout()
plt.savefig(os.path.join(plot_path, 'l-curves_velocities.png'), bbox_inches='tight')