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
import numpy as np
from fenics import *
from fenics_ice import config
from pathlib import Path
from fenics_ice import mesh as fice_mesh

#Plotting imports
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec
import matplotlib.tri as tri
from configobj import ConfigObj
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-conf", type=str,
                    default="../../../config.ini", help="pass config file")
args = parser.parse_args()
config_file = args.conf
configuration = ConfigObj(os.path.expanduser(config_file))

#Load main repository path
MAIN_PATH = configuration['main_path']
sys.path.append(MAIN_PATH)

# Paths to data
plot_path = os.path.join(MAIN_PATH, 'plots/')
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

from meshtools import meshtools

path_output = os.path.join(MAIN_PATH, 'output/04_run_inv_lcurves')

exp_names = ['gamma_alpha', 'gamma_beta', 'delta_beta_gnd']

gamma_alpha = meshtools.get_data_for_experiment(path_output, exp_names[0])
gamma_beta = meshtools.get_data_for_experiment(path_output, exp_names[1])
delta_beta_gnd = meshtools.get_data_for_experiment(path_output, exp_names[2])

run_files = os.path.join(MAIN_PATH,
                         'scripts/run_stages/run_inversion')

toml = os.path.join(run_files, 'smith.toml')

params = config.ConfigParser(toml, top_dir=Path(MAIN_PATH))

#Reading mesh
mesh_in = fice_mesh.get_mesh(params)

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

gamma_alpha_xml = meshtools.get_xml_from_exp(path_output, exp_names[0])
gamma_beta_xml = meshtools.get_xml_from_exp(path_output, exp_names[1])
delta_beta_gnd_xml = meshtools.get_xml_from_exp(path_output, exp_names[2])

print('Check if these are the right files for some of the extreme values')
print(delta_beta_gnd_xml)

# Read alpha extremes output for gamma_alpha exp
alpha_min_xml = gamma_alpha_xml[2]
alpha_max_xml = gamma_alpha_xml[0]

U_alpha_min_xml = gamma_alpha_xml[-1]
U_alpha_max_xml = gamma_alpha_xml[1]

# Read beta extremes output for gamma_beta exp
beta_min_xml = gamma_beta_xml[3]
beta_max_xml = gamma_beta_xml[0]

U_beta_min_xml = gamma_beta_xml[-1]
U_beta_max_xml = gamma_beta_xml[2]

# Read beta extremes output for delta_beta_gnd exp
betagnd_min_xml = delta_beta_gnd_xml[3]
betagnd_max_xml = delta_beta_gnd_xml[0]

U_betagnd_min_xml = delta_beta_gnd_xml[-1]
U_betagnd_max_xml = delta_beta_gnd_xml[2]

# Compute vertex values for each parameter function
# in the mesh
v_alpha_min = meshtools.compute_vertex_for_parameter_field(alpha_min_xml,
                                                           param_space=Qp,
                                                           dg_space=M,
                                                           mesh_in=mesh_in)

v_alpha_max = meshtools.compute_vertex_for_parameter_field(alpha_max_xml,
                                                           param_space=Qp,
                                                           dg_space=M,
                                                           mesh_in=mesh_in)

v_beta_min = meshtools.compute_vertex_for_parameter_field(beta_min_xml,
                                                          param_space=Qp,
                                                          dg_space=M,
                                                          mesh_in=mesh_in)
v_beta_max = meshtools.compute_vertex_for_parameter_field(beta_max_xml,
                                                           param_space=Qp,
                                                          dg_space=M,
                                                          mesh_in=mesh_in)

v_bgnd_min = meshtools.compute_vertex_for_parameter_field(betagnd_min_xml,
                                                          param_space=Qp,
                                                          dg_space=M,
                                                          mesh_in=mesh_in)
v_bgnd_max = meshtools.compute_vertex_for_parameter_field(betagnd_max_xml,
                                                          param_space=Qp,
                                                          dg_space=M,
                                                          mesh_in=mesh_in)

# Getting velocity vertex for extreme values of parameters
v_U_alpha_min = meshtools.compute_vertex_for_velocity_field(U_alpha_min_xml, v_space=V,
                                                           q_space=Q, mesh_in=mesh_in)

v_U_alpha_max = meshtools.compute_vertex_for_velocity_field(U_alpha_max_xml, v_space=V,
                                                           q_space=Q, mesh_in=mesh_in)

v_U_beta_min = meshtools.compute_vertex_for_velocity_field(U_beta_min_xml, v_space=V,
                                                           q_space=Q, mesh_in=mesh_in)

v_U_beta_max = meshtools.compute_vertex_for_velocity_field(U_beta_max_xml, v_space=V,
                                                           q_space=Q, mesh_in=mesh_in)

v_U_bgnd_min = meshtools.compute_vertex_for_velocity_field(U_betagnd_min_xml, v_space=V,
                                                           q_space=Q, mesh_in=mesh_in)

v_U_bgnd_max = meshtools.compute_vertex_for_velocity_field(U_betagnd_max_xml, v_space=V,
                                                           q_space=Q, mesh_in=mesh_in)

# Get mesh triangulation
x = mesh_in.coordinates()[:,0]
y = mesh_in.coordinates()[:,1]
t = mesh_in.cells()
trim = tri.Triangulation(x, y, t)

#Now plotting .. this will be long!
cmap_params = plt.get_cmap('RdBu_r')

g = 1.2
tick_options_lc = {'axis':'both','which':'both','bottom':True,
    'top':False,'left':True,'right':False,'labelleft':True, 'labelbottom':True}

tick_options_mesh = {'axis':'both','which':'both','bottom':False,
    'top':False,'left':False,'right':False,'labelleft':False, 'labelbottom':False}

fig1 = plt.figure(figsize=(14*g, 12*g))#, constrained_layout=True)
spec = gridspec.GridSpec(3, 3, hspace=0.3, wspace=0.2)#, hspace=0.05)

ax0 = plt.subplot(spec[0])
ax0.tick_params(**tick_options_mesh)
meshtools.plot_field_in_contour_plot(x, y, t, v_alpha_min, 'alpha',
                               ax=ax0, vmin=-70, vmax=70, cmap=cmap_params, add_mesh=True)
ax0.set_title("gamma-alpha = "+ str(np.min(gamma_alpha['gamma_alpha'])), fontsize=14)
at = AnchoredText('a', prop=dict(size=14), frameon=True, loc='upper left')
ax0.add_artist(at)

ax1 = plt.subplot(spec[1])
ax1.tick_params(**tick_options_mesh)
meshtools.plot_field_in_contour_plot(x, y, t, v_alpha_max, 'alpha',
                               ax=ax1, vmin=-70, vmax=70, cmap=cmap_params, add_mesh=True)
ax1.set_title("gamma-alpha = "+ str(np.max(gamma_alpha['gamma_alpha'])), fontsize=14)
at = AnchoredText('b', prop=dict(size=14), frameon=True, loc='upper left')
ax1.add_artist(at)


ax2 = plt.subplot(spec[2])
ax2.tick_params(**tick_options_lc)
meshtools.plot_lcurve_scatter(gamma_alpha, 'gamma_alpha', ax=ax2,
                              xlim_min=None, xlim_max=10e12, ylim_min=None,
                              ylim_max=8e6, xytext=(40,5), rot=40)
ax2.set_title("gamma-alpha $\it{L-curve}$", fontsize=14)
at = AnchoredText('c', prop=dict(size=14), frameon=True, loc='upper right')
ax2.add_artist(at)

ax3 = plt.subplot(spec[3])
ax3.tick_params(**tick_options_mesh)
meshtools.plot_field_in_contour_plot(x, y, t, v_beta_min, 'beta',
                                     ax=ax3, vmin=-30, vmax=1000, cmap=cmap_params, add_mesh=True)
ax3.set_title("gamma-beta = " + str(np.min(gamma_beta['gamma_beta'])), fontsize=14)
at = AnchoredText('d', prop=dict(size=14), frameon=True, loc='upper left')
ax3.add_artist(at)

ax4 = plt.subplot(spec[4])
ax4.tick_params(**tick_options_mesh)
meshtools.plot_field_in_contour_plot(x, y, t, v_beta_max, 'beta',
                                     ax=ax4, vmin=-30, vmax=1000, cmap=cmap_params, add_mesh=True)
ax4.set_title("gamma-beta = " + str(np.max(gamma_beta['gamma_beta'])), fontsize=14)
at = AnchoredText('e', prop=dict(size=14), frameon=True, loc='upper left')
ax4.add_artist(at)

ax5 = plt.subplot(spec[5])
ax5.tick_params(**tick_options_lc)
meshtools.plot_lcurve_scatter(gamma_beta, 'gamma_beta', ax=ax5,
                              xlim_min=-10, xlim_max=10e11, ylim_min=None,
                              ylim_max=5e6, xytext=(35,15), rot=0)
ax5.set_title("gamma-beta $\it{L-curve}$", fontsize=14)
at = AnchoredText('f', prop=dict(size=14), frameon=True, loc='upper right')
ax5.add_artist(at)

ax6 = plt.subplot(spec[6])
ax6.tick_params(**tick_options_mesh)
meshtools.plot_field_in_contour_plot(x, y, t, v_bgnd_min, 'beta',
                                     ax=ax6, vmin=-30, vmax=1000, cmap=cmap_params, add_mesh=True)
ax6.set_title("delta-beta-gnd = " + str(np.min(delta_beta_gnd['delta_beta_gnd'])), fontsize=14)
at = AnchoredText('h', prop=dict(size=14), frameon=True, loc='upper left')
ax6.add_artist(at)

ax7 = plt.subplot(spec[7])
ax7.tick_params(**tick_options_mesh)
meshtools.plot_field_in_contour_plot(x, y, t, v_bgnd_max, 'beta',
                                     ax=ax7, vmin=-30, vmax=1000, cmap=cmap_params, add_mesh=True)
ax7.set_title("delta-beta-gnd = " + str(np.max(delta_beta_gnd['delta_beta_gnd'])), fontsize=14)
at = AnchoredText('i', prop=dict(size=14), frameon=True, loc='upper left')
ax7.add_artist(at)

ax8 = plt.subplot(spec[8])
ax8.tick_params(**tick_options_lc)
meshtools.plot_lcurve_scatter(delta_beta_gnd, 'delta_beta_gnd', ax=ax8,
                              xlim_min=None, xlim_max=10e12, ylim_min=None,
                              ylim_max=5e6, xytext=(35,15), rot=35)
ax8.set_title("delta-beta-gnd $\it{L-curve}$", fontsize=14)
at = AnchoredText('j', prop=dict(size=14), frameon=True, loc='upper right')
ax8.add_artist(at)

plt.tight_layout()
plt.savefig(os.path.join(plot_path, 'l-curves_alpha_beta.png'), bbox_inches='tight')

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
meshtools.plot_field_in_contour_plot(x, y, t, v_U_alpha_min, 'U [m/yr]',
                               ax=ax0, vmin=0, vmax=2000, cmap=cmap_params)
ax0.set_title("gamma-alpha = "+str(np.min(gamma_alpha['gamma_alpha'])), fontsize=14)
at = AnchoredText('a', prop=dict(size=14), frameon=True, loc='upper left')
ax0.add_artist(at)

ax1 = plt.subplot(spec[1])
ax1.tick_params(**tick_options_mesh)
meshtools.plot_field_in_contour_plot(x, y, t, v_U_alpha_max, 'U [m/yr]',
                               ax=ax1, vmin=0, vmax=2000, cmap=cmap_params)
ax1.set_title("gamma-alpha = "+ str(np.max(gamma_alpha['gamma_alpha'])), fontsize=14)
at = AnchoredText('b', prop=dict(size=14), frameon=True, loc='upper left')
ax1.add_artist(at)


ax2 = plt.subplot(spec[2])
ax2.tick_params(**tick_options_lc)
meshtools.plot_lcurve_scatter(gamma_alpha, 'gamma_alpha', ax=ax2,
                              xlim_min=None, xlim_max=10e12, ylim_min=None,
                              ylim_max=8e6, xytext=(40,5), rot=40)
ax2.set_title("gamma-alpha $\it{L-curve}$", fontsize=14)
at = AnchoredText('c', prop=dict(size=14), frameon=True, loc='upper right')
ax2.add_artist(at)

ax3 = plt.subplot(spec[3])
ax3.tick_params(**tick_options_mesh)
meshtools.plot_field_in_contour_plot(x, y, t, v_U_beta_min, 'U [m/yr]',
                               ax=ax3, vmin=0, vmax=2000, cmap=cmap_params)
ax3.set_title("gamma-beta = " + str(np.min(gamma_beta['gamma_beta'])), fontsize=14)
at = AnchoredText('d', prop=dict(size=14), frameon=True, loc='upper left')
ax3.add_artist(at)

ax4 = plt.subplot(spec[4])
ax4.tick_params(**tick_options_mesh)
meshtools.plot_field_in_contour_plot(x, y, t, v_U_beta_max, 'U [m/yr]',
                               ax=ax4, vmin=0, vmax=2000, cmap=cmap_params)
ax4.set_title("gamma-beta = " + str(np.max(gamma_beta['gamma_beta'])), fontsize=14)
at = AnchoredText('e', prop=dict(size=14), frameon=True, loc='upper left')
ax4.add_artist(at)

ax5 = plt.subplot(spec[5])
ax5.tick_params(**tick_options_lc)
meshtools.plot_lcurve_scatter(gamma_beta, 'gamma_beta', ax=ax5,
                              xlim_min=-10, xlim_max=10e11, ylim_min=None,
                              ylim_max=5e6, xytext=(35,15), rot=0)
ax5.set_title("gamma-beta $\it{L-curve}$", fontsize=14)
at = AnchoredText('f', prop=dict(size=14), frameon=True, loc='upper right')
ax5.add_artist(at)

ax6 = plt.subplot(spec[6])
ax6.tick_params(**tick_options_mesh)
meshtools.plot_field_in_contour_plot(x, y, t, v_U_bgnd_min, 'U [m/yr]',
                               ax=ax6, vmin=0, vmax=2000, cmap=cmap_params)
ax6.set_title("delta-beta-gnd = "+ str(np.min(delta_beta_gnd['delta_beta_gnd'])), fontsize=14)
at = AnchoredText('h', prop=dict(size=14), frameon=True, loc='upper left')
ax6.add_artist(at)

ax7 = plt.subplot(spec[7])
ax7.tick_params(**tick_options_mesh)
meshtools.plot_field_in_contour_plot(x, y, t, v_U_bgnd_max, 'U [m/yr]',
                               ax=ax7, vmin=0, vmax=2000, cmap=cmap_params)
ax7.set_title("delta-beta-gnd = " + str(np.max(delta_beta_gnd['delta_beta_gnd'])), fontsize=14)
at = AnchoredText('i', prop=dict(size=14), frameon=True, loc='upper left')
ax7.add_artist(at)

ax8 = plt.subplot(spec[8])
ax8.tick_params(**tick_options_lc)
meshtools.plot_lcurve_scatter(delta_beta_gnd, 'delta_beta_gnd', ax=ax8,
                              xlim_min=None, xlim_max=10e12, ylim_min=None,
                              ylim_max=5e6, xytext=(35,15), rot=35)
ax8.set_title("delta-beta-gnd $\it{L-curve}$", fontsize=14)
at = AnchoredText('j', prop=dict(size=14), frameon=True, loc='upper right')
ax8.add_artist(at)

plt.tight_layout()
plt.savefig(os.path.join(plot_path, 'l-curves_velocities.png'), bbox_inches='tight')