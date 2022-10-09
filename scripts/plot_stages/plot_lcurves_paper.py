"""
Plot inversion l-curves results and fields of extremes parameter values
for over-fitting and under-fitting visualization of J_reg.
and a workflow based on the L-curves inflection points

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
import pickle
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
parser.add_argument("-toml_path_lcurve", type=str,
                    default="../run_experiments/run_workflow/smith_cloud.toml",
                    help="pass .toml file")
parser.add_argument("-toml_path_workflow", type=str,
                    default="../run_experiments/run_workflow/smith_cloud.toml",
                    help="pass .toml file")
parser.add_argument("-sub_plot_dir", type=str,
                    default="temp", help="pass sub plot directory to store the plots")
args = parser.parse_args()
config_file = args.conf
configuration = ConfigObj(os.path.expanduser(config_file))

#Load main repository path
MAIN_PATH = configuration['main_path']
sys.path.append(MAIN_PATH)

from ficetools import graphics, utils_funcs

# Paths to data
sub_plot_dir = args.sub_plot_dir
plot_path = os.path.join(MAIN_PATH, 'plots/'+ sub_plot_dir)
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

tomlf_lcurve = args.toml_path_lcurve
params_lcurve = conf.ConfigParser(tomlf_lcurve)
path_output = os.path.join(params_lcurve.io.output_dir, 'inversion')

tomlf_workflow = args.toml_path_workflow
params_workflow = conf.ConfigParser(tomlf_workflow)

## Get inv sigma files
exp_outdir_fwd = utils_funcs.define_stage_output_paths(params_workflow, 'time')
exp_outdir_errp = utils_funcs.define_stage_output_paths(params_workflow, 'error_prop')

#Config five
fnames = utils_funcs.get_file_names_for_path_plot(params_workflow)
Q_fname = fnames[0]
sigma_fname = fnames[1]
sigma_prior_fname = fnames[2]
print(Q_fname)
print(sigma_fname)
print(sigma_prior_fname)

Qfile = exp_outdir_fwd / Q_fname
sigmafile = exp_outdir_errp / sigma_fname
sigmapriorfile = exp_outdir_errp / sigma_prior_fname

assert Qfile.is_file()
assert sigmafile.is_file()
assert sigmapriorfile.is_file()

### Reading QoI path and sigma QoI
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

s = 2*sigma_interp
sp = 2*sigma_prior_interp

x_qoi = dQ_t
y_qoi = dQ_vals - dQ_vals[0]

# Set parameter range for gamma-alpha and gamma-beta exp.
param_min = 1e-04
param_max = 1e+04

# Compute round steps for each
steps = int(np.log10(param_max) - np.log10(param_min)) + 1

# Make a list of the parameters values
param_range = np.geomspace(param_min, param_max,
                           steps, dtype=np.float64).tolist()

name_suff = ['gamma_alpha', 'gamma_beta']

exp_names = []

for i in range(steps):
    phase_suffix_name = utils_funcs.composeFileName(name_suff[0], param_range[i])
    exp_names.append(phase_suffix_name)
for i in range(steps):
    phase_suffix_name = utils_funcs.composeFileName(name_suff[1], param_range[i])
    exp_names.append(phase_suffix_name)

j_paths = []
for exp in exp_names:
    j_path = utils_funcs.get_path_for_experiment(path_output, exp)
    j_paths.append(j_path)

gamma_alpha = utils_funcs.get_data_for_experiment(j_paths[0:9])
gamma_alpha.to_csv(os.path.join(plot_path, 'gamma_alpha_lcurve.csv'))
gamma_beta = utils_funcs.get_data_for_experiment(j_paths[9:9+9])
gamma_beta.to_csv(os.path.join(plot_path, 'gamma_beta_lcurve.csv'))


#Reading mesh
mesh_in = fice_mesh.get_mesh(params_lcurve)
x, y, t = graphics.read_fenics_ice_mesh(mesh_in)

# Compute the function spaces from the Mesh
Q = FunctionSpace(mesh_in, 'Lagrange',1)
Qh = FunctionSpace(mesh_in, 'Lagrange',3)
M = FunctionSpace(mesh_in, 'DG', 0)

if not params_lcurve.mesh.periodic_bc:
    Qp = Q
    V = VectorFunctionSpace(mesh_in,'Lagrange', 1, dim=2)
else:
    Qp = fice_mesh.get_periodic_space(params_lcurve, mesh_in, dim=1)
    V =  fice_mesh.get_periodic_space(params_lcurve, mesh_in, dim=2)

path_diag = os.path.join(params_lcurve.io.diagnostics_dir, 'inversion')

alpha_min_xml = utils_funcs.get_xml_from_exp(path_diag, exp_names[0], 'alpha', '1E-4')
alpha_max_xml = utils_funcs.get_xml_from_exp(path_diag, exp_names[8], 'alpha', '1E+4')
beta_min_xml = utils_funcs.get_xml_from_exp(path_diag, exp_names[9], 'beta', '1E-4')
beta_max_xml = utils_funcs.get_xml_from_exp(path_diag, exp_names[17], 'beta', '1E+4')

print('Check if these are the right files for some of the extreme values')
assert os.path.exists(alpha_min_xml)
assert os.path.exists(alpha_max_xml)
assert os.path.exists(beta_min_xml)
assert os.path.exists(beta_max_xml)
print(beta_min_xml)
print(beta_max_xml)

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

#Now plotting .. this will be long!
# Setting color scale levels
#min_alpha = np.min(v_alpha_max)
#max_alpha = np.max(v_alpha_max)
min_alpha = 6
max_alpha = 40

#min_beta = np.min(v_beta_max)
#max_beta = np.max(v_beta_max)

min_beta = 400
max_beta = 800

g = 1.2
tick_options_lc = {'axis':'both','which':'both','bottom':True,
    'top':False,'left':True,'right':False,'labelleft':True, 'labelbottom':True}

tick_options_mesh = {'axis':'both','which':'both','bottom':False,
    'top':False,'left':False,'right':False,'labelleft':False, 'labelbottom':False}

colors = sns.color_palette()

rcParams['axes.labelsize'] = 15
rcParams['xtick.labelsize'] = 15
rcParams['ytick.labelsize'] = 15
rcParams['axes.titlesize'] = 15

cmap_params_alpha = plt.cm.get_cmap('YlOrBr')
cmap_params_bglen = plt.cm.get_cmap('YlGnBu')

fig1 = plt.figure(figsize=(15*g, 7*g))#, constrained_layout=True)
spec = gridspec.GridSpec(2, 4, hspace=0.4, wspace=0.55, width_ratios=[1.5, 1.5, 1.3, 1.3])

ax0 = plt.subplot(spec[0])
ax0.tick_params(**tick_options_mesh)
graphics.plot_field_in_contour_plot(x, y, t, v_alpha_min, r'Sliding parameter $\alpha$',
                               ax=ax0, vmin=min_alpha, vmax=max_alpha, cmap=cmap_params_alpha, add_mesh=True)
ax0.set_title(r'$\gamma_{\alpha} = $'+ str(np.min(gamma_alpha['gamma_alpha'])), fontsize=14)
at = AnchoredText('a', prop=dict(size=14), frameon=True, loc='upper left')
ax0.add_artist(at)

ax1 = plt.subplot(spec[1])
ax1.tick_params(**tick_options_mesh)
graphics.plot_field_in_contour_plot(x, y, t, v_alpha_max, r'Sliding parameter $\alpha$',
                               ax=ax1, vmin=min_alpha, vmax=max_alpha, cmap=cmap_params_alpha, add_mesh=True)
ax1.set_title(r'$\gamma_{\alpha} = $'+ str(np.max(gamma_alpha['gamma_alpha'])), fontsize=14)
at = AnchoredText('b', prop=dict(size=14), frameon=True, loc='upper left')
ax1.add_artist(at)

ax2 = plt.subplot(spec[2])
ax2.tick_params(**tick_options_lc)
graphics.plot_lcurve_scatter(gamma_alpha, 'gamma_alpha', ax=ax2,
                              xlim_min=None, xlim_max=10e9, ylim_min=None,
                              ylim_max=None, xytext=(40,5), rot=30)
ax2.set_title(r'$\gamma_{\alpha}  \it{L-curve}$', fontsize=14)
at = AnchoredText('c', prop=dict(size=14), frameon=True, loc='upper right')
ax2.add_artist(at)

ax3 = plt.subplot(spec[3])
ax3.plot(x_qoi, y_qoi, color=colors[3], label='QoI projection')
ax3.fill_between(x_qoi, y_qoi-s, y_qoi+s, facecolor=colors[3], alpha=0.3)
ax3.set_xlabel('Time (yrs)')
ax3.set_ylabel(r'$QoI: VAF$ $(m^3)$')
ax3.legend(loc='lower left')
at = AnchoredText('d', prop=dict(size=14), frameon=True, loc='upper right')
ax3.add_artist(at)

ax4 = plt.subplot(spec[4])
ax4.tick_params(**tick_options_mesh)
graphics.plot_field_in_contour_plot(x, y, t, v_beta_min, r'Ice stiffness parameter $\beta$',
                                     ax=ax4, vmin=min_beta, vmax=max_beta, cmap=cmap_params_bglen, add_mesh=True)
ax4.set_title(r'$\gamma_{\beta} = $' + str(np.min(gamma_beta['gamma_beta'])), fontsize=14)
at = AnchoredText('e', prop=dict(size=14), frameon=True, loc='upper left')
ax4.add_artist(at)

ax5 = plt.subplot(spec[5])
ax5.tick_params(**tick_options_mesh)
graphics.plot_field_in_contour_plot(x, y, t, v_beta_max, r'Ice stiffness parameter $\beta$',
                                     ax=ax5, vmin=min_beta, vmax=max_beta, cmap=cmap_params_bglen, add_mesh=True)
ax5.set_title(r'$\gamma_{\beta} = $' + str(np.max(gamma_beta['gamma_beta'])), fontsize=14)
at = AnchoredText('f', prop=dict(size=14), frameon=True, loc='upper left')
ax5.add_artist(at)

ax6 = plt.subplot(spec[6])
ax6.tick_params(**tick_options_lc)
graphics.plot_lcurve_scatter(gamma_beta, 'gamma_beta', ax=ax6,
                              xlim_min=None, xlim_max=10e9, ylim_min=None,
                              ylim_max=None, xytext=(45,1), rot=30)
ax6.set_title(r'$\gamma_{\beta}  \it{L-curve}$', fontsize=14)
at = AnchoredText('g', prop=dict(size=14), frameon=True, loc='upper right')
ax6.add_artist(at)

ax7 = plt.subplot(spec[7])
ax7.tick_params(**tick_options_lc)
ax7.semilogy(x_qoi, sp, color=colors[3], linestyle='dashed', label='prior')
ax7.semilogy(x_qoi, s, color=colors[3], label='posterior')
ax7.legend()
ax7.set_xlabel('Time (yrs)')
ax7.set_ylabel(r'$\sigma_{QoI}$ $(m^3)$')
at = AnchoredText('h', prop=dict(size=14), frameon=True, loc='upper left')
ax7.add_artist(at)

plt.tight_layout()
plt.savefig(os.path.join(plot_path, 'l-curves_alpha_beta_sigma_wfl.png'), bbox_inches='tight')

