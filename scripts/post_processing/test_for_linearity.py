import sys
import os
import salem
from pathlib import Path
import numpy as np
from fenics import *
from fenics_ice import config as conf
from fenics_ice import mesh as fice_mesh
import argparse

#Plotting imports
from matplotlib import pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from configobj import ConfigObj
from collections import defaultdict
from decimal import Decimal


# Load configuration file for more order in paths
parser = argparse.ArgumentParser()
parser.add_argument("-conf", type=str, default="../../../config.ini", help="pass config file")
parser.add_argument("-toml_path_workflow_i", type=str,
                    default="../run_experiments/run_workflow/smith_cloud.toml",
                    help="pass .toml file")
parser.add_argument("-toml_path_workflow_m", type=str,
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

# Toml for itslive lcurve workflow
tomlf_workflow = args.toml_path_workflow_i
params_workflow = conf.ConfigParser(tomlf_workflow)

# Toml for Measures lcurve workflow
tomlf_workflow_m =  args.toml_path_workflow_m
params_workflow_m = conf.ConfigParser(tomlf_workflow_m)

#Reading mesh
mesh_in = fice_mesh.get_mesh(params_workflow)
x, y, t = graphics.read_fenics_ice_mesh(mesh_in)

# Compute the function spaces from the Mesh
Q = FunctionSpace(mesh_in, 'Lagrange',1)
Qh = FunctionSpace(mesh_in, 'Lagrange',3)
M = FunctionSpace(mesh_in, 'DG', 0)

if not params_workflow.mesh.periodic_bc:
    Qp = Q
    V = VectorFunctionSpace(mesh_in,'Lagrange', 1, dim=2)
else:
    Qp = fice_mesh.get_periodic_space(params_workflow, mesh_in, dim=1)
    V =  fice_mesh.get_periodic_space(params_workflow, mesh_in, dim=2)

# Now we read Alpha and Beta fields from each run
# The diagnostic path is the same for both runs
path_diag = os.path.join(params_workflow.io.diagnostics_dir, 'inversion')

phase_suffix_il = params_workflow.inversion.phase_suffix

exp_outdir_il = Path(path_diag) / phase_suffix_il

file_alpha_il = "_".join((params_workflow.io.run_name+phase_suffix_il, 'alpha.xml'))
file_bglen_il = "_".join((params_workflow.io.run_name+phase_suffix_il, 'beta.xml'))

alpha_il = exp_outdir_il / file_alpha_il
bglen_il = exp_outdir_il / file_bglen_il

assert alpha_il.is_file(), "File not found"
assert bglen_il.is_file(), "File not found"

# For measures
phase_suffix_me = params_workflow_m.inversion.phase_suffix

exp_outdir_me = Path(path_diag) / phase_suffix_me

file_alpha_me = "_".join((params_workflow_m.io.run_name+phase_suffix_me, 'alpha.xml'))
file_bglen_me = "_".join((params_workflow_m.io.run_name+phase_suffix_me, 'beta.xml'))

alpha_me = exp_outdir_me / file_alpha_me
bglen_me = exp_outdir_me / file_bglen_me

assert alpha_me.is_file(), "File not found"
assert bglen_me.is_file(), "File not found"

# Define function spaces for alpha only and uv_comp
alpha_live = Function(Qp, str(alpha_il))
alpha_measures = Function(Qp, str(alpha_me))

alpha_ilp = project(alpha_live, M)
alpha_mep = project(alpha_measures, M)

alpha_v_il = alpha_ilp.compute_vertex_values(mesh_in)
alpha_v_me = alpha_mep.compute_vertex_values(mesh_in)

# Beta space
beta_il = Function(Qp, str(bglen_il))
beta_me = Function(Qp, str(bglen_me))

beta_ilp = project(beta_il, M)
beta_mep = project(beta_me, M)

beta_v_il = beta_ilp.compute_vertex_values(mesh_in)
beta_v_me = beta_mep.compute_vertex_values(mesh_in)

## Now we read forward runs results to get the sensitivities

# Read output data to plot
# Same output dir for both runs
out_il = params_workflow.io.output_dir
phase_name = params_workflow.time.phase_name
run_name = params_workflow.io.run_name

fwd_outdir_il = Path(out_il) / phase_name / phase_suffix_il
fwd_outdir_me = Path(out_il) / phase_name / phase_suffix_me

file_qts_il = "_".join((params_workflow.io.run_name+phase_suffix_il, 'dQ_ts.h5'))
file_qts_me = "_".join((params_workflow_m.io.run_name+phase_suffix_me, 'dQ_ts.h5'))

hdffile_il = fwd_outdir_il / file_qts_il
hdffile_me = fwd_outdir_me / file_qts_me

assert hdffile_il.is_file(), "File not found"
assert hdffile_me.is_file(), "File not found"

from ufl import finiteelement

el = finiteelement.FiniteElement("Lagrange", mesh_in.ufl_cell(), 1)
mixedElem = el * el

Q_f = FunctionSpace(mesh_in, mixedElem)
dQ_f = Function(Q_f)

#Get the n_sens
num_sens = np.arange(0, params_workflow.time.num_sens)
print('Years')
n_zero = num_sens[0]
n_last = num_sens[-1]
print(num_sens)

# Get sensitivity output for dQ/da and dQ/db
# along each num_sen
fwd_v_alpha_il = defaultdict(list)
fwd_v_beta_il = defaultdict(list)
fwd_v_alpha_me = defaultdict(list)
fwd_v_beta_me = defaultdict(list)

results_dot_alpha = []
results_dot_beta = []

# So we know to which num_sen the sensitivity
# output belongs
nametosum = 'fwd_n_'

for n in num_sens:

    # We get the sensitivities for every num_sen
    # as a vector component
    # For measures
    dq_dalpha_me, dq_dbeta_me = utils_funcs.compute_vertex_for_dV_components(dQ_f,
                                                                       mesh_in,
                                                                       str(hdffile_me),
                                                                       'dQdalphaXbeta',
                                                                       n,
                                                                       mult_mmatrix=False)
    # We do not need to multiply by the mass matrix! more info check the compute_vertex_for_dV_component()

    dq_dalpha_il, dq_dbeta_il = utils_funcs.compute_vertex_for_dV_components(dQ_f,
                                                                       mesh_in,
                                                                       str(hdffile_il),
                                                                       'dQdalphaXbeta',
                                                                       n,
                                                                       mult_mmatrix=False)


    dot_alpha_il = np.dot(dq_dalpha_il, alpha_v_il - alpha_v_me)
    dot_beta_il = np.dot(dq_dbeta_il, beta_v_il - beta_v_me)

    print('%.2E' % Decimal(dot_alpha_il))
    print('%.2E' % Decimal(dot_beta_il))

    fwd_v_alpha_me[f'{nametosum}{n}'].append(dq_dalpha_me)
    fwd_v_beta_me[f'{nametosum}{n}'].append(dq_dbeta_me)

    fwd_v_alpha_il[f'{nametosum}{n}'].append(dq_dalpha_il)
    fwd_v_beta_il[f'{nametosum}{n}'].append(dq_dbeta_il)

    results_dot_alpha.append(dot_alpha_il)
    results_dot_beta.append(dot_beta_il)

## Now we get the rest of the information for sigma Q plot
qoi_dict_il = graphics.get_data_for_sigma_path_from_toml(tomlf_workflow, main_dir_path=Path(MAIN_PATH))
qoi_dict_m = graphics.get_data_for_sigma_path_from_toml(tomlf_workflow_m, main_dir_path=Path(MAIN_PATH))

print('We get x and sigma_t (years) to interpolate')
print(qoi_dict_il.keys())

dot_alpha_line = np.interp(qoi_dict_il['x'], qoi_dict_il['sigma_t'], results_dot_alpha)
dot_beta_line = np.interp(qoi_dict_il['x'], qoi_dict_il['sigma_t'], results_dot_beta)

########### Now we plot things ####################################
color_palette = sns.color_palette("deep")

rcParams['axes.labelsize'] = 10
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 5
sns.set_context('poster')

fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(1, 1, 1)

p1, = ax.plot(qoi_dict_il['x'], np.abs(qoi_dict_il['y']-qoi_dict_m['y']),
              linestyle='dashed', color=color_palette[3], label='', linewidth=3)
p2, = ax.plot(qoi_dict_il['x'], dot_alpha_line, color=color_palette[0], label='', linewidth=3)
p3, = ax.plot(qoi_dict_il['x'], dot_beta_line, color=color_palette[1], label='', linewidth=3)
ax.axhline(y=0, color='k', linewidth=2.5)
plt.legend(handles = [p1, p2, p3],
           labels = [r'$\Delta$ abs($VAF_{ITSLIVE}$ - $VAF_{MEaSUREs}$)',
                     r'$\delta Q / \delta \alpha$ . ($\alpha_{ITSLIVE}$ - $\alpha_{MEaSUREs}$)',
                     r'$\delta Q / \delta \beta$ . ($\beta_{ITSLIVE}$ - $\beta_{MEaSUREs}$)'],
           frameon=True, fontsize=18)

ax.set_ylabel(r'$\Delta$ abs(Q$_{I}$ - Q$_{M}$) [$m^3$]')
ax.set_xlabel('Time [yrs]')

plt.tight_layout()
plt.savefig(os.path.join(plot_path, 'linearity_test.png'), bbox_inches='tight')
