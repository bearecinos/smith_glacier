import sys
import argparse
import numpy as np
from scipy.interpolate import LinearNDInterpolator
import salem
import pyproj
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from configobj import ConfigObj
from fenics_ice import config as conf
from fenics_ice import mesh as fice_mesh

import matplotlib.ticker as ticker
from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.tri as tri


# Load configuration file for more order in paths
parser = argparse.ArgumentParser()
parser.add_argument("-conf", type=str, default="../../../config.ini", help="pass config file")
parser.add_argument("-toml_path", type=str, default="../run_experiments/run_workflow/smith_cloud.toml",
                    help="pass .toml file")

args = parser.parse_args()
config_file = args.conf
configuration = ConfigObj(os.path.expanduser(config_file))

# Define main repository path
MAIN_PATH = configuration['main_path']
sys.path.append(MAIN_PATH)

from ficetools import utils_funcs, graphics, velocity
from ficetools.backend import FunctionSpace, VectorFunctionSpace, Function, project

# Get the right toml
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
diag_il = params.io.diagnostics_dir
phase_suffix_il = params.inversion.phase_suffix

# This will be the same for both runs
phase_name = params.inversion.phase_name
run_name = params.io.run_name

exp_outdir = Path(diag_il) / phase_name / phase_suffix_il

file_u = "_".join((params.io.run_name+phase_suffix_il, 'U.xml'))
file_uvobs = "_".join((params.io.run_name+phase_suffix_il, 'uv_cloud.xml'))
file_alpha = "_".join((params.io.run_name+phase_suffix_il, 'alpha.xml'))
file_bglen = "_".join((params.io.run_name+phase_suffix_il, 'beta.xml'))
file_bed = "_".join((params.io.run_name+phase_suffix_il, 'bed.xml'))
file_h = "_".join((params.io.run_name+phase_suffix_il, 'thick.xml'))

U = exp_outdir / file_u
uv_obs = exp_outdir / file_uvobs
alpha = exp_outdir / file_alpha
bglen = exp_outdir / file_bglen
bed = exp_outdir / file_bed
H = exp_outdir / file_h

assert U.is_file(), "File not found"
assert uv_obs.is_file(), "File not found"
assert alpha.is_file(), "File not found"
assert bglen.is_file(), "File not found"
assert bed.is_file(), "File not found"
assert H.is_file(), "File not found"

# Define function spaces for alpha only and uv_comp
alpha_f = Function(Qp, str(alpha))
alpha_p = project(alpha_f, M)
alpha_v = alpha_p.compute_vertex_values(mesh_in)

beta_f = Function(Qp, str(bglen))
beta_p = project(beta_f, M)
beta_v = beta_p.compute_vertex_values(mesh_in)

# Velocity observations
uv = Function(M, str(uv_obs))
uv_obs_f = project(uv, Q)
uv_obs_v = uv_obs_f.compute_vertex_values(mesh_in)

# Model velocities
U_v =  utils_funcs.compute_vertex_for_velocity_field(str(U), V, Q, mesh_in)

#Bed
bed_f = Function(Qp, str(bed))
bed_p = project(bed_f, M)
bed_v = bed_p.compute_vertex_values(mesh_in)

#Thickness
thick_f = Function(M, str(H))
thick_p = project(thick_f, Q)
thick_v = thick_p.compute_vertex_values(mesh_in)

# Get mesh triangulation
x, y, t = graphics.read_fenics_ice_mesh(mesh_in)
trim = tri.Triangulation(x, y, t)

path_dan_grid = os.path.join(MAIN_PATH, 'scripts/prepro_stages/02_gridded_data/grid_for_bea.npy')

D = np.load(path_dan_grid)
X = D[0]
Y = D[1]

points = np.vstack(([x],[y]))

linTriFn = LinearNDInterpolator(points.T, alpha_v)

xi, yi = np.meshgrid(X, Y)

print('Only need to do interpolation')
zi_lin = linTriFn(xi, yi)

#print(zi_lin.shape)