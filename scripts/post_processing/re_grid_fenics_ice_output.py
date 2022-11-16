import sys
import argparse
import numpy as np
from scipy.interpolate import griddata
import h5py
import numpy.ma as ma
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

path_dan_grid = os.path.join(MAIN_PATH, 'scripts/post_processing/grid_for_bea.npy')

D = np.load(path_dan_grid)
X = D[0]
Y = D[1]

print('Only need to do the interpolations')
uv_obs_int = griddata((x, y), uv_obs_v.ravel(),
                   (X, Y),
                   method='nearest', fill_value=np.nan)

alpha_int = griddata((x, y), alpha_v.ravel(),
                   (X, Y),
                   method='nearest', fill_value=np.nan)

beta_int = griddata((x, y), beta_v.ravel(),
                   (X, Y),
                   method='nearest', fill_value=np.nan)

U_int = griddata((x, y), U_v.ravel(),
                   (X, Y),
                   method='nearest', fill_value=np.nan)

bed_int = griddata((x, y), bed_v.ravel(),
                   (X, Y),
                   method='nearest', fill_value=np.nan)

thick_int = griddata((x, y), thick_v.ravel(),
                   (X, Y),
                   method='nearest', fill_value=np.nan)

assert uv_obs_int.shape == X.shape
assert uv_obs_int.shape == Y.shape
assert alpha_int.shape == X.shape
assert alpha_int.shape == Y.shape
assert beta_int.shape == X.shape
assert beta_int.shape == Y.shape
assert U_int.shape == X.shape
assert U_int.shape == Y.shape
assert bed_int.shape == X.shape
assert bed_int.shape == Y.shape
assert thick_int.shape == X.shape
assert thick_int.shape == Y.shape

## Get rid of interpolated values in the ocean using bedmachine mask
input_dir = params.io.input_dir
file_bm = Path(input_dir) / params.io.bed_data_file

BM_file = h5py.File(file_bm, 'r')

bm_x = BM_file['x'][:]
bm_y = BM_file['y'][:]

bmxx, bmyy = np.meshgrid(bm_x, bm_y)

bm_surf = BM_file['surf'][:]

bm_surf_int = griddata((bmxx.ravel(), bmyy.ravel()), bm_surf.ravel(),
                       (X, Y), method='nearest', fill_value=np.nan)

array_new = ma.masked_where(bm_surf_int == 0, bm_surf_int)

uv_obs_int[array_new.mask] = np.nan
alpha_int[array_new.mask] = np.nan
beta_int[array_new.mask] = np.nan
U_int[array_new.mask] = np.nan
bed_int[array_new.mask] = np.nan
thick_int[array_new.mask] = np.nan

output_local_dir = os.path.join(MAIN_PATH, 'output/07_post_processing_model/'+ phase_suffix_il)
if not os.path.exists(output_local_dir):
    os.makedirs(output_local_dir)

np.savez(os.path.join(output_local_dir, 'fenics_ice_output_gridded'),
         X=X,
         Y=Y,
         vel_obs=uv_obs_int,
         vel_model=U_int,
         alpha=alpha_int,
         beta=beta_int,
         bed=bed_int,
         thick=thick_int)
