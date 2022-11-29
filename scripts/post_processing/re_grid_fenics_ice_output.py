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
file_alpha = "_".join((params.io.run_name+phase_suffix_il, 'alpha.xml'))
file_bglen = "_".join((params.io.run_name+phase_suffix_il, 'beta.xml'))
file_float_c1 = "_".join((params.io.run_name + params.inversion.phase_suffix, 'float.xml'))

U = exp_outdir / file_u
alpha = exp_outdir / file_alpha
bglen = exp_outdir / file_bglen
path_float_c1 = exp_outdir / file_float_c1

assert U.is_file(), "File not found"
assert alpha.is_file(), "File not found"
assert bglen.is_file(), "File not found"
assert path_float_c1.is_file(), "File not found"

# Define function spaces for alpha only and uv_comp
alpha_f = Function(Qp, str(alpha))
alpha_p = project(alpha_f, M)
alpha_v = alpha_p.compute_vertex_values(mesh_in)

beta_f = Function(Qp, str(bglen))
beta_p = project(beta_f, M)
beta_v = beta_p.compute_vertex_values(mesh_in)

# Model velocities
U_v =  utils_funcs.compute_vertex_for_velocity_field(str(U), V, Q, mesh_in)

# Where the floating ice ise in Fenics_ice
float_fun_c1 = Function(M, str(path_float_c1))
float_pro_c1 = project(float_fun_c1, M)
float_v_c1 = float_pro_c1.compute_vertex_values(mesh_in)

# Get mesh triangulation
x, y, t = graphics.read_fenics_ice_mesh(mesh_in)

alpha_v[float_v_c1 > 0] = 0


# We load Bedmachine already cropped to the smith glacier domain
path_bm = Path(os.path.join(params.io.input_dir, 'smith_bedmachine.h5'))
f_bm = h5py.File(path_bm, 'r')
x_bm = f_bm['x'][:]
y_bm = f_bm['y'][:]

surf = f_bm['surf'][:]
thick = f_bm['thick'][:]

bed = f_bm['bed'][:]

x_grid_bm, y_grid_bm = np.meshgrid(x_bm, y_bm)

### We load now itslive original data
path_vel = Path(os.path.join(params.io.input_dir, params.obs.vel_file))
f_vel = h5py.File(path_vel, 'r')
x_cloud = f_vel['x_cloud'][:]
y_cloud = f_vel['y_cloud'][:]

u_cloud = f_vel['u_cloud'][:]
v_cloud = f_vel['v_cloud'][:]

u_cloud_std = f_vel['u_cloud_std'][:]
v_cloud_std = f_vel['v_cloud_std'][:]

path_dan_grid = os.path.join(MAIN_PATH, 'scripts/post_processing/grid_for_bea_500.npy')

D = np.load(path_dan_grid)
X = D[0]
Y = D[1]

# Interpolate bed machine stuff
bed_int = griddata((x_grid_bm.ravel(), y_grid_bm.ravel()), bed.ravel(),
                   (X, Y),
                   method='nearest', fill_value=np.nan)
thick_int = griddata((x_grid_bm.ravel(), y_grid_bm.ravel()), thick.ravel(),
                   (X, Y),
                   method='nearest', fill_value=np.nan)
surf_int = griddata((x_grid_bm.ravel(), y_grid_bm.ravel()), surf.ravel(),
                   (X, Y),
                   method='nearest', fill_value=np.nan)

array_new = ma.masked_where(surf_int == 0, surf_int)

# Interpolate itslive stuff
u_obs_int = griddata((x_cloud, y_cloud), u_cloud,
                   (X, Y),
                   method='nearest', fill_value=np.nan)

v_obs_int = griddata((x_cloud, y_cloud), v_cloud,
                   (X, Y),
                   method='nearest', fill_value=np.nan)

u_std_int = griddata((x_cloud, y_cloud), u_cloud_std,
                   (X, Y),
                   method='nearest', fill_value=np.nan)

v_std_int = griddata((x_cloud, y_cloud), v_cloud_std,
                   (X, Y),
                   method='nearest', fill_value=np.nan)

u_obs_int[array_new.mask] = np.nan
v_obs_int[array_new.mask] = np.nan
u_std_int[array_new.mask] = np.nan
v_std_int[array_new.mask] = np.nan

vv = (u_obs_int**2 + v_obs_int**2)**0.5
vv[array_new.mask] = np.nan

## Now we interpolate fenics_ice data
alpha_int = griddata((x, y), alpha_v.ravel(),
                   (X, Y),
                   method='nearest', fill_value=np.nan)

beta_int = griddata((x, y), beta_v.ravel(),
                   (X, Y),
                   method='nearest', fill_value=np.nan)

U_int = griddata((x, y), U_v.ravel(),
                   (X, Y),
                   method='nearest', fill_value=np.nan)

alpha_int[array_new.mask] = np.nan
beta_int[array_new.mask] = np.nan
U_int[array_new.mask] = np.nan

assert vv.shape == X.shape
assert vv.shape == Y.shape
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
assert surf_int.shape == X.shape
assert surf_int.shape == Y.shape


output_local_dir = os.path.join(MAIN_PATH, 'output/07_post_processing_model/'+ phase_suffix_il)
if not os.path.exists(output_local_dir):
    os.makedirs(output_local_dir)

np.savez(os.path.join(output_local_dir, 'fenics_ice_output_gridded_500'),
         X=X,
         Y=Y,
         vel_obs=vv,
         u_obs = u_obs_int,
         v_obs = v_obs_int,
         u_obs_std = u_std_int,
         v_obs_std = v_std_int,
         vel_model=U_int,
         alpha=alpha_int,
         beta=beta_int,
         bed=bed_int,
         thick=thick_int,
         surf=surf_int)
