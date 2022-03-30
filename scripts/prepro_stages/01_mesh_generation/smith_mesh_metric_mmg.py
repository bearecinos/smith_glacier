"""
Produce a metric which can be used by gmsh to make a good quality variable
mesh for Smith Glacier.

Steps:

  - Loads velocity data & computes eigenstrains
  - Combines these patchy strain maps into full coverage
  - Produces a metric from the strain
  - Loads a mask indicating ice/ocean/rock
  - Polygonizes mask raster into ice/ocean (ignores rock)
  - Labels boundaries of ice polygon (calving front or natural)
  - Uses MMG (subprocess) to adapt mesh to metric
  - Produces FEniCS-ready Mesh & MeshValueCollection

Possible future work:

  - Other options for metric:
       non-linear strain dependence
       proximity to calving front
  - Generalise extent definition (Smith specific at present)
  - Package methods into a class?
"""
import os
import sys
import numpy as np
from netCDF4 import Dataset as NCDataset
import gmsh
import meshio
from configobj import ConfigObj
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-conf", type=str, default="../../../config.ini", help="pass config file")
args = parser.parse_args()
config_file = args.conf
config = ConfigObj(os.path.expanduser(config_file))

# Main directory path
MAIN_PATH = config['main_path']
sys.path.append(MAIN_PATH)

from ficetools import mesh as meshtools

# In files
velocity_netcdf = config['velocity_ase_series']
#mask_h5 = fice_source_dir/"input/smith_500m_input/smith_geom.h5"
bedmachine = config['bedmachine']
use_bedmachine = True

# Out files
output_path = os.path.join(MAIN_PATH,
                            'output/01_mesh')
if not os.path.exists(output_path):
    os.makedirs(output_path)

mesh_outfile = os.path.join(output_path, 'smith_variable_ocean')

lc_params = {'min': 500.0,
             'max': 4000.0,
             'strain_scale': 30.0}

mesh_extent = {'xmin': -1608000.0,
               'xmax': -1382500.0,
               'ymin': -717500.0,
               'ymax': -528500.0}

meshtools.set_mmg_lib(config['mmg_path'])

###############################################################
# Load the velocity data, compute eigenstrain, produce metric
###############################################################

vel_data = NCDataset(velocity_netcdf)
years = meshtools.get_netcdf_vel_years(vel_data)
dx = meshtools.get_dx(vel_data)
vel_xx, vel_yy = meshtools.get_netcdf_coords(vel_data)

# Get the nx*ny*2*2 strain matrices for each year & compute coverage
strain_mats = {}
for year in years:
    vx = vel_data.variables[f'vx{year}'][:]
    vy = vel_data.variables[f'vy{year}'][:]
    strain_mats[year] = meshtools.get_eigenstrain_rate(vx, vy, dx)

# Stack into 1 array and take the mean through time
all_strain_mats = np.stack(list(strain_mats.values()))
mean_strain_mat = np.nanmean(all_strain_mats, axis=0)

# Compute sum of absolute value of eigenvalues
eigsum = np.sum(np.abs(mean_strain_mat), axis=2)

# Turn it into a metric
metric = meshtools.simple_strain_metric(eigsum, lc_params)

###############################################################
# Polygonize mask raster, generate gmsh domain
###############################################################

if use_bedmachine:
    # Get the mask from BedMachine_Antarctica
    bedmachine_data = NCDataset(bedmachine)
    mask, mask_transform = meshtools.slice_netcdf(bedmachine_data, 'mask', mesh_extent)

    # mask: [0 = ocean, 1 = ice-free land, 2 = grounded ice, 3 = float ice, 4 = Lake Vostok]
    # We want to ignore nunataks (1) and just impose min thick, and at this stage we don't
    # care about grounded/floating, so:
    #
    # 0 -> 0
    # 1,2,3,4 -> 1
    mask = (mask >= 1).astype(np.int)

else:

    # Previously just read from smith geom in fenics_ice repo
    mask = meshtools.get_smith_ice_mask(mask_h5, resolve_nunataks=False)
    mask_transform = meshtools.get_affine_transform(mask_h5)

gmsh_ring, ice_labels, ocean_labels = meshtools.generate_boundary(mask,
                                                                  mask_transform,
                                                                  simplify_tol=1.0e3,
                                                                  bbox=mesh_extent)

ice_tag, ocean_tags = meshtools.build_gmsh_domain(gmsh_ring, ice_labels, ocean_labels)

meshtools.tags_to_file({'ice': [ice_tag], 'ocean': ocean_tags}, mesh_outfile+"_BCs.txt")

# Create the (not yet adapted) mesh
gmsh.model.mesh.generate(2)
gmsh.write(mesh_outfile+".msh")


# Add the post-processing data (the metric) via interpolation
interped_metric = meshtools.interp_to_gmsh(metric, vel_xx, vel_yy)
meshtools.write_medit_sol(interped_metric, mesh_outfile+".sol")

gmsh.finalize()

###############################################################
# Adapt with MMG, then produce FEniCS ready Mesh and MeshValueCollection
###############################################################

meshtools.gmsh_to_medit(mesh_outfile+".msh", mesh_outfile+".mesh")

# Adapt with MMG
meshtools.run_mmg_adapt(mesh_outfile+".mesh", mesh_outfile+".sol", hgrad=1.3, hausd=100.0)
meshtools.remove_medit_corners(mesh_outfile+".o.mesh")

# This 'mixed' mesh works fine for MMG (tris and lines), but fenics
# can't handle it. Need to feed in a triangle-only mesh and a facet-function.
adapted_mesh = meshio.read(mesh_outfile+".o.mesh")


# Extract the pts & triangle elements, write to file
fenics_mesh = meshtools.extract_tri_mesh(adapted_mesh)
meshio.write(mesh_outfile+".xdmf", fenics_mesh)

fmesh = meshtools.load_fenics_mesh(mesh_outfile+".xdmf")
mvc = meshtools.lines_to_mvc(adapted_mesh, fmesh, marker_name="medit:ref")

meshtools.write_mvc(mvc, mesh_outfile+"_ff.xdmf")

# Clear up intermediate files
meshtools.delete_intermediates(mesh_outfile)
